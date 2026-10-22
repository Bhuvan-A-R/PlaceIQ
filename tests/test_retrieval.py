"""Tests: BM25, vector index, RRF, metadata filters, evidence checks, context, reranking."""
from __future__ import annotations

from conftest import retriever, sample_corpus

from src.retrieval.bm25_search import BM25Index
from src.retrieval.context import assemble_context
from src.retrieval.hybrid_search import reciprocal_rank_fusion, term_coverage
from src.retrieval.reranker import detect_companies, normalize_company
from src.retrieval.text_utils import informative_terms, tokenize
from src.retrieval.vector_search import HashingEmbedder, VectorIndex
from src.schemas import RetrievedChunk


# ------------------------------------------------------------------------ RRF
def test_rrf_scores_match_formula():
    s = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    assert abs(s[1] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(s[2] - 1 / 62) < 1e-12
    assert abs(s[3] - (1 / 63 + 1 / 61)) < 1e-12


def test_rrf_item_in_both_lists_beats_item_in_one():
    s = reciprocal_rank_fusion([[10, 20, 30], [20, 40, 50]], k=60)
    best = max(s, key=s.get)
    assert best == 20


def test_rrf_constant_changes_gap_and_empty_input_ok():
    small = reciprocal_rank_fusion([[1, 2]], k=1)
    large = reciprocal_rank_fusion([[1, 2]], k=100)
    assert (small[1] - small[2]) > (large[1] - large[2])
    assert reciprocal_rank_fusion([], k=60) == {}
    assert reciprocal_rank_fusion([[]], k=60) == {}


# ----------------------------------------------------------------------- text
def test_tokenizer_keeps_decimals_and_stems_plurals():
    assert "6.5" in tokenize("minimum CGPA of 6.5")
    assert tokenize("backlogs") == tokenize("backlog")
    assert "the" not in informative_terms("what is the minimum CGPA")


# ----------------------------------------------------------------------- BM25
def test_bm25_finds_exact_numeric_term():
    _, chunks = sample_corpus()
    idx = BM25Index(chunks)
    hits = idx.search("6.5", k=5)
    assert hits
    top_text = chunks[hits[0][0]].text
    assert "6.5" in top_text


def test_bm25_respects_allowed_filter_and_drops_zero_scores():
    _, chunks = sample_corpus()
    idx = BM25Index(chunks)
    allowed = {i for i, c in enumerate(chunks) if c.doc_type == "placement_faq"}
    hits = idx.search("offer drive", k=10, allowed=allowed)
    assert hits and all(i in allowed for i, _ in hits)
    assert idx.search("zzzxqv", k=5) == []


# --------------------------------------------------------------------- vector
def test_vector_index_ranks_related_chunk_first_and_scores_in_range():
    _, chunks = sample_corpus()
    vi = VectorIndex(chunks, HashingEmbedder())
    hits = vi.search("bond period months", k=3)
    assert hits and all(-1.0 <= sc <= 1.0001 for _, sc in hits)
    assert "bond" in chunks[hits[0][0]].text.lower()


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.encode(["minimum cgpa"]), e.encode(["minimum cgpa"])
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


# ------------------------------------------------------------- hybrid + filters
def test_hybrid_direct_question_hits_expected_document():
    res = retriever().search("What is the minimum CGPA for Nimbus Soft?")
    assert res.evidence_sufficient
    top = res.results[0].chunk
    assert top.source_filename == "nimbus_soft_eligibility_SYNTHETIC.pdf"
    assert "Minimum CGPA | 7.0" in top.text


def test_company_filter_is_auto_detected_and_excludes_other_companies():
    res = retriever().search("What are Orbit Analytics selection rounds?")
    assert res.filters["company"].startswith("Orbit Analytics")
    for r in res.results:
        assert r.chunk.company is None or r.chunk.company.startswith("Orbit Analytics")
    assert res.results[0].chunk.section.endswith("Selection Rounds")


def test_explicit_company_filter_without_general_docs():
    res = retriever().search("bond period", company="Quanta Grid Systems (synthetic)", include_general=False)
    assert res.results and all(r.chunk.company for r in res.results)
    assert all("Quanta" in r.chunk.company for r in res.results)


def test_doc_type_filter():
    res = retriever().search("offer", doc_types=["placement_faq"])
    assert res.results and all(r.chunk.doc_type == "placement_faq" for r in res.results)


def test_policy_question_returns_policy_clause_with_page():
    res = retriever().search("Can a student attend another drive after a dream offer?")
    assert res.evidence_sufficient
    top = res.results[0].chunk
    assert top.doc_type == "placement_policy" and "3.2" in top.text and top.page_label == "2"


def test_rephrased_question_still_retrieves_company_eligibility():
    res = retriever().search("I have one active backlog. Can I apply to Orbit Analytics?")
    assert res.evidence_sufficient
    assert any("Active backlogs | 1" in r.chunk.text for r in res.results)


def test_unanswerable_question_is_flagged_as_insufficient():
    res = retriever().search("What was the average package of last year's CSE batch?")
    assert not res.evidence_sufficient
    assert res.reason


def test_nonsense_query_is_insufficient_even_if_fallback_embedder_returns_noise():
    # The hashing fallback can return low-similarity collisions; the term-coverage check
    # must still reject the query. (The contract is "not sufficient", not "no results".)
    res = retriever().search("zzzxqv qqqwww")
    assert not res.evidence_sufficient
    assert res.term_coverage == 0.0 or not res.results
    assert all(r.bm25_score is None for r in res.results)  # nothing matched on keywords


def test_search_per_company_runs_independent_retrievals():
    r = retriever()
    out = r.search_per_company("minimum CGPA", r.companies, doc_types=["eligibility_circular"])
    assert set(out) == set(r.companies)
    for comp, res in out.items():
        assert res.results and all(x.chunk.company == comp for x in res.results)


def test_all_retrieved_chunks_exist_in_corpus():
    _, chunks = sample_corpus()
    valid = {c.chunk_id for c in chunks}
    for q in ["bond period", "selection rounds", "debarment", "CGPA 6.5", "withdraw"]:
        for r in retriever().search(q).results:
            assert r.chunk.chunk_id in valid


# ------------------------------------------------------------------- reranker
def test_conflicting_circular_versions_are_both_kept_and_annotated():
    res = retriever().search("minimum CGPA", company="Quanta Grid Systems (synthetic)",
                             doc_types=["eligibility_circular"], include_general=False)
    files = {r.chunk.source_filename for r in res.results}
    assert "quanta_grid_eligibility_SYNTHETIC.pdf" in files
    assert "quanta_grid_eligibility_revised_SYNTHETIC.pdf" in files  # old one is not silently dropped
    notes = " ".join(n for r in res.results for n in r.notes)
    assert "older version" in notes and "multiple versions" in notes
    newest = [r for r in res.results if "latest version" in " ".join(r.notes)]
    assert newest and all(r.chunk.effective_date == "2025-10-15" for r in newest)


def test_company_detection_and_normalisation():
    comps = ["Nimbus Soft (synthetic)", "Orbit Analytics (synthetic)"]
    assert detect_companies("tell me about nimbus", comps) == ["Nimbus Soft (synthetic)"]
    assert detect_companies("compare Nimbus Soft and Orbit Analytics", comps) == comps
    assert detect_companies("what is the policy", comps) == []
    assert normalize_company("Nimbus Soft (synthetic)") == "nimbus soft"


# -------------------------------------------------------------------- context
def test_term_coverage_bounds():
    assert term_coverage("minimum cgpa", ["Minimum CGPA 7.0"]) == 1.0
    assert term_coverage("minimum cgpa", ["nothing relevant"]) == 0.0
    assert term_coverage("the", ["anything"]) == 0.0


def test_context_assembly_dedupes_and_respects_budget_and_labels():
    res = retriever().search("minimum CGPA eligibility", top_k=6)
    doubled = res.results + res.results  # duplicates must be removed
    bundle = assemble_context(doubled, token_budget=10_000)
    ids = [rc.chunk.chunk_id for _, rc in bundle.items]
    assert len(ids) == len(set(ids))
    assert bundle.text.count("[S1]") == 1
    assert all(f"file={rc.chunk.source_filename}" in bundle.text and f"page={rc.chunk.page_label}" in bundle.text
               for _, rc in bundle.items)

    tiny = assemble_context(res.results, token_budget=60)
    assert len(tiny.items) >= 1  # always keeps the best chunk
    assert tiny.dropped_for_budget >= 1
    assert set(tiny.label_to_chunk_id().values()) <= {rc.chunk.chunk_id for rc in res.results}
