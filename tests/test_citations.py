"""Citation integrity, unanswerable questions, chatbot behaviour, evaluation harness."""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("PLACEBOT_LOG_ANSWERS", "false")

from conftest import retriever, sample_corpus

from config.settings import get_settings
from src.evaluation.evaluate import evaluate_one, load_questions, run_evaluation, summarize
from src.generation.chatbot import Citation, PlaceBot
from src.generation.citation_validator import strip_invalid_citations, validate_answer, verify_citation_record
from src.generation.llm_client import MockLLM
from src.generation.prompts import NOT_FOUND_MESSAGE
from src.generation.query_rewriter import normalize_numbers, rewrite_query


def _bot() -> PlaceBot:
    s = dataclasses.replace(get_settings(), log_answers=False)
    return PlaceBot(sample_corpus()[1], s, retriever=retriever(), llm=MockLLM())


def _corpus():
    return {c.chunk_id: c for c in sample_corpus()[1]}


# ------------------------------------------------------------------ validator unit tests
def test_validator_accepts_supported_cited_numbers():
    corpus = _corpus()
    c = next(c for c in corpus.values() if "Minimum CGPA | 7.0" in c.text)
    rep = validate_answer("The minimum CGPA is 7.0 [S1].", {"S1": c.chunk_id}, corpus)
    assert rep.ok


def test_validator_flags_label_not_in_context():
    corpus = _corpus()
    c = next(iter(corpus.values()))
    rep = validate_answer("Something [S1] and more [S7].", {"S1": c.chunk_id}, corpus)
    assert rep.invalid_labels == ["S7"] and not rep.citations_ok


def test_validator_flags_chunk_missing_from_corpus():
    rep = validate_answer("Fact [S1].", {"S1": "ghost-chunk"}, _corpus())
    assert rep.missing_chunks == ["S1"]


def test_validator_flags_number_not_in_cited_chunk():
    corpus = _corpus()
    c = next(c for c in corpus.values() if "Minimum CGPA | 7.0" in c.text)
    rep = validate_answer("The minimum CGPA is 9.5 [S1].", {"S1": c.chunk_id}, corpus)
    assert rep.unsupported_numbers and not rep.claims_ok


def test_validator_flags_uncited_numeric_claim():
    corpus = _corpus()
    c = next(iter(corpus.values()))
    rep = validate_answer("The CTC is 40 LPA.", {"S1": c.chunk_id}, corpus)
    assert rep.uncited_numeric and not rep.ok


def test_strip_invalid_citations_removes_only_bad_labels():
    assert strip_invalid_citations("a [S1] b [S9]", ["S9"]) == "a [S1] b "


def test_verify_citation_record_rejects_fabricated_page_or_file():
    corpus = _corpus()
    c = next(iter(corpus.values()))
    assert verify_citation_record(c.chunk_id, c.source_filename, c.page_label, corpus)
    assert not verify_citation_record(c.chunk_id, c.source_filename, "99", corpus)
    assert not verify_citation_record(c.chunk_id, "fake.pdf", c.page_label, corpus)
    assert not verify_citation_record("nope", c.source_filename, c.page_label, corpus)


# ---------------------------------------------------------------- chatbot integrity
def test_answer_citations_are_real_indexed_passages():
    corpus = _corpus()
    ans = _bot().ask("What is the minimum CGPA for Nimbus Soft?")
    assert ans.found and ans.citations
    for c in ans.citations:
        assert c.chunk_id in corpus
        assert verify_citation_record(c.chunk_id, c.filename, c.page, corpus)
        assert c.passage == corpus[c.chunk_id].text
    assert "7.0" in ans.text
    assert ans.validation.ok


def test_mock_mode_is_labelled_and_never_claims_llm():
    ans = _bot().ask("What is the CTC for Nimbus Soft?")
    assert "mock" in ans.mode and "no LLM" in ans.mode
    assert any("no language model" in w.lower() for w in ans.warnings)


def test_unanswerable_question_returns_not_found_without_citations():
    ans = _bot().ask("What was the average package of last year's CSE batch?")
    assert ans.status == "not_found" and ans.text == NOT_FOUND_MESSAGE and not ans.citations


def test_other_unanswerable_questions():
    for q in ("What is the hostel fee at the campus?", "Does Nimbus Soft offer stock options to freshers?"):
        ans = _bot().ask(q)
        assert ans.status == "not_found", q


def test_policy_answer_contains_clause_number_and_policy_source():
    ans = _bot().ask_policy("Can a student attend another drive after a dream offer?")
    assert ans.found and "3.2" in ans.text
    assert any(c.doc_type == "placement_policy" for c in ans.citations)


def test_policy_mode_only_uses_policy_and_faq_documents():
    ans = _bot().ask_policy("What are the rules for debarment?")
    assert ans.citations and all(c.doc_type in ("placement_policy", "placement_faq") for c in ans.citations)


def test_follow_up_resolves_company_from_history():
    bot = _bot()
    hist = [("user", "What is the bond period for Quanta Grid?"), ("assistant", "24 months")]
    ans = bot.ask("And its minimum CGPA?", hist)
    assert ans.company and ans.company.startswith("Quanta")
    assert "Minimum CGPA" in ans.text


def test_rewriter_number_words_and_no_false_company_carryover():
    assert normalize_numbers("I have one active backlog") == "I have 1 active backlog"
    assert normalize_numbers("What is the one-offer rule?") == "What is the one-offer rule?"
    comps = ["Nimbus Soft (synthetic)"]
    rw = rewrite_query("What happens if I withdraw after the first round?",
                       [("user", "min CGPA for Nimbus Soft?")], comps)
    assert rw.company is None  # policy question must not inherit the company


def test_conflicting_versions_are_both_surfaced():
    ans = _bot().ask("What is the minimum CGPA for Quanta Grid Systems?")
    files = {c.filename for c in ans.citations}
    assert any("Multiple versions" in w for w in ans.warnings) or len(files) > 1


def test_cross_company_query_is_deterministic_and_states_scope():
    ans = _bot().ask("Which indexed companies allow EEE students with 6.5 CGPA?")
    assert ans.found and "deterministic" in ans.mode
    assert "Orbit Analytics" in ans.text and "Quanta Grid" in ans.text
    assert "only the 3 companies" in ans.text
    match = dict(zip(ans.table["Company"], ans.table["Match"]))
    assert match["Orbit Analytics (synthetic)"] == "Yes"
    assert match["Quanta Grid Systems (synthetic)"] == "Undetermined"  # conflicting circulars
    assert match["Nimbus Soft (synthetic)"] == "No"  # EEE not listed


def test_citation_from_chunk_round_trip():
    ch = next(iter(_corpus().values()))
    c = Citation.from_chunk("S1", ch)
    assert (c.filename, c.page, c.passage) == (ch.source_filename, ch.page_label, ch.text)


# --------------------------------------------------------------------- evaluation
def test_evaluation_dataset_covers_required_cases():
    qs = load_questions()
    cats = {q["category"] for q in qs}
    assert {"direct", "policy", "rephrased", "multi_document", "unanswerable"} <= cats
    assert len(qs) >= 6


def test_evaluation_runs_and_records_all_columns():
    df = run_evaluation(_bot())
    for col in ("query", "expected_source", "expected_evidence", "retrieved_top_k", "retrieval_hit",
                "answer_correct", "citation_correct", "unsupported_claims", "notes"):
        assert col in df.columns
    sm = summarize(df)
    assert sm["unanswerable_handled_rate"] == 1.0
    assert sm["unsupported_claim_count"] == 0
    assert sm["retrieval_hit_rate"] >= 0.8


def test_evaluate_one_detects_fabricated_answer_to_unanswerable():
    bot = _bot()
    q = {"id": "X", "category": "direct", "query": "What is the minimum CGPA for Nimbus Soft?", "company": "",
         "expected_source": "", "expected_evidence": "", "expected_answer_contains": "", "answerable": "no"}
    row = evaluate_one(bot, q)  # answerable question mislabelled as unanswerable -> flagged
    assert row["answer_correct"] is False and "possible fabrication" in row["notes"]
