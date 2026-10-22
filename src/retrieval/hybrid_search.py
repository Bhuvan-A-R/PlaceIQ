"""Hybrid retrieval: metadata filter -> BM25 + vector -> RRF -> rerank -> evidence check."""
from __future__ import annotations

from typing import Optional, Sequence

from config.settings import Settings, get_settings
from src.retrieval.bm25_search import BM25Index
from src.retrieval.reranker import detect_companies, normalize_company, rerank
from src.retrieval.text_utils import informative_terms, tokenize
from src.retrieval.vector_search import Embedder, VectorIndex, get_embedder
from src.schemas import Chunk, RetrievalResult, RetrievedChunk


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Combine several best-first rankings into one score per item.

    score(d) = sum over rankings of 1 / (k + rank), rank starting at 1.
    Items absent from a ranking simply receive no contribution from it.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def term_coverage(query: str, texts: Sequence[str]) -> float:
    """Fraction of informative query terms that appear in the given passages."""
    terms = informative_terms(query)
    if not terms:
        return 0.0
    vocab: set[str] = set()
    for t in texts:
        vocab.update(tokenize(t))
    return sum(1 for t in terms if t in vocab) / len(terms)


class HybridRetriever:
    """Owns the BM25 and vector indexes over a fixed chunk list."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        settings: Optional[Settings] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunks = list(chunks)
        self.embedder = embedder or get_embedder(self.settings)
        self.bm25 = BM25Index(self.chunks)
        self.vectors = VectorIndex(self.chunks, self.embedder)
        self.companies = sorted({c.company for c in self.chunks if c.company})

    # ------------------------------------------------------------------ filters
    def _allowed(
        self,
        company: Optional[str],
        doc_types: Optional[Sequence[str]],
        include_general: bool,
        chunk_ids: Optional[set[str]] = None,
    ) -> Optional[set[int]]:
        if not company and not doc_types and chunk_ids is None:
            return None
        want = normalize_company(company) if company else ""
        allowed: set[int] = set()
        for i, c in enumerate(self.chunks):
            if doc_types and c.doc_type not in doc_types:
                continue
            if chunk_ids is not None and c.chunk_id not in chunk_ids:
                continue
            if want:
                comp = normalize_company(c.company)
                if comp:
                    if comp != want and want not in comp:
                        continue
                elif not include_general:
                    continue
            allowed.add(i)
        return allowed

    @staticmethod
    def _strip_company_terms(query: str, company: str) -> str:
        name_tokens = set(tokenize(normalize_company(company)))
        kept = [w for w in query.split() if not (set(tokenize(w)) & name_tokens)]
        return " ".join(kept) if kept else query

    # ------------------------------------------------------------------- search
    def search(
        self,
        query: str,
        company: Optional[str] = None,
        doc_types: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        include_general: bool = True,
        auto_detect_company: bool = True,
    ) -> RetrievalResult:
        """Retrieve evidence for a query.

        If ``company`` is None and ``auto_detect_company`` is True, a company named in the query
        is used as a metadata filter (only when exactly one is detected).
        """
        s = self.settings
        k = top_k or s.top_k
        detected: list[str] = []
        if company is None and auto_detect_company:
            detected = detect_companies(query, self.companies)
            if len(detected) == 1:
                company = detected[0]
        allowed = self._allowed(company, doc_types, include_general)
        filters = {"company": company, "doc_types": list(doc_types) if doc_types else None,
                   "detected_companies": detected}

        # When a company filter is active every candidate already belongs to that company, so the
        # company's own name would only add noise to scoring; strip it from the scoring query.
        scoring_query = self._strip_company_terms(query, company) if company else query
        bm = self.bm25.search(scoring_query, k=s.candidate_k, allowed=allowed)
        ve = self.vectors.search(scoring_query, k=s.candidate_k, allowed=allowed)
        bm_rank = {idx: (r, sc) for r, (idx, sc) in enumerate(bm, start=1)}
        ve_rank = {idx: (r, sc) for r, (idx, sc) in enumerate(ve, start=1)}
        fused = reciprocal_rank_fusion([[i for i, _ in bm], [i for i, _ in ve]], k=s.rrf_k)

        retrieved: list[RetrievedChunk] = []
        for idx, score in fused.items():
            rc = RetrievedChunk(
                chunk=self.chunks[idx],
                rrf_score=score,
                bm25_score=bm_rank[idx][1] if idx in bm_rank else None,
                bm25_rank=bm_rank[idx][0] if idx in bm_rank else None,
                vector_score=ve_rank[idx][1] if idx in ve_rank else None,
                vector_rank=ve_rank[idx][0] if idx in ve_rank else None,
            )
            retrieved.append(rc)
        retrieved.sort(key=lambda r: r.rrf_score, reverse=True)
        retrieved = rerank(retrieved, [company] if company else detected)[:k]

        result = RetrievalResult(query=query, results=retrieved, filters=filters, embedder=self.embedder.name)
        self._judge_evidence(result)
        return result

    def search_per_company(
        self,
        query: str,
        companies: Sequence[str],
        doc_types: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, RetrievalResult]:
        """Retrieve separately for each company (for comparisons / cross-company questions)."""
        return {
            c: self.search(query, company=c, doc_types=doc_types, top_k=top_k, include_general=False,
                           auto_detect_company=False)
            for c in companies
        }

    # --------------------------------------------------------- evidence checks
    def _judge_evidence(self, result: RetrievalResult) -> None:
        """Set ``evidence_sufficient``. Scores are signals, not probabilities."""
        s = self.settings
        if not result.results:
            result.evidence_sufficient, result.reason = False, "no passages matched the query"
            return
        top = result.results[:3]
        cov = term_coverage(result.query, [r.chunk.text + " " + (r.chunk.section or "") + " " +
                                           (r.chunk.company or "") for r in top])
        result.term_coverage = cov
        best_bm25 = max((r.bm25_score or 0.0) for r in result.results)
        best_vec = max((r.vector_score or 0.0) for r in result.results)
        strong_signal = best_bm25 >= s.min_bm25_score or best_vec >= s.min_vector_score
        if not strong_signal:
            result.evidence_sufficient = False
            result.reason = (f"retrieval signals below thresholds (bm25={best_bm25:.2f}<{s.min_bm25_score}, "
                             f"vector={best_vec:.2f}<{s.min_vector_score})")
        elif cov < s.min_term_coverage:
            result.evidence_sufficient = False
            result.reason = (f"only {cov:.0%} of the query's key terms appear in the top passages "
                             f"(minimum {s.min_term_coverage:.0%})")
        else:
            result.evidence_sufficient = True
            result.reason = f"key-term coverage {cov:.0%}; best bm25={best_bm25:.2f}, best vector={best_vec:.2f}"
