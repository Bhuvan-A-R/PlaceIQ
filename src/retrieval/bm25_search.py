"""Keyword retrieval with BM25 (rank-bm25 if installed, otherwise an equivalent built-in)."""
from __future__ import annotations

import math
from typing import Optional, Sequence

from src.retrieval.text_utils import index_text, tokenize
from src.schemas import Chunk

try:  # preferred backend per project spec
    from rank_bm25 import BM25Okapi as _RankBM25
except ImportError:  # pragma: no cover - exercised when rank-bm25 is absent
    _RankBM25 = None


class _BuiltinBM25Okapi:
    """Minimal BM25Okapi matching rank_bm25's formulation (k1=1.5, b=0.75, epsilon=0.25)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / max(self.corpus_size, 1)
        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in corpus:
            freqs: dict[str, int] = {}
            for w in doc:
                freqs[w] = freqs.get(w, 0) + 1
            self.doc_freqs.append(freqs)
            for w in freqs:
                df[w] = df.get(w, 0) + 1
        self.idf: dict[str, float] = {}
        idf_sum, negatives = 0.0, []
        for w, n in df.items():
            v = math.log(self.corpus_size - n + 0.5) - math.log(n + 0.5)
            self.idf[w] = v
            idf_sum += v
            if v < 0:
                negatives.append(w)
        self.average_idf = idf_sum / max(len(self.idf), 1)
        floor = self.epsilon * self.average_idf
        for w in negatives:
            self.idf[w] = floor

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(q, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


class BM25Index:
    """BM25 index over chunks. Scores are relative retrieval signals, not probabilities."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._tokens = [tokenize(index_text(c.company, c.section, c.text)) for c in self.chunks]
        impl = _RankBM25 or _BuiltinBM25Okapi
        self.backend = "rank-bm25" if _RankBM25 else "builtin-bm25"
        self._bm25 = impl(self._tokens) if self._tokens else None

    def search(
        self, query: str, k: int = 20, allowed: Optional[set[int]] = None
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk_index, score) pairs, best first, score > 0 only."""
        if not self._bm25:
            return []
        q = tokenize(query, remove_stopwords=True)
        if not q:
            return []
        scores = list(self._bm25.get_scores(q))
        numeric_terms = [term for term in q if any(ch.isdigit() for ch in term)]
        order = sorted(
            range(len(scores)),
            key=lambda i: (
                any(term in self.chunks[i].text.lower() for term in numeric_terms),
                scores[i],
            ),
            reverse=True,
        )
        out: list[tuple[int, float]] = []
        for i in order:
            if allowed is not None and i not in allowed:
                continue
            if scores[i] <= 0:
                break
            out.append((i, float(scores[i])))
            if len(out) >= k:
                break
        return out
