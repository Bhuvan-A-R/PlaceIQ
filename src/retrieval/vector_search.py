"""Semantic retrieval: embeddings + vector index (FAISS if installed, else NumPy).

Embedder selection:
* ``SentenceTransformerEmbedder`` - real semantic embeddings (preferred).
* ``HashingEmbedder`` - offline fallback when sentence-transformers or the model weights are
  unavailable. It is *lexical* (hashed word/bigram features), NOT truly semantic. The active
  embedder name is exposed via ``.name`` so the UI and evaluation report can disclose it.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Optional, Protocol, Sequence

import numpy as np

from config.settings import Settings, get_settings
from src.retrieval.text_utils import index_text, tokenize
from src.schemas import Chunk

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # L2-normalised float32 (n, d)
        ...


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class HashingEmbedder:
    """Deterministic hashed bag-of-words/bigrams embedder (no downloads, no training)."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"hashing-fallback-{dim}"

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            toks = tokenize(text, remove_stopwords=True)
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for a, b in zip(toks, toks[1:]):
                counts[f"{a}_{b}"] = counts.get(f"{a}_{b}", 0) + 1
            for feat, c in counts.items():
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * (1.0 + math.log(c))
        return _l2(out)


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return _l2(np.asarray(vecs, dtype="float32"))


def get_embedder(settings: Optional[Settings] = None, force_fallback: bool = False) -> Embedder:
    """Return the best available embedder, logging a warning when falling back."""
    s = settings or get_settings()
    if not force_fallback:
        try:
            return SentenceTransformerEmbedder(s.embedding_model)
        except Exception as exc:  # ImportError, offline model download failure, etc.
            log.warning("sentence-transformers unavailable (%s); using hashing fallback embedder", exc)
    return HashingEmbedder()


class VectorIndex:
    """Cosine-similarity index over chunk embeddings."""

    # Cosines at or below this are treated as noise and not returned. This is a candidate
    # filter only; the answer/not-found decision uses Settings.min_vector_score separately.
    NOISE_FLOOR = 0.05

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [index_text(c.company, c.section, c.text) for c in self.chunks]
        self.matrix = embedder.encode(texts) if texts else np.zeros((0, 1), dtype="float32")
        self.backend = "numpy"
        self._faiss = None
        try:  # optional acceleration
            import faiss  # type: ignore

            if len(self.chunks):
                idx = faiss.IndexFlatIP(self.matrix.shape[1])
                idx.add(self.matrix)
                self._faiss, self.backend = idx, "faiss"
        except ImportError:
            pass

    def search(
        self, query: str, k: int = 20, allowed: Optional[set[int]] = None
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk_index, cosine) pairs, best first. Scores are signals only."""
        if not len(self.chunks):
            return []
        q = self.embedder.encode([query])
        if self._faiss is not None and allowed is None:
            scores, ids = self._faiss.search(q, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0 and s > self.NOISE_FLOOR]
        sims = (self.matrix @ q[0]).astype("float64")
        order = np.argsort(-sims)
        out: list[tuple[int, float]] = []
        for i in order:
            if allowed is not None and int(i) not in allowed:
                continue
            if sims[i] <= self.NOISE_FLOOR:  # hash collisions / unrelated text give tiny positive cosines
                break
            out.append((int(i), float(sims[i])))
            if len(out) >= k:
                break
        return out
