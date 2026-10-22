"""Shared test helpers (plain functions, cached, so tests also run outside pytest fixtures)."""
from __future__ import annotations

import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from src.ingestion.pipeline import ingest_directory  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.vector_search import HashingEmbedder  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "sample"


@functools.lru_cache(maxsize=1)
def sample_corpus():
    """(records, chunks) from the synthetic sample PDFs."""
    records, chunks = ingest_directory(SAMPLE_DIR, get_settings())
    synthetic_ids = {r.doc_id for r in records if r.synthetic}
    return (
        [r for r in records if r.doc_id in synthetic_ids],
        [c for c in chunks if c.doc_id in synthetic_ids],
    )


@functools.lru_cache(maxsize=1)
def retriever() -> HybridRetriever:
    """Retriever using the offline hashing embedder so tests never need model downloads."""
    _, chunks = sample_corpus()
    return HybridRetriever(chunks, embedder=HashingEmbedder())
