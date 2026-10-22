"""Central configuration for PlaceBot.

All values come from environment variables (optionally loaded from a .env file) with safe
defaults. No secrets are hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time so tests run in minimal environments
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # Paths
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    sample_dir: Path = PROJECT_ROOT / "data" / "sample"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("PLACEBOT_LLM_PROVIDER", "mock").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("PLACEBOT_LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("PLACEBOT_LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("PLACEBOT_LLM_BASE_URL", ""))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("PLACEBOT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Chunking
    chunk_target_tokens: int = field(default_factory=lambda: _int("PLACEBOT_CHUNK_TARGET_TOKENS", 300))
    chunk_max_tokens: int = field(default_factory=lambda: _int("PLACEBOT_CHUNK_MAX_TOKENS", 400))
    chunk_overlap_tokens: int = field(default_factory=lambda: _int("PLACEBOT_CHUNK_OVERLAP_TOKENS", 40))

    # Retrieval
    top_k: int = field(default_factory=lambda: _int("PLACEBOT_TOP_K", 6))
    candidate_k: int = field(default_factory=lambda: _int("PLACEBOT_CANDIDATE_K", 20))
    rrf_k: int = field(default_factory=lambda: _int("PLACEBOT_RRF_K", 60))
    context_token_budget: int = field(default_factory=lambda: _int("PLACEBOT_CONTEXT_TOKEN_BUDGET", 1800))
    min_bm25_score: float = field(default_factory=lambda: _float("PLACEBOT_MIN_BM25_SCORE", 0.5))
    min_vector_score: float = field(default_factory=lambda: _float("PLACEBOT_MIN_VECTOR_SCORE", 0.20))
    min_term_coverage: float = field(default_factory=lambda: _float("PLACEBOT_MIN_TERM_COVERAGE", 0.40))

    # Generation
    llm_max_tokens: int = field(default_factory=lambda: _int("PLACEBOT_LLM_MAX_TOKENS", 700))
    llm_timeout_s: int = field(default_factory=lambda: _int("PLACEBOT_LLM_TIMEOUT_S", 60))

    # Privacy: log only source provenance (never student profile data or raw question text)
    log_answers: bool = field(default_factory=lambda: os.getenv("PLACEBOT_LOG_ANSWERS", "true").lower() in ("1", "true", "yes"))

    # Documents
    max_upload_mb: int = field(default_factory=lambda: _int("PLACEBOT_MAX_UPLOAD_MB", 20))

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.sample_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a fresh Settings object (re-reads the environment)."""
    return Settings()
