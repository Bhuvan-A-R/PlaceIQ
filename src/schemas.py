"""Shared Pydantic data models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

DOC_TYPES = (
    "placement_policy",
    "company_jd",
    "eligibility_circular",
    "placement_faq",
    "recruitment_process",
    "unknown",
)


class PageText(BaseModel):
    """Extracted text of a single PDF page (1-indexed)."""

    page: int
    text: str


class DocumentRecord(BaseModel):
    """Registry entry for one ingested source document."""

    doc_id: str
    filename: str
    sha256: str
    doc_type: str = "unknown"
    company: Optional[str] = None
    effective_date: Optional[str] = None
    version: Optional[str] = None
    synthetic: bool = False
    num_pages: int = 0
    num_chunks: int = 0
    status: str = "indexed"  # indexed | error | empty | encrypted | scanned | duplicate
    error: Optional[str] = None


class Chunk(BaseModel):
    """A retrievable passage with full provenance metadata."""

    chunk_id: str
    doc_id: str
    source_filename: str
    doc_type: str
    company: Optional[str] = None
    section: str = ""
    page_start: int
    page_end: int
    text: str
    effective_date: Optional[str] = None
    version: Optional[str] = None
    synthetic: bool = False
    token_count: int = 0

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


class RetrievedChunk(BaseModel):
    """A chunk plus the retrieval signals that surfaced it."""

    chunk: Chunk
    rrf_score: float = 0.0
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    final_score: float = 0.0
    notes: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Output of hybrid retrieval, including the evidence verdict."""

    query: str
    results: list[RetrievedChunk] = Field(default_factory=list)
    evidence_sufficient: bool = False
    reason: str = ""
    filters: dict = Field(default_factory=dict)
    embedder: str = ""
    term_coverage: float = 0.0
