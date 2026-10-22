"""End-to-end ingestion: PDF -> clean -> chunk -> persisted corpus.

Outputs (in ``data/processed``):
* ``chunks.jsonl``  - one JSON chunk per line
* ``registry.json`` - one DocumentRecord per source file, including errors and duplicates
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from config.settings import Settings, get_settings
from src.ingestion.chunker import chunk_document
from src.ingestion.cleaner import clean_pages
from src.ingestion.metadata import detect_metadata
from src.ingestion.pdf_loader import PDFLoadError, file_sha256, load_pdf
from src.schemas import Chunk, DocumentRecord


def make_doc_id(filename: str, sha: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")
    return f"{stem}-{sha[:6]}"


def ingest_file(
    path: str | Path,
    settings: Settings | None = None,
    known_hashes: dict[str, str] | None = None,
) -> tuple[DocumentRecord, list[Chunk]]:
    """Ingest one PDF. Never raises for bad input; returns a record with a status instead.

    ``known_hashes`` maps sha256 -> doc_id of already-ingested files, used to flag duplicates.
    """
    s = settings or get_settings()
    p = Path(path)
    try:
        sha = file_sha256(p)
    except OSError as exc:
        rec = DocumentRecord(doc_id=make_doc_id(p.name, "000000"), filename=p.name, sha256="",
                             status="error", error=str(exc))
        return rec, []
    doc_id = make_doc_id(p.name, sha)
    if known_hashes and sha in known_hashes:
        return (
            DocumentRecord(doc_id=doc_id, filename=p.name, sha256=sha, status="duplicate",
                           error=f"Identical content already indexed as {known_hashes[sha]}"),
            [],
        )
    try:
        pages = load_pdf(p, max_mb=s.max_upload_mb)
    except PDFLoadError as exc:
        return (
            DocumentRecord(doc_id=doc_id, filename=p.name, sha256=sha, status=exc.status, error=exc.message),
            [],
        )
    cleaned = clean_pages(pages)
    meta = detect_metadata(cleaned, p.name)
    doc_meta = {"doc_id": doc_id, "filename": p.name, **meta}
    chunks = chunk_document(
        cleaned,
        doc_meta,
        target_tokens=s.chunk_target_tokens,
        max_tokens=s.chunk_max_tokens,
        overlap_tokens=s.chunk_overlap_tokens,
    )
    rec = DocumentRecord(
        doc_id=doc_id,
        filename=p.name,
        sha256=sha,
        doc_type=meta["doc_type"],
        company=meta["company"],
        effective_date=meta["effective_date"],
        version=meta["version"],
        synthetic=meta["synthetic"],
        num_pages=len(pages),
        num_chunks=len(chunks),
        status="indexed" if chunks else "empty",
        error=None if chunks else "No chunks produced",
    )
    return rec, chunks


def ingest_directory(
    directory: str | Path, settings: Settings | None = None
) -> tuple[list[DocumentRecord], list[Chunk]]:
    """Ingest every ``*.pdf`` in a directory (non-recursive, sorted for determinism)."""
    s = settings or get_settings()
    records: list[DocumentRecord] = []
    chunks: list[Chunk] = []
    seen: dict[str, str] = {}
    for pdf in sorted(Path(directory).glob("*.pdf")):
        rec, ch = ingest_file(pdf, s, known_hashes=seen)
        records.append(rec)
        chunks.extend(ch)
        if rec.status == "indexed":
            seen[rec.sha256] = rec.doc_id
    return records, chunks


def save_corpus(records: Iterable[DocumentRecord], chunks: Iterable[Chunk], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    with open(out / "registry.json", "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2)


def load_corpus(in_dir: str | Path) -> tuple[list[DocumentRecord], list[Chunk]]:
    d = Path(in_dir)
    reg_path, ch_path = d / "registry.json", d / "chunks.jsonl"
    if not reg_path.exists() or not ch_path.exists():
        raise FileNotFoundError(
            f"No indexed corpus in {d}. Run: python -m scripts.build_index --source data/sample"
        )
    records = [DocumentRecord(**r) for r in json.loads(reg_path.read_text(encoding="utf-8"))]
    chunks = [Chunk.model_validate_json(line) for line in ch_path.read_text(encoding="utf-8").splitlines() if line]
    return records, chunks
