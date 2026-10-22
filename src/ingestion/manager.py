"""Document management: upload validation, duplicate detection, versioned replacement, re-indexing.

Rules: files are never overwritten silently. A name clash gets a numbered suffix; replacing a
document marks the old file as *superseded* (kept on disk, excluded from the index) and records
who replaced what in ``data/processed/document_history.jsonl``.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from src.ingestion.pdf_loader import file_sha256
from src.ingestion.pipeline import ingest_file, save_corpus
from src.schemas import Chunk, DocumentRecord

PDF_MAGIC = b"%PDF-"


class UploadError(ValueError):
    """Raised for files that fail validation (type, size, emptiness)."""


def _history_path(s: Settings) -> Path:
    return s.processed_dir / "document_history.jsonl"


def _superseded_path(s: Settings) -> Path:
    return s.processed_dir / "superseded.json"


def load_superseded(s: Optional[Settings] = None) -> dict[str, dict]:
    s = s or get_settings()
    p = _superseded_path(s)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def record_event(event: str, filename: str, s: Optional[Settings] = None, **extra: object) -> None:
    s = s or get_settings()
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, "file": filename, **extra}
    with open(_history_path(s), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_history(s: Optional[Settings] = None) -> list[dict]:
    s = s or get_settings()
    p = _history_path(s)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def validate_upload(filename: str, data: bytes, s: Optional[Settings] = None) -> None:
    """Check extension, magic bytes, size and emptiness. Raises :class:`UploadError`."""
    s = s or get_settings()
    if not filename.lower().endswith(".pdf"):
        raise UploadError("Only .pdf files are accepted.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise UploadError(f"File exceeds the {s.max_upload_mb} MB limit.")
    if not data.lstrip()[:5].startswith(PDF_MAGIC):
        raise UploadError("The file does not look like a valid PDF (missing %PDF header).")


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    return stem or "document.pdf"


def _unique_path(directory: Path, name: str) -> Path:
    p = directory / name
    n = 2
    while p.exists():
        p = directory / f"{Path(name).stem}__v{n}{Path(name).suffix}"
        n += 1
    return p


def known_hashes(s: Optional[Settings] = None) -> dict[str, str]:
    """sha256 -> filename of every PDF already stored in raw/ or sample/."""
    s = s or get_settings()
    out: dict[str, str] = {}
    for d in (s.raw_dir, s.sample_dir):
        if d.is_dir():
            for p in sorted(d.glob("*.pdf")):
                try:
                    out.setdefault(file_sha256(p), p.name)
                except OSError:
                    continue
    return out


def save_upload(filename: str, data: bytes, replaces: Optional[str] = None,
                s: Optional[Settings] = None) -> tuple[Path, Optional[str]]:
    """Validate and store an upload in ``data/raw``. Returns (path, duplicate_of).

    If the content duplicates an existing file nothing is written and ``duplicate_of`` names it.
    If ``replaces`` names an existing file, that file is marked superseded (not deleted).
    """
    s = s or get_settings()
    validate_upload(filename, data, s)
    sha = hashlib.sha256(data).hexdigest()
    dup = known_hashes(s).get(sha)
    if dup:
        record_event("duplicate_rejected", filename, s, duplicate_of=dup)
        return Path(dup), dup
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(s.raw_dir, _safe_name(filename))
    dest.write_bytes(data)
    record_event("uploaded", dest.name, s, sha256=sha, replaces=replaces)
    if replaces:
        mark_superseded(replaces, dest.name, s)
    return dest, None


def mark_superseded(old_filename: str, new_filename: str, s: Optional[Settings] = None) -> None:
    s = s or get_settings()
    sup = load_superseded(s)
    sup[old_filename] = {"replaced_by": new_filename, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    _superseded_path(s).write_text(json.dumps(sup, indent=2), encoding="utf-8")
    record_event("superseded", old_filename, s, replaced_by=new_filename)


def restore_superseded(filename: str, s: Optional[Settings] = None) -> None:
    s = s or get_settings()
    sup = load_superseded(s)
    if sup.pop(filename, None) is not None:
        _superseded_path(s).write_text(json.dumps(sup, indent=2), encoding="utf-8")
        record_event("restored", filename, s)


def rebuild_index(include_sample: bool = True, s: Optional[Settings] = None) -> tuple[list[DocumentRecord], list[Chunk]]:
    """Re-ingest sample (optional) and raw PDFs, skipping superseded files, and save the corpus."""
    s = s or get_settings()
    s.ensure_dirs()
    sup = load_superseded(s)
    dirs = ([s.sample_dir] if include_sample else []) + [s.raw_dir]
    records: list[DocumentRecord] = []
    chunks: list[Chunk] = []
    seen: dict[str, str] = {}
    for d in dirs:
        for pdf in sorted(d.glob("*.pdf")):
            if pdf.name in sup:
                continue
            rec, ch = ingest_file(pdf, s, known_hashes=seen)
            records.append(rec)
            chunks.extend(ch)
            if rec.status == "indexed":
                seen[rec.sha256] = rec.doc_id
    save_corpus(records, chunks, s.processed_dir)
    record_event("reindexed", "*", s, documents=len(records), chunks=len(chunks), superseded=len(sup))
    return records, chunks
