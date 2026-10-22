"""PDF text extraction.

Primary backend: PyMuPDF (``fitz``). If PyMuPDF is not installed, ``pypdf`` is used as a
fallback so the pipeline still runs. Failure modes (corrupted, encrypted, empty, scanned)
are reported via :class:`PDFLoadError` with a ``status`` code instead of crashing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.schemas import PageText

MIN_CHARS_PER_DOC = 30  # below this the PDF is treated as empty/scanned


class PDFLoadError(Exception):
    """Raised when a PDF cannot be turned into usable text."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status  # error | empty | encrypted | scanned
        self.message = message


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_pdf_file(path: str | Path, max_mb: int = 20) -> None:
    """Check extension, size and magic bytes before parsing."""
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        raise PDFLoadError("error", f"{p.name}: not a .pdf file")
    if not p.exists():
        raise PDFLoadError("error", f"{p.name}: file not found")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise PDFLoadError("error", f"{p.name}: {size_mb:.1f} MB exceeds {max_mb} MB limit")
    if p.stat().st_size == 0:
        raise PDFLoadError("empty", f"{p.name}: file is empty")
    with open(p, "rb") as f:
        if not f.read(1024).lstrip().startswith(b"%PDF"):
            raise PDFLoadError("error", f"{p.name}: missing PDF header (corrupted or not a PDF)")


def _extract_pymupdf(path: Path) -> list[PageText]:
    import fitz  # type: ignore

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # corrupted
        raise PDFLoadError("error", f"{path.name}: cannot open ({exc})") from exc
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise PDFLoadError("encrypted", f"{path.name}: PDF is encrypted/password protected")
        pages = []
        for i, page in enumerate(doc, start=1):
            pages.append(PageText(page=i, text=page.get_text("text") or ""))
        return pages
    finally:
        doc.close()


def _extract_pypdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader  # type: ignore

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFLoadError("error", f"{path.name}: cannot open ({exc})") from exc
    if reader.is_encrypted:
        raise PDFLoadError("encrypted", f"{path.name}: PDF is encrypted/password protected")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append(PageText(page=i, text=page.extract_text() or ""))
        except Exception as exc:
            raise PDFLoadError("error", f"{path.name}: page {i} failed ({exc})") from exc
    return pages


def backend_name() -> str:
    try:
        import fitz  # noqa: F401

        return "pymupdf"
    except ImportError:
        return "pypdf"


def load_pdf(path: str | Path, max_mb: int = 20) -> list[PageText]:
    """Extract per-page text from a PDF.

    Raises:
        PDFLoadError: with status ``error``, ``empty``, ``encrypted`` or ``scanned``.
    """
    p = Path(path)
    validate_pdf_file(p, max_mb=max_mb)
    if backend_name() == "pymupdf":
        pages = _extract_pymupdf(p)
    else:
        pages = _extract_pypdf(p)
    if not pages:
        raise PDFLoadError("empty", f"{p.name}: PDF has no pages")
    total_chars = sum(len(pg.text.strip()) for pg in pages)
    if total_chars == 0:
        raise PDFLoadError(
            "scanned",
            f"{p.name}: no extractable text. This looks like a scanned PDF; OCR is required "
            "(e.g. run `ocrmypdf` on it first) and then re-upload.",
        )
    if total_chars < MIN_CHARS_PER_DOC:
        raise PDFLoadError("empty", f"{p.name}: almost no text extracted ({total_chars} chars)")
    return pages
