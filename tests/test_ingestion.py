"""Tests: PDF extraction, cleaning, chunk metadata, chunking behaviour, bad-file handling."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from conftest import SAMPLE_DIR, sample_corpus  # noqa: F401

from src.ingestion.chunker import chunk_document, count_tokens, is_heading, is_table_line
from src.ingestion.cleaner import clean_pages
from src.ingestion.metadata import detect_metadata
from src.ingestion.pdf_loader import PDFLoadError, load_pdf
from src.ingestion.pipeline import ingest_directory, ingest_file
from src.schemas import PageText


# ----------------------------------------------------------------- extraction
def test_pdf_extraction_returns_text_and_page_numbers():
    pages = load_pdf(SAMPLE_DIR / "nimbus_soft_eligibility_SYNTHETIC.pdf")
    assert pages[0].page == 1
    assert "Minimum CGPA" in pages[0].text and "7.0" in pages[0].text


def test_multipage_policy_keeps_page_numbers():
    pages = load_pdf(SAMPLE_DIR / "placement_policy_SYNTHETIC.pdf")
    assert len(pages) >= 2
    assert "One-Offer Rule" in pages[1].text


def _raises(status: str, fn) -> bool:
    try:
        fn()
    except PDFLoadError as exc:
        return exc.status == status
    return False


def test_non_pdf_corrupted_and_empty_files_are_rejected_gracefully():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "notes.txt").write_text("hello")
        (d / "broken.pdf").write_bytes(b"this is not a pdf at all")
        (d / "empty.pdf").write_bytes(b"")
        assert _raises("error", lambda: load_pdf(d / "notes.txt"))
        assert _raises("error", lambda: load_pdf(d / "broken.pdf"))
        assert _raises("empty", lambda: load_pdf(d / "empty.pdf"))
        rec, chunks = ingest_file(d / "broken.pdf")
        assert rec.status == "error" and chunks == [] and rec.error


def test_scanned_or_blank_pdf_reports_ocr_needed():
    from reportlab.pdfgen import canvas

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "blank.pdf"
        c = canvas.Canvas(str(p))
        c.showPage()
        c.save()
        assert _raises("scanned", lambda: load_pdf(p))
        rec, _ = ingest_file(p)
        assert rec.status == "scanned" and "OCR" in (rec.error or "")


def test_encrypted_pdf_is_flagged():
    from pypdf import PdfReader, PdfWriter

    with tempfile.TemporaryDirectory() as d:
        src = SAMPLE_DIR / "placement_faq_SYNTHETIC.pdf"
        w = PdfWriter()
        for pg in PdfReader(str(src)).pages:
            w.add_page(pg)
        w.encrypt("secret")
        out = Path(d) / "locked.pdf"
        with open(out, "wb") as f:
            w.write(f)
        assert _raises("encrypted", lambda: load_pdf(out))


def test_duplicate_files_are_detected_not_double_indexed():
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(SAMPLE_DIR / "placement_faq_SYNTHETIC.pdf", Path(d) / "a.pdf")
        shutil.copy(SAMPLE_DIR / "placement_faq_SYNTHETIC.pdf", Path(d) / "b_copy.pdf")
        records, chunks = ingest_directory(d)
        statuses = sorted(r.status for r in records)
        assert statuses == ["duplicate", "indexed"]
        assert len({c.doc_id for c in chunks}) == 1


# -------------------------------------------------------------------- cleaning
def test_cleaner_removes_page_numbers_and_rejoins_hyphenation():
    pages = [PageText(page=1, text="Eligi-\nbility rules\n\n\n\nMinimum CGPA 7.0\n3")]
    out = clean_pages(pages)[0].text
    assert "Eligibility rules" in out and "7.0" in out
    assert not out.rstrip().endswith("\n3") and "\n\n\n" not in out


def test_cleaner_strips_repeated_header_footer_lines():
    pages = [PageText(page=i, text=f"ACME CONFIDENTIAL\nbody line {i}\nfooter") for i in range(1, 5)]
    out = clean_pages(pages)
    assert all("ACME CONFIDENTIAL" not in p.text for p in out)
    assert all(f"body line {p.page}" in p.text for p in out)


# -------------------------------------------------------------------- metadata
def test_metadata_detected_from_header_lines():
    pages = load_pdf(SAMPLE_DIR / "quanta_grid_eligibility_revised_SYNTHETIC.pdf")
    meta = detect_metadata(clean_pages(pages), "whatever.pdf")
    assert meta["doc_type"] == "eligibility_circular"
    assert meta["company"].startswith("Quanta Grid Systems")
    assert meta["effective_date"] == "2025-10-15" and meta["version"] == "2.0"
    assert meta["synthetic"] is True


def test_metadata_maps_additional_sample_companies():
    assert detect_metadata([PageText(page=1, text="Role: analyst")],
                           "JD-Control Tower Analytics Intern.pdf")["company"] == "Duroflex"
    assert detect_metadata([PageText(page=1, text="Full-Stack Developer")],
                           "JD_DemandNxt_Full-Stack AI Developer.pdf")["company"] \
        == "DemandNXT Business Services Pvt. Ltd."


def test_policy_has_no_company():
    pages = load_pdf(SAMPLE_DIR / "placement_policy_SYNTHETIC.pdf")
    meta = detect_metadata(clean_pages(pages), "placement_policy_SYNTHETIC.pdf")
    assert meta["doc_type"] == "placement_policy" and meta["company"] is None


def test_all_five_required_document_types_are_present_in_samples():
    records, _ = sample_corpus()
    types = {r.doc_type for r in records if r.status == "indexed"}
    assert {"placement_policy", "company_jd", "eligibility_circular", "placement_faq", "recruitment_process"} <= types


# ------------------------------------------------------------------- chunking
def test_heading_and_table_line_detection():
    assert is_heading("1. Eligibility Criteria")
    assert is_heading("3.1 Something short")  # numbered clause lines count as headings only if short
    assert not is_heading("CTC | 12 LPA")  # two-column table row is NOT a heading
    assert not is_heading("Document Type: Company Job Description")
    assert not is_heading("This is a normal sentence that ends with a period.")
    assert is_table_line("CTC | 12 LPA") and not is_table_line("no pipes here")


def test_every_chunk_has_complete_metadata_and_valid_pages():
    records, chunks = sample_corpus()
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    doc_pages = {r.doc_id: r.num_pages for r in records}
    for c in chunks:
        assert c.chunk_id and c.doc_id and c.source_filename and c.doc_type != ""
        assert c.text.strip()
        assert 1 <= c.page_start <= c.page_end <= doc_pages[c.doc_id]
        assert c.synthetic is True  # all fixture samples are synthetic
        assert c.effective_date and c.version


def test_company_documents_carry_company_metadata():
    _, chunks = sample_corpus()
    for c in chunks:
        if c.doc_type in ("company_jd", "eligibility_circular"):
            assert c.company, c.chunk_id


def test_chunks_respect_max_tokens():
    _, chunks = sample_corpus()
    assert all(c.token_count <= 400 for c in chunks)


def test_compensation_table_chunk_keeps_header_with_values():
    _, chunks = sample_corpus()
    comp = [c for c in chunks if "nimbus_soft_JD" in c.source_filename and "Compensation" in c.section]
    assert len(comp) == 1
    assert comp[0].text.splitlines()[0] == "Item | Details"
    assert "CTC | 12 LPA" in comp[0].text and "Bond period | 12 months" in comp[0].text


def test_long_table_is_split_with_header_repeated_in_every_piece():
    rows = [f"Row {i} | value {i} | note about row {i}" for i in range(60)]
    text = "1. Big Table\nName | Value | Note\n" + "\n".join(rows)
    pages = [PageText(page=1, text=text)]
    meta = {"doc_id": "d1", "filename": "big.pdf", "doc_type": "company_jd"}
    chunks = chunk_document(pages, meta, target_tokens=100, max_tokens=150, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.splitlines()[0] == "Name | Value | Note"
    got = [ln for c in chunks for ln in c.text.splitlines() if ln.startswith("Row ")]
    assert got == rows  # nothing lost, nothing duplicated, order kept


def test_page_range_tracks_section_spanning_pages():
    pages = [
        PageText(page=1, text="1. Policy\nfirst line on page one"),
        PageText(page=2, text="second line on page two"),
    ]
    chunks = chunk_document(pages, {"doc_id": "d", "filename": "f.pdf"})
    assert len(chunks) == 1 and (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert chunks[0].page_label == "1-2"


def test_text_overlap_carries_trailing_lines_when_section_is_split():
    lines = [f"Sentence number {i} has several words to take up some token budget here." for i in range(40)]
    pages = [PageText(page=1, text="1. Long Section\n" + "\n".join(lines))]
    chunks = chunk_document(pages, {"doc_id": "d", "filename": "f.pdf"}, target_tokens=80, max_tokens=120, overlap_tokens=30)
    assert len(chunks) > 2
    first_tail = chunks[0].text.splitlines()[-1]
    assert first_tail in chunks[1].text.splitlines()


def test_count_tokens_monotonic():
    assert count_tokens("") == 0
    assert count_tokens("a b c d") < count_tokens("a b c d e f g h")
