"""Document management: validation, duplicates, versioned replacement, superseded exclusion."""
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from conftest import SAMPLE_DIR

from config.settings import get_settings
from src.ingestion.manager import (
    UploadError, load_superseded, read_history, rebuild_index, restore_superseded, save_upload, validate_upload,
)


def _settings():
    tmp = Path(tempfile.mkdtemp())
    return dataclasses.replace(get_settings(), raw_dir=tmp / "raw", processed_dir=tmp / "proc", log_answers=False)


def _faq_bytes() -> bytes:
    return (SAMPLE_DIR / "placement_faq_SYNTHETIC.pdf").read_bytes()


def _raises(fn, *a):
    try:
        fn(*a)
    except UploadError:
        return True
    return False


def test_validation_rejects_bad_files():
    s = _settings()
    assert _raises(validate_upload, "a.txt", b"x", s)
    assert _raises(validate_upload, "a.pdf", b"", s)
    assert _raises(validate_upload, "a.pdf", b"not pdf", s)
    big = dataclasses.replace(s, max_upload_mb=0)
    assert _raises(validate_upload, "a.pdf", b"%PDF-1.4 x", big)
    validate_upload("a.pdf", _faq_bytes(), s)


def test_duplicate_content_is_not_stored():
    s = _settings()
    path, dup = save_upload("copy.pdf", _faq_bytes(), s=s)
    assert dup == "placement_faq_SYNTHETIC.pdf"
    assert not (s.raw_dir / "copy.pdf").exists()
    assert read_history(s)[-1]["event"] == "duplicate_rejected"


def test_name_clash_never_overwrites():
    s = _settings()
    a, _ = save_upload("circ.pdf", _faq_bytes() + b"\n%a", s=s)
    b, _ = save_upload("circ.pdf", _faq_bytes() + b"\n%b", s=s)
    assert a != b and a.exists() and b.exists()


def test_replacement_supersedes_old_and_rebuild_excludes_it():
    s = _settings()
    new, _ = save_upload("faq_v2.pdf", _faq_bytes() + b"\n%edit", replaces="placement_faq_SYNTHETIC.pdf", s=s)
    assert "placement_faq_SYNTHETIC.pdf" in load_superseded(s)
    recs, _ = rebuild_index(include_sample=True, s=s)
    names = {r.filename for r in recs}
    assert "placement_faq_SYNTHETIC.pdf" not in names and new.name in names
    assert (SAMPLE_DIR / "placement_faq_SYNTHETIC.pdf").exists()  # superseded file is kept on disk
    restore_superseded("placement_faq_SYNTHETIC.pdf", s)
    assert load_superseded(s) == {}
