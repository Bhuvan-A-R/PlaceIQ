"""Documents and Sources: inventory, upload, replacement, re-indexing, provenance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import get_settings
from src.generation.chatbot import PlaceBot
from src.ingestion.manager import (
    UploadError, load_superseded, read_history, rebuild_index, restore_superseded, save_upload,
)
from src.schemas import DocumentRecord
from src.ui.common import hero, refresh_corpus
from src.utils.audit import read_log


def render(bot: PlaceBot, records: list[DocumentRecord]) -> None:
    hero("Documents and Sources", "What is indexed, where it came from, and how to update it.")
    s = get_settings()
    t1, t2, t3, t4 = st.tabs(["Indexed documents", "Upload / replace", "Re-index", "History and provenance"])

    with t1:
        df = pd.DataFrame([{"File": r.filename, "Type": r.doc_type, "Company": r.company or "—",
                            "Effective date": r.effective_date or "unknown", "Version": r.version or "unknown",
                            "Pages": r.num_pages, "Chunks": r.num_chunks, "Status": r.status,
                            "Synthetic": "yes" if r.synthetic else "no", "Note": r.error or ""} for r in records])
        st.dataframe(df, hide_index=True, width="stretch")
        ok = df[df["Status"] == "indexed"]
        a, b = st.columns(2)
        a.write("**Documents by type**")
        a.dataframe(ok.groupby("Type").size().rename("Documents").reset_index(), hide_index=True)
        b.write("**Documents by company**")
        b.dataframe(ok.groupby("Company").size().rename("Documents").reset_index(), hide_index=True)
        for r in records:
            if r.status == "scanned":
                st.warning(f"{r.filename}: no extractable text. This looks scanned; run OCR (for example with "
                           "ocrmypdf) and re-upload.")
            elif r.status in ("error", "encrypted", "empty"):
                st.error(f"{r.filename}: {r.status} — {r.error}")
            elif r.status == "duplicate":
                st.info(f"{r.filename}: duplicate — {r.error}")
        sup = load_superseded(s)
        if sup:
            st.write("**Superseded documents (kept on disk, excluded from the index)**")
            for old, info in sup.items():
                c1, c2 = st.columns([4, 1])
                c1.write(f"{old} → replaced by {info['replaced_by']} ({info['ts']})")
                if c2.button("Restore", key=f"restore_{old}"):
                    restore_superseded(old)
                    st.info("Restored. Re-index to include it again.")

    with t2:
        st.write(f"PDF only, up to {s.max_upload_mb} MB. Files are never overwritten silently; duplicates are detected.")
        up = st.file_uploader("Upload a PDF", type=["pdf"])
        active = [r.filename for r in records if r.status == "indexed"]
        replaces = st.selectbox("Does this replace an existing document?", ["No, add as new"] + active)
        if up is not None and st.button("Save upload", type="primary"):
            try:
                path, dup = save_upload(up.name, up.getvalue(), None if replaces.startswith("No") else replaces)
                if dup:
                    st.info(f"Identical content already exists as '{dup}'. Nothing was saved.")
                else:
                    st.success(f"Saved as {path.name}. Go to the Re-index tab to index it.")
            except UploadError as exc:
                st.error(str(exc))

    with t3:
        include = st.checkbox("Include synthetic sample documents", value=True)
        st.caption("Re-indexing rebuilds the corpus from data/sample (optional) and data/raw, skipping superseded files.")
        if st.button("Re-index now", type="primary"):
            with st.spinner("Ingesting PDFs..."):
                recs, chunks = rebuild_index(include_sample=include)
            refresh_corpus()
            st.success(f"Indexed {sum(r.status == 'indexed' for r in recs)} documents, {len(chunks)} passages.")
            bad = [r for r in recs if r.status not in ("indexed",)]
            for r in bad:
                st.error(f"{r.filename}: {r.status} — {r.error}")
            st.caption("Switch to another page and back (or refresh) to see the new index everywhere.")

    with t4:
        st.write("**Document history**")
        hist = read_history(s)
        st.dataframe(pd.DataFrame(hist[::-1]) if hist else pd.DataFrame({"info": ["No events yet."]}),
                     hide_index=True, width="stretch")
        st.write("**Which source versions supplied recent answers**")
        st.caption("Stores a short hash of the question, never the student profile or the question text.")
        log = read_log(s, 100)
        rows = [{"time": r["ts"], "kind": r["kind"], "status": r["status"], "mode": r["mode"],
                 "sources": "; ".join(f"{x['filename']} (v{x['version'] or '?'}, {x['effective_date'] or '?'})"
                                      for x in r["sources"])} for r in log[::-1]]
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["No answers logged yet."]}),
                     hide_index=True, width="stretch")
