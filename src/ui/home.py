"""Home dashboard: every number is computed from the loaded corpus or this session."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.generation.chatbot import PlaceBot
from src.schemas import DocumentRecord
from src.ui.common import SUGGESTED, go_to, hero
from config.settings import get_settings
from pathlib import Path
from datetime import datetime


def render(bot: PlaceBot, records: list[DocumentRecord]) -> None:
    hero("PlaceIQ", "RAG-Powered Placement & Career Companion")
    indexed = [r for r in records if r.status == "indexed"]
    problems = [r for r in records if r.status not in ("indexed",)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed documents", len(indexed))
    c2.metric("Indexed companies", len(bot.catalog))
    c3.metric("Searchable passages", len(bot.chunks))
    c4.metric("Documents with issues", len(problems))
    if any(r.synthetic for r in indexed):
        n = sum(1 for r in indexed if r.synthetic)
        st.info(f"{n} of {len(indexed)} indexed documents are labelled synthetic sample data, not official placement documents.")

    a, b = st.columns(2)
    with a:
        st.subheader("Quick links")
        st.button("Check my eligibility", on_click=go_to, args=("Eligibility Checker",), type="primary",
                  width="stretch")
        st.button("Compare companies", on_click=go_to, args=("Company Comparison",), width="stretch")
        st.button("Placement policies", on_click=go_to, args=("Placement Policies",), width="stretch")
        st.subheader("Suggested questions")
        for i, q in enumerate(SUGGESTED):
            st.button(q, key=f"sugg_{i}", on_click=go_to, args=("Ask PlaceBot", q), width="stretch")
    with b:
        st.subheader("Recent queries (this session)")
        recent = st.session_state.get("recent", [])
        if recent:
            for q in reversed(recent[-6:]):
                st.write(f"• {q}")
        else:
            st.caption("No queries yet in this session.")
        st.subheader("Index status and freshness")
        chunks_file = Path(get_settings().processed_dir) / "chunks.jsonl"
        if chunks_file.exists():
            st.write(f"Index last built: **{datetime.fromtimestamp(chunks_file.stat().st_mtime):%Y-%m-%d %H:%M}**")
        st.write(f"Search: BM25 (`{bot.retriever.bm25.backend}`) + vectors (`{bot.retriever.vectors.backend}`, "
                 f"embedder `{bot.retriever.embedder.name}`)")
        st.write(f"Answer mode: **{bot.llm.name}**")
        if bot.llm_fallback_reason:
            st.warning(bot.llm_fallback_reason)
        rows = []
        for r in indexed:
            rows.append({"Type": r.doc_type, "Effective date": r.effective_date})
        if rows:
            df = pd.DataFrame(rows)
            g = df.groupby("Type")["Effective date"].agg(lambda s: f"{s.dropna().min()} → {s.dropna().max()}"
                                                          if s.notna().any() else "unknown").reset_index()
            g.columns = ["Document type", "Effective date range"]
            st.dataframe(g, hide_index=True, width="stretch")
        for r in problems:
            st.error(f"{r.filename}: {r.status} — {r.error}")
