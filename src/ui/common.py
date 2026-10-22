"""Shared Streamlit helpers: styling, corpus loading, answer/citation rendering."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import streamlit as st

from config.settings import get_settings
from src.generation.chatbot import Answer, PlaceBot
from src.ingestion.pipeline import load_corpus
from src.schemas import DocumentRecord

WHITE, BLUE, ORANGE, RED = "#ffffff", "#1769aa", "#e67e22", "#c0392b"
NAVY, GREEN, AMBER = "#163b63", BLUE, ORANGE
PAGES = ["Home Dashboard", "Ask PlaceBot", "Eligibility Checker", "Company Explorer", "Company Comparison",
         "Placement Policies", "Documents and Sources", "Evaluation Dashboard", "About and Disclaimer"]
ICONS = {page: "" for page in PAGES}
SUGGESTED = [
    "What is the minimum CGPA for Nimbus Soft?",
    "What are the selection rounds for Orbit Analytics?",
    "What are the branch, CGPA and backlog requirements for IBM CIO?",
    "What are the eligibility requirements for GammaStack?",
    "What are the eligibility requirements for Wipfli India LLP?",
    "What are the eligibility requirements for DemandNXT?",
    "Which branches are preferred for Shriram Finance?",
    "What are the roles and skills required for the Duroflex Control Tower internships?",
    "What is the CTC and bond period for Quanta Grid Systems?",
    "Which indexed companies allow EEE students with 6.5 CGPA?",
    "What are the selection rounds for IBM CIO?",
    "What are the selection rounds for GammaStack?",
    "What is the placement policy for accepting another offer?",
    "What are the rules about active and past backlogs?",
    "What documents should I carry to a placement drive?",
    "What is the recruitment process for placement drives?",
]

def _theme_css(dark: bool) -> str:
    background = "#111827" if dark else "#ffffff"
    surface = "#1f2937" if dark else "#ffffff"
    sidebar = "#0f172a" if dark else "#f7faff"
    text = "#f3f4f6" if dark else "#111111"
    muted = "#cbd5e1" if dark else "#334e68"
    border = "#374151" if dark else "#d9e5f2"
    input_background = "#111827" if dark else "#ffffff"
    chip_background = "#263b53" if dark else "#eaf3fb"
    active_background = "#263b53" if dark else "#eaf3fb"
    return f"""
<style>
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: {background}; }}
[data-testid="stAppViewContainer"] .main, [data-testid="stAppViewContainer"] .main * {{ color: {text}; }}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li {{ color: {muted}; }}
[data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4 {{ color: {text}; }}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {text} !important; }}
[data-testid="stAppViewContainer"] a {{ color: {BLUE}; }}
[data-testid="stSidebar"] {{ background: {sidebar}; border-right: 1px solid {border}; }}
[data-testid="stSidebar"] * {{ color: {text} !important; }}
[data-testid="stSidebar"] [aria-checked="true"] {{ background: {active_background}; border-left: 3px solid {ORANGE}; }}
.pb-hero {{ background: linear-gradient(120deg, {BLUE}, #2f86c1); color: {WHITE}; padding: 1.35rem 1.6rem;
  border-radius: 10px; margin-bottom: 1.2rem; box-shadow: 0 4px 14px #1769aa22; }}
.pb-hero h1 {{ margin: 0; font-size: 1.8rem; color: {WHITE}; }}
.pb-hero p {{ margin: .35rem 0 0; opacity: .92; }}
.pb-badge {{ display: inline-block; padding: .2rem .75rem; border-radius: 6px; font-weight: 700;
  font-size: .85rem; color: {WHITE}; }}
.pb-chip {{ display: inline-block; padding: .15rem .6rem; border-radius: 5px; font-size: .78rem;
  background: {chip_background}; color: {text}; margin-left: .4rem; }}
div.stButton > button {{ border-radius: 6px; border-color: {border}; }}
div.stButton > button:not([kind="primary"]) {{ background: {surface}; color: {text}; }}
div.stButton > button[kind="primary"] {{ background: {BLUE}; border-color: {BLUE}; color: {WHITE}; }}
div.stButton > button:hover {{ border-color: {ORANGE}; color: {ORANGE}; }}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input::placeholder, [data-testid="stNumberInput"] input::placeholder,
[data-testid="stSelectbox"] div, [data-testid="stSelectbox"] input,
[data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] * {{ color: {text}; }}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div {{ background: {input_background}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {border}; }}
.stAlert {{ background: {surface}; color: {text}; border-color: {border}; }}
[data-testid="stExpander"] {{ background: {surface}; border-color: {border}; }}
.pb-verdict {{ padding: .8rem 1rem; border-radius: 8px; color: {WHITE}; font-size: 1.15rem; font-weight: 700; }}
</style>
"""


def inject_css(dark: bool = False) -> None:
    st.markdown(_theme_css(dark), unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="pb-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="pb-badge" style="background:{color}">{html.escape(text)}</span>'


def verdict_color(verdict_value: str) -> str:
    return {"ELIGIBLE": GREEN, "NOT ELIGIBLE": RED}.get(verdict_value, AMBER)


def outcome_style(v: str) -> str:
    return {"Pass": f"background-color:#e3f4e8;color:{GREEN}", "Fail": f"background-color:#fde7e7;color:{RED}",
            "Unclear": f"background-color:#fff1d6;color:{AMBER}"}.get(v, "color:#5f6b7a")


# ------------------------------------------------------------------------ corpus
def corpus_stamp() -> str:
    s = get_settings()
    p = Path(s.processed_dir) / "chunks.jsonl"
    return f"{p.stat().st_mtime_ns}:{s.llm_provider}:{s.llm_model}" if p.exists() else "missing"


@st.cache_resource(show_spinner="Loading corpus and building search indexes...")
def _load_bot(stamp: str) -> tuple[PlaceBot, list[DocumentRecord]]:
    s = get_settings()
    records, chunks = load_corpus(s.processed_dir)
    return PlaceBot(chunks, s), records


def load_bot() -> Optional[tuple[PlaceBot, list[DocumentRecord]]]:
    """Return (bot, records) or None (after showing setup help) if no corpus has been built."""
    stamp = corpus_stamp()
    if stamp == "missing":
        st.warning("No indexed documents found yet.")
        st.write("Build the index from the synthetic sample PDFs (or your own PDFs in `data/raw`):")
        if st.button("Build index from sample data", type="primary"):
            from src.ingestion.manager import rebuild_index
            with st.spinner("Ingesting PDFs..."):
                rebuild_index(include_sample=True)
            _load_bot.clear()
            st.rerun()
        st.code("python -m scripts.build_index --source data/sample", language="bash")
        return None
    try:
        return _load_bot(stamp)
    except Exception as exc:  # corrupt corpus files etc.
        st.error(f"Could not load the indexed corpus: {exc}")
        return None


def refresh_corpus() -> None:
    _load_bot.clear()


# --------------------------------------------------------------------- rendering
def render_citations(answer: Answer) -> None:
    if not answer.citations:
        return
    st.markdown("**Sources**")
    for c in answer.citations:
        title = f"[{c.label}] {c.filename} · page {c.page} · {c.doc_type.replace('_', ' ')}"
        with st.expander(title):
            st.markdown(f"**Company:** {c.company or 'n/a'}  \n**Section:** {c.section or 'n/a'}  \n"
                        f"**Effective date:** {c.effective_date or 'unknown'} · **Version:** {c.version or 'unknown'}")
            st.text(c.passage)


def render_retrieval(answer: Answer, label: str = "Retrieved passages") -> None:
    res = answer.retrieval
    if not res or not res.results:
        return
    with st.expander(f"{label} ({len(res.results)})"):
        st.caption(f"Embedder: {res.embedder} · company filter: {res.filters.get('company') or 'none'} · "
                   f"evidence check: {res.reason}. Scores are retrieval signals, not probabilities.")
        for i, r in enumerate(res.results, 1):
            c = r.chunk
            st.markdown(f"**{i}. {c.source_filename}** · p.{c.page_label} · {c.section} · "
                        f"RRF {r.rrf_score:.4f} · BM25 {r.bm25_score if r.bm25_score is not None else '–'} · "
                        f"vector {f'{r.vector_score:.2f}' if r.vector_score is not None else '–'}")
            for n in r.notes:
                st.caption(f"⚠ {n}")
            st.text(c.text)


def render_answer(answer: Answer) -> None:
    if answer.found:
        st.markdown(badge("ANSWERED · grounded in retrieved passages", GREEN) +
                    f'<span class="pb-chip">{html.escape(answer.mode)}</span>', unsafe_allow_html=True)
        st.markdown(answer.text)
        if answer.table is not None:
            st.dataframe(answer.table, width="stretch", hide_index=True)
    else:
        st.markdown(badge("INFORMATION NOT FOUND", AMBER), unsafe_allow_html=True)
        st.markdown(answer.text)
    for w in answer.warnings:
        st.caption(f"Note: {w}")
    if answer.validation and answer.found and not answer.validation.ok and answer.table is None:
        st.warning("Citation check reported issues with this answer; treat it with caution.")
    render_citations(answer)
    render_retrieval(answer, "Retrieved passages (not sufficient to answer)" if not answer.found
                     else "Retrieved passages")


def go_to(page: str, question: Optional[str] = None) -> None:
    """Button callback: switch page, optionally pre-filling a question for the chat page."""
    st.session_state["nav"] = page
    if question:
        st.session_state["pending_q"] = question
