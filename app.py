"""PlaceIQ: RAG-Powered Placement & Career Companion.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PlaceIQ: RAG-Powered Placement & Career Companion", layout="wide")

from src.ui import about, chat, companies, documents, eligibility, evaluation_page, home, policies  # noqa: E402
from src.ui.common import ICONS, PAGES, inject_css, load_bot  # noqa: E402


def main() -> None:
    st.session_state.setdefault("nav", PAGES[0])
    st.session_state.setdefault("theme", "Light")
    theme = st.sidebar.radio("Appearance", ["Light", "Dark"], key="theme", horizontal=True)
    inject_css(dark=theme == "Dark")
    st.sidebar.markdown("## PlaceIQ")
    st.sidebar.caption("RAG-Powered Placement & Career Companion")
    page = st.sidebar.radio("Navigate", PAGES, key="nav", format_func=lambda p: p)
    st.sidebar.divider()
    st.sidebar.caption("Decision aid only. Official placement rules and the placement cell take precedence.")

    if page == "About and Disclaimer":
        about.render()
        return
    loaded = load_bot()
    if loaded is None:
        return
    bot, records = loaded
    if page == "Home Dashboard":
        home.render(bot, records)
    elif page == "Ask PlaceBot":
        chat.render(bot)
    elif page == "Eligibility Checker":
        eligibility.render(bot)
    elif page == "Company Explorer":
        companies.render_explorer(bot)
    elif page == "Company Comparison":
        companies.render_comparison(bot)
    elif page == "Placement Policies":
        policies.render(bot)
    elif page == "Documents and Sources":
        documents.render(bot, records)
    elif page == "Evaluation Dashboard":
        evaluation_page.render(bot)


main()
