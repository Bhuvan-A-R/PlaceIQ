"""Ask PlaceBot: conversational Q&A with citations."""
from __future__ import annotations

import streamlit as st

from src.generation.chatbot import PlaceBot
from src.ui.common import SUGGESTED, hero, render_answer


def render(bot: PlaceBot) -> None:
    hero("Ask PlaceBot", "Answers come only from the indexed placement documents, with sources.")
    chat = st.session_state.setdefault("chat", [])
    st.session_state.setdefault("recent", [])
    top = st.columns([2, 1])
    company = top[0].selectbox("Company filter (optional)", ["All companies"] + sorted(bot.catalog))
    if top[1].button("Clear conversation"):
        chat.clear()
        st.rerun()
    if bot.llm.is_mock:
        st.caption("Demo mode (no LLM configured): answers are extractive lines from retrieved passages.")
    if not chat:
        st.markdown("**Try:**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"ex_{i}", width="stretch"):
                st.session_state["pending_q"] = q
                st.rerun()

    for m in chat:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["text"])
            else:
                render_answer(m["answer"])

    pending = st.session_state.pop("pending_q", None)
    typed = st.chat_input("Ask about eligibility, packages, rounds or policies")
    question = pending or typed
    if question:
        history = [(m["role"], m["text"]) for m in chat]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                ans = bot.ask(question, history, company=None if company == "All companies" else company)
            render_answer(ans)
        chat.append({"role": "user", "text": question})
        chat.append({"role": "assistant", "text": ans.text, "answer": ans})
        st.session_state["recent"].append(question)
