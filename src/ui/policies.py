"""Placement Policy Assistant: policy/FAQ-only retrieval with clause citations."""
from __future__ import annotations

import streamlit as st

from src.generation.chatbot import PlaceBot
from src.ui.common import hero, render_answer

TOPICS = {
    "One-offer rule": "What is the one-offer rule?",
    "Regular, Dream and Super Dream categories": "What are the Regular, Dream and Super Dream categories?",
    "Offer acceptance": "When is an offer considered accepted and can it be exchanged?",
    "Debarment": "When can a student be debarred?",
    "Mandatory drive attendance": "Is attendance mandatory after registering for a drive?",
    "Eligibility after receiving an offer": "Can a student attend another drive after receiving an offer?",
    "Withdrawing from a process": "How can a student withdraw from a recruitment process?",
    "Company-specific exceptions": "Can a company circular override the placement policy?",
}


def render(bot: PlaceBot) -> None:
    hero("Placement Policies", "Policy answers cite the clause and document. Nothing is assumed beyond the indexed policy.")
    st.caption("Only the policies in the indexed documents are used. If they are silent, PlaceBot says so "
               "instead of inventing a rule.")
    cols = st.columns(2)
    for i, (label, q) in enumerate(TOPICS.items()):
        if cols[i % 2].button(label, key=f"pol_{i}", width="stretch"):
            st.session_state["policy_q"] = q
    custom = st.text_input("Or ask your own policy question")
    question = custom.strip() or st.session_state.get("policy_q")
    if not question:
        return
    st.markdown(f"**Question:** {question}")
    with st.spinner("Searching policy documents..."):
        ans = bot.ask_policy(question)
    render_answer(ans)
    st.info("If the policy is ambiguous or your situation is not covered, confirm with the official placement cell.")
