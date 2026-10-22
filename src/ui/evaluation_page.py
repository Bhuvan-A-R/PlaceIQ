"""Evaluation Dashboard: results computed by src.evaluation.evaluate."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import get_settings
from src.evaluation.evaluate import run_evaluation, summarize, sweep_threshold
from src.generation.chatbot import PlaceBot
from src.retrieval.hybrid_search import HybridRetriever
from src.ui.common import hero


def render(bot: PlaceBot) -> None:
    hero("Evaluation Dashboard", "Retrieval, answer and citation quality on the evaluation questions.")
    s = get_settings()
    out = Path(s.processed_dir) / "eval_results.csv"
    st.caption("The question set is small and written for the synthetic corpus. Scores show the pipeline "
               "works on these cases; they are not a guarantee on real documents.")
    if st.button("Run evaluation now", type="primary"):
        quiet = dataclasses.replace(s, log_answers=False)
        eval_bot = PlaceBot(bot.chunks, quiet, retriever=bot.retriever, llm=bot.llm)
        with st.spinner("Evaluating..."):
            df = run_evaluation(eval_bot)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
    if not out.exists():
        st.info("No results yet. Click 'Run evaluation now' or run `python -m src.evaluation.evaluate`.")
        return
    df = pd.read_csv(out, dtype=str).fillna("")
    sm = summarize(df)
    cols = st.columns(5)
    cols[0].metric("Retrieval hit rate", f"{sm['retrieval_hit_rate']:.0%}")
    cols[1].metric("Answer correct", f"{sm['answer_correct_rate']:.0%}")
    cols[2].metric("Citation correct", f"{sm['citation_correct_rate']:.0%}")
    cols[3].metric("Unanswerable handled", f"{sm['unanswerable_handled_rate']:.0%}")
    cols[4].metric("Unsupported claims", sm["unsupported_claim_count"])
    st.caption(f"Mode used: {df['mode'].iloc[0] if len(df) else 'n/a'}. In demo mode this evaluates extractive "
               "output, not an LLM.")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "eval_results.csv", "text/csv")
    with st.expander("Threshold sweep (min term coverage)"):
        if st.button("Run sweep"):
            quiet = dataclasses.replace(s, log_answers=False)
            st.dataframe(sweep_threshold(bot.chunks, quiet, bot.retriever.embedder), hide_index=True)
