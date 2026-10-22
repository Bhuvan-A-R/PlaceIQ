"""Eligibility Checker: deterministic verdicts from criteria extracted from indexed documents."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.eligibility.criteria_extractor import BRANCH
from src.eligibility.rules_engine import StudentProfile, evaluate_company
from src.eligibility.verdicts import EligibilityReport, Outcome
from src.generation.chatbot import PlaceBot
from src.ui.common import hero, outcome_style, verdict_color
from src.utils.audit import log_answer


def _branch_options(bot: PlaceBot) -> list[str]:
    seen: dict[str, str] = {}
    for p in bot.catalog.values():
        if p.criteria:
            for c in p.criteria.by_field.get(BRANCH, []):
                if isinstance(c.value, tuple):
                    for b in c.value:
                        seen.setdefault(b.upper(), b)
    return sorted(seen.values())


def report_table(rep: EligibilityReport) -> pd.DataFrame:
    rows = []
    for r in rep.results:
        rows.append({"Criterion": r.title, "Company requirement": r.requirement, "Student value": r.student_value,
                     "Result": r.outcome.value,
                     "Source": "; ".join(s.label() for s in r.sources) or "—"})
    return pd.DataFrame(rows)


def render_report(rep: EligibilityReport) -> None:
    st.markdown(f'<div class="pb-verdict" style="background:{verdict_color(rep.verdict.value)}">'
                f'{rep.company}: {rep.verdict.value}</div>', unsafe_allow_html=True)
    st.write(f"**Reason:** {rep.reason}")
    df = report_table(rep)
    st.dataframe(df.style.map(outcome_style, subset=["Result"]), width="stretch", hide_index=True)
    for w in rep.warnings:
        st.caption(f"Note: {w}")
    srcs = rep.all_sources()
    if srcs:
        with st.expander(f"Evidence passages ({len(srcs)})"):
            for s in srcs:
                st.markdown(f"**{s.filename}** · page {s.page} · {s.section} · effective {s.effective_date or 'unknown'}"
                            f" · version {s.version or 'unknown'}")
                st.text(s.passage)


def render(bot: PlaceBot) -> None:
    hero("Eligibility Checker", "Deterministic checks against criteria extracted from the indexed documents.")
    st.info("No name, USN or phone number is needed. Your inputs stay in this browser session and are not logged.")
    branches = _branch_options(bot)
    with st.form("elig_form"):
        c1, c2, c3 = st.columns(3)
        branch_sel = c1.selectbox("Branch", branches + ["Other (type below)"], index=None, placeholder="Select branch")
        other_branch = c1.text_input("Other branch (exact name used by the company)")
        cgpa = c2.number_input("Current CGPA", 0.0, 10.0, value=None, step=0.01, format="%.2f")
        active = c3.number_input("Active backlogs", 0, 50, value=None, step=1)
        c4, c5, c6, c7 = st.columns(4)
        hist = c4.number_input("Backlogs cleared in the past", 0, 50, value=None, step=1)
        tenth = c5.number_input("10th percentage", 0.0, 100.0, value=None, step=0.1)
        twelfth = c6.number_input("12th percentage", 0.0, 100.0, value=None, step=0.1)
        gap = c7.number_input("Academic gap (years)", 0.0, 10.0, value=None, step=0.5)
        status = st.selectbox("Graduation status (informational)", ["Not specified", "Pursuing", "Graduated"])
        names = sorted(bot.catalog)
        chosen = st.multiselect("Companies to check", names, default=names[:1])
        go = st.form_submit_button("Check eligibility", type="primary")
    st.caption("Leave a field empty if you don't know it: the result will be UNCLEAR rather than assumed.")

    if go:
        branch = other_branch.strip() if branch_sel and branch_sel.startswith("Other") else branch_sel
        st.session_state["profile"] = StudentProfile(
            branch=branch or None, cgpa=cgpa, active_backlogs=None if active is None else int(active),
            history_backlogs=None if hist is None else int(hist), tenth_pct=tenth, twelfth_pct=twelfth,
            gap_years=gap, graduation_status=None if status == "Not specified" else status)
        st.session_state["elig_companies"] = chosen
    profile = st.session_state.get("profile")
    chosen = st.session_state.get("elig_companies", [])
    if not profile or not chosen:
        return

    reports = [evaluate_company(profile, bot.catalog[c].criteria) for c in chosen if c in bot.catalog]
    for r in reports:
        log_answer("eligibility", r.company, r.verdict.value, "deterministic rules engine",
                   [bot.corpus[s.chunk_id] for s in r.all_sources() if s.chunk_id in bot.corpus])
    if len(reports) > 1:
        st.subheader("Summary")
        summ = pd.DataFrame([{"Company": r.company, "Verdict": r.verdict.value,
                              "Failed": ", ".join(x.title for x in r.failed) or "—",
                              "Unclear": ", ".join(x.title for x in r.unclear) or "—"} for r in reports])
        st.dataframe(summ, hide_index=True, width="stretch")
        st.download_button("Download summary (CSV)", summ.to_csv(index=False).encode("utf-8-sig"),
                           "eligibility_summary.csv", "text/csv")
    for r in reports:
        st.divider()
        render_report(r)
