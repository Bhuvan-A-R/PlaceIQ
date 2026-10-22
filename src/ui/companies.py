"""Company Explorer and Company Comparison pages."""
from __future__ import annotations

import streamlit as st

from src.companies.comparison import comparison_table, directory_table, to_csv_bytes
from src.companies.company_catalog import NOT_STATED, filter_companies
from src.eligibility.rules_engine import StudentProfile, evaluate_company
from src.generation.chatbot import PlaceBot
from src.ui.common import hero
from src.ui.eligibility import _branch_options


def _scope_note(bot: PlaceBot) -> None:
    st.caption(f"Covers only the {len(bot.catalog)} companies present in the indexed documents, not all employers. "
               f"'{NOT_STATED}' means the documents are silent; it never means zero.")


def render_explorer(bot: PlaceBot) -> None:
    hero("Company Explorer", "Searchable directory built from the indexed documents.")
    _scope_note(bot)
    st.session_state.setdefault("company_explorer_applied_filters", {
        "branch": None, "cgpa": None, "backlogs": None, "ctc_min": None, "ctc_max": None,
        "location": "", "employment_type": "", "role": "", "search": "", "unknown": True,
    })
    with st.expander("Filters", expanded=True):
        current = st.session_state["company_explorer_applied_filters"]
        with st.form("company_explorer_filter_form"):
            c1, c2, c3 = st.columns(3)
            branch_options = _branch_options(bot)
            saved_branch = current["branch"]
            branch_index = branch_options.index(saved_branch) if saved_branch in branch_options else None
            branch = c1.selectbox("Branch", branch_options, index=branch_index, placeholder="Any",
                                  key="company_filter_branch")
            cgpa = c2.number_input("My CGPA", 0.0, 10.0, value=current["cgpa"], step=0.01,
                                   key="company_filter_cgpa")
            backlogs = c3.number_input("My active backlogs", 0, 50, value=current["backlogs"], step=1,
                                       key="company_filter_backlogs")
            c4, c5, c6, c7 = st.columns(4)
            ctc_min = c4.number_input("Min CTC (LPA)", 0.0, 500.0, value=current["ctc_min"], step=0.5,
                                      key="company_filter_ctc_min")
            ctc_max = c5.number_input("Max CTC (LPA)", 0.0, 500.0, value=current["ctc_max"], step=0.5,
                                      key="company_filter_ctc_max")
            location = c6.text_input("Location contains", value=current["location"], key="company_filter_location")
            emp = c7.text_input("Role type contains (e.g. internship, full-time)",
                                value=current["employment_type"], key="company_filter_employment")
            role = st.text_input("Job role contains", value=current["role"], key="company_filter_role")
            search = st.text_input("Search company name", value=current["search"], key="company_filter_search")
            unknown = st.checkbox("Also show companies where a filtered field is not stated or conflicting",
                                  value=current["unknown"], key="company_filter_unknown")
            submitted = st.form_submit_button("Apply filters", type="primary", width="stretch")
        if submitted:
            st.session_state["company_explorer_applied_filters"] = {
                "branch": branch, "cgpa": cgpa, "backlogs": backlogs, "ctc_min": ctc_min, "ctc_max": ctc_max,
                "location": location, "employment_type": emp, "role": role, "search": search, "unknown": unknown,
            }
            st.rerun()
    filters = st.session_state["company_explorer_applied_filters"]
    profiles = filter_companies(bot.catalog.values(), branch=filters["branch"], cgpa=filters["cgpa"],
                                active_backlogs=None if filters["backlogs"] is None else int(filters["backlogs"]),
                                ctc_min=filters["ctc_min"], ctc_max=filters["ctc_max"],
                                location=filters["location"], employment_type=filters["employment_type"],
                                role=filters["role"], include_unknown=filters["unknown"])
    search = filters["search"]
    if search:
        profiles = [p for p in profiles if search.lower() in p.name.lower()]
    st.write(f"**{len(profiles)}** of {len(bot.catalog)} indexed companies match.")
    if not profiles:
        return
    df = directory_table(profiles)
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download directory (CSV)", to_csv_bytes(df.set_index("Company")), "companies.csv", "text/csv")

    name = st.selectbox("Company details", [p.name for p in profiles])
    p = bot.catalog[name]
    st.subheader(p.name)
    left, right = st.columns(2)
    with left:
        st.write(f"**Role:** {p.role or NOT_STATED}")
        st.write(f"**Employment type:** {p.employment_type or NOT_STATED}")
        st.write(f"**CTC:** {p.ctc_text or NOT_STATED}")
        st.write(f"**Fixed:** {p.fixed_text or NOT_STATED}  \n**Variable:** {p.variable_text or NOT_STATED}")
        st.write(f"**Location:** {p.location or NOT_STATED}")
        st.write(f"**Bond:** {p.bond_text or NOT_STATED}")
    with right:
        st.write(f"**Eligible branches:** {p.branches_text}")
        st.write(f"**Minimum CGPA:** {p.min_cgpa_text}")
        st.write(f"**Active backlogs:** {p.backlog_text}")
        st.write(f"**10th / 12th:** {p.tenth_text} / {p.twelfth_text}")
        st.write("**Selection rounds:** " + ("; ".join(f"{i}. {r}" for i, r in enumerate(p.rounds, 1))
                                             if p.rounds else NOT_STATED))
        st.write(f"**Last verified (latest document date):** {p.last_verified or NOT_STATED}")
    if any("CONFLICT" in t for t in (p.min_cgpa_text, p.branches_text, p.backlog_text)):
        st.warning("Conflicting documents were found for this company. Confirm the applicable rule with the placement cell.")
    st.caption("Sources: " + ", ".join(p.documents))
    if p.synthetic:
        st.caption("Synthetic sample data, not an official company or institutional document.")


def render_comparison(bot: PlaceBot) -> None:
    hero("Company Comparison", "Side-by-side view using the indexed documents.")
    _scope_note(bot)
    names = sorted(bot.catalog)
    chosen = st.multiselect("Choose two or more companies", names, default=names[:2])
    if len(chosen) < 2:
        st.info("Select at least two companies.")
        return
    df = comparison_table([bot.catalog[c] for c in chosen])
    st.dataframe(df, width="stretch")
    st.download_button("Download comparison (CSV)", to_csv_bytes(df), "comparison.csv", "text/csv")
    profile = st.session_state.get("profile")
    if isinstance(profile, StudentProfile):
        st.subheader("Your eligibility (from the Eligibility Checker form)")
        st.caption("Computed with the same deterministic rules; your inputs are kept only in this session.")
        for c in chosen:
            r = evaluate_company(profile, bot.catalog[c].criteria)
            st.write(f"**{c}:** {r.verdict.value} — {r.reason}")
