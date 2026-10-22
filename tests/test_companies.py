"""Company catalog, filters and comparison (missing != zero)."""
from __future__ import annotations

from conftest import sample_corpus

from src.companies.comparison import comparison_table, directory_table
from src.companies.company_catalog import (
    NOT_STATED, build_catalog, filter_companies, parse_bond_months, parse_lpa,
)

ORBIT, NIMBUS, QUANTA = "Orbit Analytics (synthetic)", "Nimbus Soft (synthetic)", "Quanta Grid Systems (synthetic)"


def _cat():
    return build_catalog(sample_corpus()[1])


def test_catalog_has_each_company_once_and_no_policy_pseudo_company():
    assert set(_cat()) == {ORBIT, NIMBUS, QUANTA}


def test_profile_fields_come_from_documents():
    p = _cat()[NIMBUS]
    assert (p.role, p.ctc_lpa, p.location, p.bond_months) == ("Software Development Engineer", 12.0, "Bengaluru", 12.0)
    assert p.rounds == ["Online coding assessment", "Technical interview 1", "Technical interview 2", "HR interview"]
    assert p.sources["ctc"].filename == "nimbus_soft_JD_SYNTHETIC.pdf"


def test_missing_is_none_not_zero():
    p = _cat()[ORBIT]
    assert p.fixed_text is None and p.variable_text is None
    assert p.bond_months == 0.0 and p.bond_text == "No bond"  # explicit 'No bond' is a stated zero
    assert _cat()[NIMBUS].fixed_text == "10 LPA"


def test_parsers():
    assert parse_lpa("7.5 LPA") == 7.5 and parse_lpa("negotiable") is None
    assert parse_bond_months("2 years") == 24 and parse_bond_months("18 months") == 18
    assert parse_bond_months("No bond") == 0 and parse_bond_months("to be decided") is None


def test_comparison_table_marks_missing_as_not_stated():
    df = comparison_table(list(_cat().values()))
    assert df.loc["Fixed component", ORBIT] == NOT_STATED
    assert df.loc["Fixed component", NIMBUS] == "10 LPA"
    assert "CONFLICT" in df.loc["Minimum CGPA", QUANTA]
    assert df.loc["Selection rounds", ORBIT].startswith("1. Aptitude test")
    assert df.shape[1] == 3


def test_directory_table_shape():
    assert len(directory_table(list(_cat().values()))) == 3


def test_filters_known_mismatch_dropped_unknown_kept_optionally():
    cat = list(_cat().values())
    strict = {p.name for p in filter_companies(cat, branch="EEE", cgpa=6.5, include_unknown=False)}
    loose = {p.name for p in filter_companies(cat, branch="EEE", cgpa=6.5, include_unknown=True)}
    assert strict == {ORBIT}
    assert loose == {ORBIT, QUANTA}  # Quanta's CGPA is conflicting -> unknown; Nimbus excluded (branch)


def test_filter_salary_location_and_backlog():
    cat = list(_cat().values())
    assert {p.name for p in filter_companies(cat, ctc_min=10)} == {NIMBUS}
    assert {p.name for p in filter_companies(cat, location="pune")} == {QUANTA}
    assert {p.name for p in filter_companies(cat, active_backlogs=1, include_unknown=False)} == {ORBIT}
    assert {p.name for p in filter_companies(cat, employment_type="internship")} == {QUANTA}
