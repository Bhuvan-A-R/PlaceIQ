"""Eligibility: boundary conditions, missing/conflicting criteria, extraction, deterministic verdicts."""
from __future__ import annotations

from conftest import sample_corpus

from src.eligibility.criteria_extractor import (
    ACTIVE_BACKLOGS, BRANCH, HISTORY_BACKLOGS, MIN_CGPA, CompanyCriteria, Criterion, K_IN_SET, K_MAX, K_MIN,
    extract_company_criteria, extract_from_chunk, parse_number,
)
from src.eligibility.rules_engine import (
    StudentProfile, check_branch, check_maximum, check_minimum, evaluate_company, evaluate_from_chunks,
)
from src.eligibility.verdicts import Outcome, SourceRef, Verdict
from src.schemas import Chunk

NIMBUS = "Nimbus Soft (synthetic)"
ORBIT = "Orbit Analytics (synthetic)"
QUANTA = "Quanta Grid Systems (synthetic)"


def _chunks():
    return sample_corpus()[1]


def _full(branch="CSE", cgpa=8.0, active=0, tenth=90.0, twelfth=90.0, **kw) -> StudentProfile:
    return StudentProfile(branch=branch, cgpa=cgpa, active_backlogs=active, tenth_pct=tenth, twelfth_pct=twelfth, **kw)


def _src(i="c1", date="2025-01-01") -> SourceRef:
    return SourceRef(i, f"{i}.pdf", "1", "S", date, "1.0", "text")


def _crit(**fields) -> CompanyCriteria:
    """Hand-built criteria so rules are tested independently of any PDF."""
    cc = CompanyCriteria("Test Co")
    for f, (kind, value) in fields.items():
        cc.add(Criterion(f, kind, value, str(value), _src()))
    return cc


# ----------------------------------------------------------- pure checks / boundaries
def test_minimum_is_inclusive():
    assert check_minimum(7.0, 7.0) == Outcome.PASS
    assert check_minimum(6.99, 7.0) == Outcome.FAIL
    assert check_minimum(7.01, 7.0) == Outcome.PASS


def test_minimum_float_noise_does_not_flip_result():
    assert check_minimum(0.1 + 0.2 + 6.7, 7.0) == Outcome.PASS  # 6.999999999999999


def test_maximum_is_inclusive():
    assert check_maximum(1, 1) == Outcome.PASS
    assert check_maximum(2, 1) == Outcome.FAIL
    assert check_maximum(0, 0) == Outcome.PASS


def test_missing_student_value_is_unclear_not_pass():
    assert check_minimum(None, 7.0) == Outcome.UNCLEAR
    assert check_maximum(None, 0) == Outcome.UNCLEAR
    assert check_branch(None, ["CSE"]) == Outcome.UNCLEAR


def test_branch_exact_match_only_no_equivalence():
    assert check_branch("cse", ["CSE", "ISE"]) == Outcome.PASS
    assert check_branch("C.S.E", ["CSE"]) == Outcome.PASS  # punctuation-insensitive only
    assert check_branch("Computer Science", ["CSE"]) == Outcome.FAIL
    assert check_branch("AIML", ["CSE", "ISE"]) == Outcome.FAIL


def test_parse_number_words_and_digits():
    assert parse_number("Zero active backlogs") == 0
    assert parse_number("Maximum of one active backlog") == 1
    assert parse_number("6.5") == 6.5
    assert parse_number("n/a") is None


# -------------------------------------------------------------------- verdict rules
def _all_known(**over):
    base = {BRANCH: (K_IN_SET, ("CSE", "ISE")), MIN_CGPA: (K_MIN, 7.0), ACTIVE_BACKLOGS: (K_MAX, 0),
            "tenth_pct": (K_MIN, 60.0), "twelfth_pct": (K_MIN, 60.0)}
    base.update(over)
    return _crit(**base)


def test_eligible_only_when_all_known_and_satisfied():
    r = evaluate_company(_full(cgpa=7.0), _all_known())
    assert r.verdict == Verdict.ELIGIBLE


def test_any_confirmed_failure_is_not_eligible():
    r = evaluate_company(_full(cgpa=6.9), _all_known())
    assert r.verdict == Verdict.NOT_ELIGIBLE
    assert [x.field for x in r.failed] == [MIN_CGPA]


def test_failure_beats_unclear():
    s = _full(cgpa=6.0)
    s.tenth_pct = None  # unclear, but CGPA fail is confirmed
    assert evaluate_company(s, _all_known()).verdict == Verdict.NOT_ELIGIBLE


def test_missing_student_input_gives_unclear():
    s = _full()
    s.active_backlogs = None
    assert evaluate_company(s, _all_known()).verdict == Verdict.UNCLEAR


def test_missing_core_criterion_in_documents_gives_unclear():
    cc = _all_known()
    del cc.by_field["twelfth_pct"]
    r = evaluate_company(_full(), cc)
    assert r.verdict == Verdict.UNCLEAR
    assert any(x.field == "twelfth_pct" and x.outcome == Outcome.UNCLEAR for x in r.results)


def test_optional_criterion_absent_is_not_counted_but_warned():
    r = evaluate_company(_full(), _all_known())
    row = next(x for x in r.results if x.field == HISTORY_BACKLOGS)
    assert row.outcome == Outcome.NOT_STATED
    assert any("not stated" in w.lower() for w in r.warnings)


def test_stated_optional_criterion_is_enforced():
    cc = _all_known()
    cc.add(Criterion(HISTORY_BACKLOGS, K_MAX, 0, "0", _src()))
    assert evaluate_company(_full(history_backlogs=2), cc).verdict == Verdict.NOT_ELIGIBLE
    assert evaluate_company(_full(history_backlogs=None), cc).verdict == Verdict.UNCLEAR
    assert evaluate_company(_full(history_backlogs=0), cc).verdict == Verdict.ELIGIBLE


def test_active_and_history_backlogs_are_separate():
    cc = _all_known()
    cc.add(Criterion(HISTORY_BACKLOGS, K_MAX, 2, "2", _src()))
    r = evaluate_company(_full(active=0, history_backlogs=2), cc)
    assert r.verdict == Verdict.ELIGIBLE
    assert evaluate_company(_full(active=1, history_backlogs=0), cc).verdict == Verdict.NOT_ELIGIBLE


def test_conflicting_documents_give_unclear_and_show_both_sources():
    cc = _all_known()
    cc.by_field[MIN_CGPA] = [Criterion(MIN_CGPA, K_MIN, 6.0, "6.0", _src("old", "2025-01-01")),
                             Criterion(MIN_CGPA, K_MIN, 6.5, "6.5", _src("new", "2025-06-01"))]
    r = evaluate_company(_full(cgpa=6.2), cc)
    row = next(x for x in r.results if x.field == MIN_CGPA)
    assert row.outcome == Outcome.UNCLEAR
    assert {s.chunk_id for s in row.sources} == {"old", "new"}
    assert r.verdict == Verdict.UNCLEAR


def test_same_value_in_two_documents_is_not_a_conflict():
    cc = _all_known()
    cc.add(Criterion(MIN_CGPA, K_MIN, 7.0, "7.0", _src("dup", "2025-02-01")))
    assert cc.resolve(MIN_CGPA).status == "known"
    assert evaluate_company(_full(), cc).verdict == Verdict.ELIGIBLE


def test_ambiguous_requirement_is_unclear():
    cc = _all_known()
    cc.by_field[ACTIVE_BACKLOGS] = [Criterion(ACTIVE_BACKLOGS, K_MAX, None, "as per norms", _src(), ambiguous=True)]
    assert evaluate_company(_full(), cc).verdict == Verdict.UNCLEAR


def test_every_requirement_row_carries_a_source():
    r = evaluate_company(_full(), _all_known())
    for row in r.results:
        if row.outcome != Outcome.NOT_STATED:
            assert row.sources, row.field


# ----------------------------------------------------------- extraction from real chunks
def test_extracts_criteria_from_sample_circulars():
    crit = extract_company_criteria(_chunks(), NIMBUS)
    assert crit.resolve(MIN_CGPA).criterion.value == 7.0
    assert crit.resolve(ACTIVE_BACKLOGS).criterion.value == 0
    assert set(crit.resolve(BRANCH).criterion.value) == {"CSE", "ISE", "AIML"}
    src = crit.resolve(MIN_CGPA).criterion.source
    assert src.filename == "nimbus_soft_eligibility_SYNTHETIC.pdf" and src.page == "1"


def test_prose_history_not_restricted_is_extracted():
    crit = extract_company_criteria(_chunks(), ORBIT)
    assert crit.resolve(HISTORY_BACKLOGS).status == "known"
    assert crit.resolve(HISTORY_BACKLOGS).criterion.kind == "unrestricted"


def test_extraction_prose_fallback_and_ambiguous_label():
    ch = Chunk(chunk_id="x1", doc_id="d", source_filename="x.pdf", doc_type="eligibility_circular", company="X",
               section="Rules", page_start=1, page_end=1,
               text="Minimum CGPA of 7.5 is required.\nNo active backlogs.\nBacklogs | as per university norms")
    got = {c.field: c for c in extract_from_chunk(ch)}
    assert got[MIN_CGPA].value == 7.5
    assert got[ACTIVE_BACKLOGS].ambiguous  # a bare 'Backlogs' row must not be guessed


def test_unknown_company_is_unclear_not_eligible():
    r = evaluate_from_chunks(_full(), _chunks(), "Nonexistent Corp")
    assert r.verdict == Verdict.UNCLEAR


# ------------------------------------------------- end-to-end on the synthetic corpus
def test_example_from_spec_one_backlog_is_not_eligible():
    s = StudentProfile("AIML", 7.2, 1, None, 80, 80)
    r = evaluate_from_chunks(s, _chunks(), NIMBUS)
    assert r.verdict == Verdict.NOT_ELIGIBLE
    assert [x.field for x in r.failed] == [ACTIVE_BACKLOGS]


def test_orbit_allows_one_backlog_and_is_eligible():
    r = evaluate_from_chunks(StudentProfile("AIML", 7.2, 1, None, 80, 80), _chunks(), ORBIT)
    assert r.verdict == Verdict.ELIGIBLE


def test_orbit_cgpa_boundary_6_5_inclusive():
    assert evaluate_from_chunks(_full("ECE", 6.5, 0), _chunks(), ORBIT).verdict == Verdict.ELIGIBLE
    assert evaluate_from_chunks(_full("ECE", 6.49, 0), _chunks(), ORBIT).verdict == Verdict.NOT_ELIGIBLE


def test_quanta_conflicting_circulars_give_unclear_for_cgpa_between_thresholds():
    r = evaluate_from_chunks(_full("EEE", 6.2, 0, tenth=70, twelfth=70), _chunks(), QUANTA)
    cg = next(x for x in r.results if x.field == MIN_CGPA)
    assert cg.outcome == Outcome.UNCLEAR and len(cg.sources) == 2
    assert r.verdict == Verdict.UNCLEAR


def test_quanta_wrong_branch_is_not_eligible_even_with_conflict():
    r = evaluate_from_chunks(_full("CSE", 8.0, 0), _chunks(), QUANTA)
    assert r.verdict == Verdict.NOT_ELIGIBLE


def test_no_personal_identifier_fields_exist():
    fields = set(StudentProfile.__dataclass_fields__)
    assert not ({"name", "usn", "phone", "email"} & fields)
