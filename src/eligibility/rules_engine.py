"""Deterministic eligibility rules. No LLM is involved anywhere in this module.

Each check is a small pure function that is independently testable. The engine combines them:

* any confirmed mandatory failure             -> NOT ELIGIBLE
* else any mandatory criterion unclear        -> UNCLEAR
  (missing, ambiguous, conflicting between documents, or the student value is not provided)
* else                                        -> ELIGIBLE

Thresholds are inclusive: a minimum CGPA of 7.0 is satisfied by 7.0. Branches are compared by
exact (normalised) name; no equivalence between branches is ever inferred.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from src.eligibility.criteria_extractor import (
    ACTIVE_BACKLOGS, ALL_FIELDS, BRANCH, FIELD_TITLES, GAP, HISTORY_BACKLOGS, K_ANY, K_IN_SET, K_MAX,
    K_MIN, K_UNRESTRICTED, MIN_CGPA, TENTH, TWELFTH, CompanyCriteria, Criterion, extract_company_criteria,
)
from src.eligibility.verdicts import CriterionResult, EligibilityReport, Outcome, SourceRef, Verdict
from src.schemas import Chunk

# Criteria that must be stated in the documents; if absent the verdict cannot be ELIGIBLE.
CORE_FIELDS = (BRANCH, MIN_CGPA, ACTIVE_BACKLOGS, TENTH, TWELFTH)
_EPS = 1e-9


@dataclass
class StudentProfile:
    """Anonymous academic profile. No name/USN/phone fields exist by design."""

    branch: Optional[str] = None
    cgpa: Optional[float] = None
    active_backlogs: Optional[int] = None
    history_backlogs: Optional[int] = None  # backlogs cleared in the past
    tenth_pct: Optional[float] = None
    twelfth_pct: Optional[float] = None
    gap_years: Optional[float] = None
    graduation_status: Optional[str] = None  # informational only; no company criterion uses it


# --------------------------------------------------------------- pure check functions
def normalize_branch(name: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def check_minimum(value: Optional[float], threshold: float) -> Outcome:
    """Inclusive minimum: ``value >= threshold``. ``None`` -> UNCLEAR."""
    if value is None:
        return Outcome.UNCLEAR
    return Outcome.PASS if value + _EPS >= threshold else Outcome.FAIL


def check_maximum(value: Optional[float], limit: float) -> Outcome:
    """Inclusive maximum: ``value <= limit``. ``None`` -> UNCLEAR."""
    if value is None:
        return Outcome.UNCLEAR
    return Outcome.PASS if value <= limit + _EPS else Outcome.FAIL


def check_branch(student_branch: Optional[str], allowed: Sequence[str]) -> Outcome:
    """Exact normalised match against the allowed list (no branch equivalence inferred)."""
    if not student_branch or not normalize_branch(student_branch):
        return Outcome.UNCLEAR
    allowed_norm = {normalize_branch(b) for b in allowed}
    return Outcome.PASS if normalize_branch(student_branch) in allowed_norm else Outcome.FAIL


def describe_requirement(c: Criterion) -> str:
    if c.ambiguous:
        return f"Ambiguous: \"{c.raw}\""
    if c.kind == K_ANY:
        return "All branches"
    if c.kind == K_UNRESTRICTED:
        return "Not restricted"
    if c.kind == K_IN_SET:
        return ", ".join(c.value)  # type: ignore[arg-type]
    if c.kind == K_MIN:
        v = float(c.value)  # type: ignore[arg-type]
        if c.field == MIN_CGPA:
            return f"At least {v:.1f}"
        return f"At least {v:g}{'%' if c.field in (TENTH, TWELFTH) else ''}"
    if c.kind == K_MAX:
        return f"At most {c.value}"
    return str(c.value)


def evaluate_criterion(c: Criterion, student: StudentProfile) -> tuple[Outcome, str, str]:
    """Return (outcome, student_value_text, detail) for one non-conflicting criterion."""
    f = c.field
    if f == BRANCH:
        sv = student.branch
        svt = sv or "not provided"
        if c.kind == K_ANY:
            return Outcome.PASS, svt, "Document states all branches are eligible."
        out = check_branch(sv, c.value)  # type: ignore[arg-type]
        return out, svt, ("" if out != Outcome.FAIL else "Branch is not in the listed eligible branches.") \
            if out != Outcome.UNCLEAR else "Enter your branch to evaluate this criterion."
    mapping = {MIN_CGPA: student.cgpa, ACTIVE_BACKLOGS: student.active_backlogs,
               HISTORY_BACKLOGS: student.history_backlogs, TENTH: student.tenth_pct,
               TWELFTH: student.twelfth_pct, GAP: student.gap_years}
    sv = mapping.get(f)
    svt = "not provided" if sv is None else (f"{sv:.1f}" if f == MIN_CGPA else f"{sv:g}")
    if c.kind == K_UNRESTRICTED:
        return Outcome.PASS, svt, "Document states this is not restricted."
    if c.value is None:
        return Outcome.UNCLEAR, svt, "The requirement could not be read unambiguously from the document."
    out = check_minimum(sv, float(c.value)) if c.kind == K_MIN else check_maximum(sv, float(c.value))
    detail = "Enter your value to evaluate this criterion." if out == Outcome.UNCLEAR else ""
    return out, svt, detail


# ------------------------------------------------------------------- engine
def _rows_for_field(f: str, crit: CompanyCriteria, student: StudentProfile) -> CriterionResult:
    title = FIELD_TITLES[f]
    core = f in CORE_FIELDS
    res = crit.resolve(f)
    if res.status == "missing":
        if core:
            return CriterionResult(f, title, "Not stated in indexed documents", "-", Outcome.UNCLEAR,
                                   "Required criterion is missing from the documents; it is not assumed satisfied.", [])
        return CriterionResult(f, title, "Not stated in indexed documents", "-", Outcome.NOT_STATED,
                               "No requirement found; not counted either way.", [], mandatory=False)
    srcs = list({c.source.chunk_id: c.source for c in res.candidates}.values())
    if res.status == "ambiguous":
        raw = "; ".join(c.raw for c in res.candidates)
        return CriterionResult(f, title, f"Ambiguous: {raw}", "-", Outcome.UNCLEAR,
                               "The document wording does not state the requirement precisely.", srcs)
    if res.status == "conflict":
        reqs = [f"{describe_requirement(c)} [{c.source.filename}, effective {c.source.effective_date or 'unknown'}]"
                for c in res.candidates]
        outs = {evaluate_criterion(c, student)[0] for c in res.candidates}
        extra = ""
        if outs == {Outcome.FAIL}:
            extra = " You would not meet any of the conflicting versions, but the applicable rule must be confirmed."
        elif outs == {Outcome.PASS}:
            extra = " You would meet all of the conflicting versions, but the applicable rule must be confirmed."
        svt = evaluate_criterion(res.candidates[0], student)[1]
        return CriterionResult(f, title, " vs ".join(reqs), svt, Outcome.UNCLEAR,
                               "Documents conflict; confirm the applicable rule with the placement cell." + extra, srcs)
    c = res.criterion
    assert c is not None
    out, svt, detail = evaluate_criterion(c, student)
    return CriterionResult(f, title, describe_requirement(c), svt, out, detail, srcs)


def _verdict(company: str, results: list[CriterionResult]) -> tuple[Verdict, str]:
    mand = [r for r in results if r.mandatory]
    fails = [r for r in mand if r.outcome == Outcome.FAIL]
    if fails:
        names = ", ".join(r.title for r in fails)
        return Verdict.NOT_ELIGIBLE, f"The profile does not satisfy: {names}."
    unclear = [r for r in mand if r.outcome == Outcome.UNCLEAR]
    if unclear:
        names = ", ".join(r.title for r in unclear)
        return Verdict.UNCLEAR, f"No confirmed failure, but these criteria could not be confirmed: {names}."
    return Verdict.ELIGIBLE, "Every applicable mandatory criterion found in the documents is known and satisfied."


def evaluate_company(student: StudentProfile, criteria: CompanyCriteria) -> EligibilityReport:
    """Evaluate a student against one company's extracted criteria."""
    results = [_rows_for_field(f, criteria, student) for f in ALL_FIELDS]
    verdict, reason = _verdict(criteria.company, results)
    warnings: list[str] = []
    if any(r.outcome == Outcome.NOT_STATED for r in results):
        warnings.append("Some optional criteria (e.g. history of backlogs, academic gap) are not stated in the "
                        "indexed documents and were not evaluated.")
    if any("conflict" in r.detail.lower() for r in results):
        warnings.append("Conflicting documents were found; both sources are shown. Confirm with the placement cell.")
    warnings.append("Verdict is a decision aid computed from indexed documents. It is not an official decision; "
                    "the placement cell's rules take precedence.")
    return EligibilityReport(criteria.company, verdict, reason, results, warnings)


def evaluate_from_chunks(student: StudentProfile, chunks: Iterable[Chunk], company: str) -> EligibilityReport:
    """Extract criteria for ``company`` from chunks and evaluate. Unknown company -> UNCLEAR."""
    chunk_list = list(chunks)
    return evaluate_company(student, extract_company_criteria(chunk_list, company))
