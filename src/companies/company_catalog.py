"""Company directory built from indexed chunks. Missing data stays missing (``None``), never zero.

Fields are read from pipe-delimited table rows in company job-description / circular chunks
(``Item | Details``, ``Round N | description``). Nothing is invented to fill an empty field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from src.eligibility.criteria_extractor import (
    ACTIVE_BACKLOGS, BRANCH, K_ANY, K_IN_SET, K_MAX, K_MIN, K_UNRESTRICTED, MIN_CGPA, TENTH, TWELFTH,
    CompanyCriteria, extract_company_criteria,
)
from src.eligibility.verdicts import SourceRef
from src.retrieval.reranker import normalize_company
from src.schemas import Chunk

NOT_STATED = "Not stated"

_LABEL_PATTERNS = {
    "role": re.compile(r"^(job\s+)?(role|title|position|designation)\b", re.I),
    "employment_type": re.compile(r"employment|job type|internship|full[- ]?time|offer type", re.I),
    "ctc": re.compile(r"^(total\s+)?(ctc|package|compensation|salary)\b", re.I),
    "fixed": re.compile(r"fixed", re.I),
    "variable": re.compile(r"variable", re.I),
    "location": re.compile(r"location|place of work|posting", re.I),
    "bond": re.compile(r"bond|service agreement|service period", re.I),
}
_ROUND = re.compile(r"^(round|stage)\s*(\d+)\b", re.I)
_LPA = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?(?:\s+per\s+annum)?|l\.?p\.?a)", re.I)
_MONTHS = re.compile(r"(\d+(?:\.\d+)?)\s*(month|year|yr)s?", re.I)
_NO_BOND = re.compile(r"\b(no|nil|none|without)\b.*\bbond\b|^\s*(no|nil|none)\s*bond|^no bond$|^none$|^nil$", re.I)


def parse_lpa(text: Optional[str]) -> Optional[float]:
    """Lakhs per annum from text like '12 LPA'; ``None`` if no LPA figure is present."""
    if not text:
        return None
    m = _LPA.search(text)
    return float(m.group(1)) if m else None


def parse_bond_months(text: Optional[str]) -> Optional[float]:
    """Months of bond. 'No bond' -> 0.0 (explicitly stated); unrecognised text -> ``None``."""
    if not text:
        return None
    if _NO_BOND.search(text.strip()):
        return 0.0
    m = _MONTHS.search(text)
    if not m:
        return None
    n = float(m.group(1))
    return n * 12 if m.group(2).lower().startswith(("y",)) else n


@dataclass
class CompanyProfile:
    """Everything known about one company from the indexed documents."""

    name: str
    role: Optional[str] = None
    employment_type: Optional[str] = None
    ctc_text: Optional[str] = None
    ctc_lpa: Optional[float] = None
    fixed_text: Optional[str] = None
    variable_text: Optional[str] = None
    location: Optional[str] = None
    bond_text: Optional[str] = None
    bond_months: Optional[float] = None
    rounds: list[str] = field(default_factory=list)
    criteria: Optional[CompanyCriteria] = None
    sources: dict[str, SourceRef] = field(default_factory=dict)  # attribute -> source
    documents: list[str] = field(default_factory=list)
    last_verified: Optional[str] = None  # latest effective date among the company's documents
    synthetic: bool = False

    # ------- criteria helpers (display strings; None-safe)
    def _crit_text(self, f: str) -> str:
        if not self.criteria:
            return NOT_STATED
        res = self.criteria.resolve(f)
        if res.status == "missing":
            return NOT_STATED
        if res.status == "ambiguous":
            return "Ambiguous in source"
        from src.eligibility.rules_engine import describe_requirement
        if res.status == "conflict":
            return " | ".join(f"{describe_requirement(c)} ({c.source.effective_date or '?'})" for c in res.candidates) \
                + " (CONFLICT)"
        return describe_requirement(res.criterion)  # type: ignore[arg-type]

    @property
    def branches_text(self) -> str:
        return self._crit_text(BRANCH)

    @property
    def min_cgpa_text(self) -> str:
        return self._crit_text(MIN_CGPA)

    @property
    def backlog_text(self) -> str:
        return self._crit_text(ACTIVE_BACKLOGS)

    @property
    def tenth_text(self) -> str:
        return self._crit_text(TENTH)

    @property
    def twelfth_text(self) -> str:
        return self._crit_text(TWELFTH)

    def known_values(self, f: str) -> list:
        """Distinct criterion values for field f (several if documents conflict)."""
        if not self.criteria:
            return []
        return [c for c in self.criteria.resolve(f).candidates]


def _src(chunk: Chunk) -> SourceRef:
    return SourceRef(chunk.chunk_id, chunk.source_filename, chunk.page_label, chunk.section,
                     chunk.effective_date, chunk.version, chunk.text)


def build_profile(chunks: Sequence[Chunk], company: str) -> CompanyProfile:
    want = normalize_company(company)
    mine = [c for c in chunks if c.company and normalize_company(c.company) == want]
    prof = CompanyProfile(name=company)
    rounds: dict[int, tuple[str, SourceRef]] = {}
    for ch in sorted(mine, key=lambda c: (c.effective_date or "", c.chunk_id)):  # newest value wins
        prof.documents.append(ch.source_filename) if ch.source_filename not in prof.documents else None
        if ch.effective_date and ch.effective_date > (prof.last_verified or ""):
            prof.last_verified = ch.effective_date
        prof.synthetic = prof.synthetic or ch.synthetic
        if ch.doc_type not in ("company_jd", "eligibility_circular", "recruitment_process"):
            continue
        for line in ch.text.splitlines():
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 2 or not cells[0] or not cells[1]:
                continue
            label, value = cells[0], cells[1]
            m = _ROUND.match(label)
            if m:
                rounds[int(m.group(2))] = (value, _src(ch))
                continue
            if ch.doc_type != "company_jd":
                continue
            low = label.lower()
            if low in ("item", "criterion"):
                continue
            for attr in ("fixed", "variable", "ctc", "bond", "location", "employment_type", "role"):
                if _LABEL_PATTERNS[attr].search(label):
                    _assign(prof, attr, value, _src(ch))
                    break
    prof.rounds = [rounds[k][0] for k in sorted(rounds)]
    if rounds:
        prof.sources["rounds"] = rounds[sorted(rounds)[0]][1]
    prof.criteria = extract_company_criteria(mine, company)
    return prof


def _assign(p: CompanyProfile, attr: str, value: str, src: SourceRef) -> None:
    if attr == "ctc":
        p.ctc_text, p.ctc_lpa = value, parse_lpa(value)
    elif attr == "fixed":
        p.fixed_text = value
    elif attr == "variable":
        p.variable_text = value
    elif attr == "bond":
        p.bond_text, p.bond_months = value, parse_bond_months(value)
    elif attr == "location":
        p.location = value
    elif attr == "employment_type":
        p.employment_type = value
    elif attr == "role":
        p.role = value
    p.sources[attr] = src


def build_catalog(chunks: Iterable[Chunk]) -> dict[str, CompanyProfile]:
    """One :class:`CompanyProfile` per company name present in the indexed chunks."""
    chunk_list = list(chunks)
    names = sorted({c.company for c in chunk_list if c.company}, key=str.lower)
    # merge names that normalise identically (e.g. 'X' and 'X (synthetic)')
    merged: dict[str, str] = {}
    for n in names:
        merged.setdefault(normalize_company(n), n)
    return {n: build_profile(chunk_list, n) for n in merged.values()}


# --------------------------------------------------------------------- filtering
def _matches_text(value: Optional[str], needle: str) -> Optional[bool]:
    if not needle:
        return True
    if value is None:
        return None
    return needle.lower() in value.lower()


def filter_companies(
    profiles: Iterable[CompanyProfile],
    branch: Optional[str] = None,
    cgpa: Optional[float] = None,
    active_backlogs: Optional[int] = None,
    ctc_min: Optional[float] = None,
    ctc_max: Optional[float] = None,
    location: str = "",
    employment_type: str = "",
    role: str = "",
    include_unknown: bool = True,
) -> list[CompanyProfile]:
    """Filter by facts that are *known* from documents.

    A filter that cannot be evaluated because the document is silent counts as "unknown":
    kept when ``include_unknown`` is True, dropped otherwise. Known mismatches are always dropped.
    Conflicting values are treated as unknown.
    """
    from src.eligibility.rules_engine import check_branch, check_maximum, check_minimum

    out: list[CompanyProfile] = []
    for p in profiles:
        checks: list[Optional[bool]] = []
        if branch:
            res = p.criteria.resolve(BRANCH) if p.criteria else None
            if res is None or res.status != "known":
                checks.append(None)
            elif res.criterion.kind == K_ANY:  # type: ignore[union-attr]
                checks.append(True)
            else:
                checks.append(check_branch(branch, res.criterion.value).value == "Pass")  # type: ignore[union-attr]
        if cgpa is not None:
            res = p.criteria.resolve(MIN_CGPA) if p.criteria else None
            checks.append(None if res is None or res.status != "known"
                          else check_minimum(cgpa, float(res.criterion.value)).value == "Pass")  # type: ignore[union-attr]
        if active_backlogs is not None:
            res = p.criteria.resolve(ACTIVE_BACKLOGS) if p.criteria else None
            if res is None or res.status != "known":
                checks.append(None)
            elif res.criterion.kind == K_UNRESTRICTED:  # type: ignore[union-attr]
                checks.append(True)
            else:
                checks.append(check_maximum(active_backlogs, float(res.criterion.value)).value == "Pass")  # type: ignore[union-attr]
        if ctc_min is not None or ctc_max is not None:
            if p.ctc_lpa is None:
                checks.append(None)
            else:
                checks.append((ctc_min is None or p.ctc_lpa >= ctc_min) and (ctc_max is None or p.ctc_lpa <= ctc_max))
        checks.append(_matches_text(p.location, location))
        checks.append(_matches_text(p.employment_type, employment_type))
        checks.append(_matches_text(p.role, role))
        if any(c is False for c in checks):
            continue
        if any(c is None for c in checks) and not include_unknown:
            continue
        out.append(p)
    return out
