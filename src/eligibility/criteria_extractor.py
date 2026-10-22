"""Extract structured eligibility criteria from indexed chunks (no company rules are hardcoded).

Primary path: pipe-delimited table rows (``Criterion | Requirement | Notes``) whose first cell
is matched to a criterion field by keyword. Fallback: a few conservative prose patterns.

Each extracted :class:`Criterion` keeps a pointer to the chunk it came from, so every
requirement can be cited. If two documents state *different* values for the same field the
result is flagged as a conflict (see :meth:`CompanyCriteria.resolve`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from src.eligibility.verdicts import SourceRef
from src.retrieval.reranker import normalize_company
from src.schemas import Chunk

# Field identifiers
BRANCH = "branch"
MIN_CGPA = "min_cgpa"
ACTIVE_BACKLOGS = "active_backlogs"
HISTORY_BACKLOGS = "history_backlogs"
TENTH = "tenth_pct"
TWELFTH = "twelfth_pct"
GAP = "gap_years"

ALL_FIELDS = (BRANCH, MIN_CGPA, ACTIVE_BACKLOGS, HISTORY_BACKLOGS, TENTH, TWELFTH, GAP)
FIELD_TITLES = {
    BRANCH: "Branch", MIN_CGPA: "Minimum CGPA", ACTIVE_BACKLOGS: "Active backlogs",
    HISTORY_BACKLOGS: "History of backlogs", TENTH: "10th percentage", TWELFTH: "12th percentage",
    GAP: "Academic gap",
}

# Kinds of requirement
K_MIN, K_MAX, K_IN_SET, K_ANY, K_UNRESTRICTED = "min", "max", "in_set", "any", "unrestricted"

_LABELS: list[tuple[str, re.Pattern[str]]] = [  # order matters: more specific first
    (HISTORY_BACKLOGS, re.compile(r"(history|past|previous|total|cleared|ever)\b.*backlog|backlog.*history", re.I)),
    (ACTIVE_BACKLOGS, re.compile(r"(active|current|standing|pending)\s+(backlog|arrear)", re.I)),
    (GAP, re.compile(r"\b(gap|break in)\b|year\s*drop|academic gap", re.I)),
    (MIN_CGPA, re.compile(r"\b(cgpa|gpa|cpi)\b", re.I)),
    (TENTH, re.compile(r"\b(10th|tenth|x\s*std|ssc|sslc)\b", re.I)),
    (TWELFTH, re.compile(r"\b(12th|twelfth|xii|puc|hsc|intermediate)\b", re.I)),
    (BRANCH, re.compile(r"\b(branch|branches|specialisation|specialization|stream|department)\b", re.I)),
]
_BACKLOG_ANY = re.compile(r"backlog|arrear", re.I)

_WORD_NUM = {"zero": 0, "no": 0, "nil": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_ANY_BRANCH = re.compile(r"\b(all|any)\s+(engineering\s+)?(branches|streams|disciplines)\b", re.I)


@dataclass
class Criterion:
    """A single extracted requirement with provenance."""

    field: str
    kind: str
    value: Union[float, int, tuple[str, ...], None]
    raw: str
    source: SourceRef
    ambiguous: bool = False

    def comparable(self) -> object:
        if self.ambiguous:
            return ("ambiguous", self.raw.lower())
        return (self.kind, self.value)


@dataclass
class Resolution:
    """Outcome of resolving one field's candidate criteria."""

    status: str  # known | missing | conflict | ambiguous
    criterion: Optional[Criterion] = None
    candidates: list[Criterion] = field(default_factory=list)


@dataclass
class CompanyCriteria:
    """All extracted criteria for one company, with every candidate kept for conflict detection."""

    company: str
    by_field: dict[str, list[Criterion]] = field(default_factory=dict)

    def add(self, c: Criterion) -> None:
        self.by_field.setdefault(c.field, []).append(c)

    def resolve(self, field_name: str) -> Resolution:
        cands = self.by_field.get(field_name, [])
        if not cands:
            return Resolution("missing")
        if any(c.ambiguous for c in cands):
            return Resolution("ambiguous", None, cands)
        distinct = {c.comparable() for c in cands}
        # newest first so the primary criterion is the most recent document
        ordered = sorted(cands, key=lambda c: (c.source.effective_date or "", c.source.version or ""), reverse=True)
        if len(distinct) > 1:
            return Resolution("conflict", ordered[0], ordered)
        return Resolution("known", ordered[0], ordered)

    def fields_present(self) -> list[str]:
        return [f for f in ALL_FIELDS if f in self.by_field]


# ------------------------------------------------------------------ value parsing
def parse_number(text: str) -> Optional[float]:
    """First number in text, or a number word (zero/one/...). ``None`` if neither is present."""
    m = _NUM.search(text)
    if m:
        return float(m.group())
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _WORD_NUM:
            return float(_WORD_NUM[w])
    return None


def parse_branches(text: str) -> tuple[str, ...]:
    parts = re.split(r"[,/;&]|\band\b|\bor\b", text)
    out = []
    for p in parts:
        p = p.strip().strip(".")
        if p and len(p) <= 40:
            out.append(p.upper())
    return tuple(dict.fromkeys(out))


def _source(chunk: Chunk) -> SourceRef:
    return SourceRef(chunk_id=chunk.chunk_id, filename=chunk.source_filename, page=chunk.page_label,
                     section=chunk.section, effective_date=chunk.effective_date, version=chunk.version,
                     passage=chunk.text)


def classify_label(label: str) -> Optional[str]:
    for fld, pat in _LABELS:
        if pat.search(label):
            return fld
    if _BACKLOG_ANY.search(label):
        return ACTIVE_BACKLOGS + "?"  # ambiguous: 'backlogs' without active/history
    return None


def build_criterion(fld: str, requirement: str, source: SourceRef, notes: str = "") -> Optional[Criterion]:
    """Turn a (field, requirement text) pair into a normalized Criterion, or None if unusable."""
    req = requirement.strip()
    low = (req + " " + notes).lower()
    raw = req if not notes else f"{req} ({notes})"
    if fld == BRANCH:
        if _ANY_BRANCH.search(req):
            return Criterion(BRANCH, K_ANY, None, raw, source)
        br = parse_branches(req)
        return Criterion(BRANCH, K_IN_SET, br, raw, source) if br else None
    if fld in (ACTIVE_BACKLOGS, HISTORY_BACKLOGS, GAP):
        if re.search(r"not restricted|no restriction|unrestricted|any number|not applicable|no limit", low) and \
                not re.search(r"\d", req):
            return Criterion(fld, K_UNRESTRICTED, None, raw, source)
        n = parse_number(req)
        if n is None:
            return Criterion(fld, K_MAX, None, raw, source, ambiguous=True)
        return Criterion(fld, K_MAX, int(n), raw, source)
    if fld in (MIN_CGPA, TENTH, TWELFTH):
        m = _NUM.search(req)
        if not m:
            return Criterion(fld, K_MIN, None, raw, source, ambiguous=True)
        return Criterion(fld, K_MIN, float(m.group()), raw, source)
    return None


# ------------------------------------------------------------------ chunk scanning
_PROSE = [
    (MIN_CGPA, re.compile(r"(?:minimum|min\.?|at least)\s+cgpa\s*(?:of|:)?\s*(\d+(?:\.\d+)?)|cgpa\s+of\s+(\d+(?:\.\d+)?)\s+or\s+(?:above|higher|more)", re.I)),
    (TENTH, re.compile(r"(?:minimum\s+)?10th\s*(?:percentage|marks)?\s*(?:of|:)\s*(\d+(?:\.\d+)?)", re.I)),
    (TWELFTH, re.compile(r"(?:minimum\s+)?12th\s*(?:percentage|marks)?\s*(?:of|:)\s*(\d+(?:\.\d+)?)", re.I)),
]
_PROSE_CGPA_REVERSE = re.compile(r"\b(\d+(?:\.\d+)?)\s+cgpa\b", re.I)
_PROSE_BACKLOG_MAX = re.compile(r"(?:maximum|max\.?|at most|up to)\s+(?:of\s+)?(\w+)\s+active\s+backlogs?", re.I)
_PROSE_BACKLOG_NONE = re.compile(r"\b(?:no|zero)\s+active\s+backlogs?", re.I)
_PROSE_HIST_FREE = re.compile(r"history\s+of\s+backlogs?\s+(?:is|are)\s+not\s+restricted", re.I)
_PROSE_BRANCH = re.compile(
    r"(?:eligible|preferred)\s*branch(?:es)?\s*(?:are|:)\s*([^.\n]+)|"
    r"\bbranches\s*:\s*([^.\n]+)", re.I
)
_ELIGIBILITY_LABELS = {
    "Branch": BRANCH,
    "Current CGPA": MIN_CGPA,
    "Active backlogs": ACTIVE_BACKLOGS,
    "Backlogs cleared in the past": HISTORY_BACKLOGS,
    "10th percentage": TENTH,
    "12th percentage": TWELFTH,
}


def _extract_eligibility_table(text: str, source: SourceRef) -> list[Criterion]:
    """Extract criteria from the appended label/value eligibility section."""
    marker = "Candidate Eligibility Details"
    if marker not in text:
        marker = "Eligibility Details Summary"
    if marker not in text:
        return []
    lines = [line.strip() for line in text.splitlines()]
    start = lines.index(marker) + 1
    labels = set(_ELIGIBILITY_LABELS) | {
        "Other branch (exact name used by the company)",
        "Academic gap (years)",
        "Graduation status (informational)",
    }
    out: list[Criterion] = []
    for i in range(start, len(lines)):
        field = _ELIGIBILITY_LABELS.get(lines[i])
        if field is None:
            continue
        values: list[str] = []
        for value in lines[i + 1:]:
            if value in labels or value.startswith("Other branch") or \
                    value.startswith("Academic gap") or value.startswith("Graduation status") or \
                    value == "Fill the entry boxes in a PDF reader.":
                break
            if value and value not in {"Your entry", "Requirement in this document",
                                       "Detail"}:
                values.append(value)
        requirement = " ".join(values).strip()
        if not requirement or requirement.lower() in {"not specified", "not stated"}:
            continue
        criterion = build_criterion(field, requirement, source)
        if criterion:
            out.append(criterion)
    return out


def extract_from_chunk(chunk: Chunk) -> list[Criterion]:
    """All criteria found in one chunk (table rows first, then prose patterns)."""
    src = _source(chunk)
    out: list[Criterion] = []
    seen_fields: set[str] = set()
    for line in chunk.text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 2 or not cells[0] or not cells[1]:
            continue
        fld = classify_label(cells[0])
        if fld is None:
            continue
        if fld.endswith("?"):  # bare 'Backlogs' label: do not guess active vs history
            out.append(Criterion(ACTIVE_BACKLOGS, K_MAX, None, cells[1], src, ambiguous=True))
            seen_fields.add(ACTIVE_BACKLOGS)
            continue
        c = build_criterion(fld, cells[1], src, cells[2] if len(cells) > 2 else "")
        if c:
            out.append(c)
            seen_fields.add(fld)
    prose_text = chunk.text.split("Eligibility Details Summary", 1)[0]
    prose_text = prose_text.split("Candidate Eligibility Details", 1)[0]
    prose = "\n".join(l for l in prose_text.splitlines() if "|" not in l)
    for fld, pat in _PROSE:
        if fld in seen_fields:
            continue
        m = pat.search(prose)
        if m:
            val = next(g for g in m.groups() if g)
            out.append(Criterion(fld, K_MIN, float(val), m.group(0).strip(), src))
    if MIN_CGPA not in seen_fields:
        m = _PROSE_CGPA_REVERSE.search(prose)
        if m:
            out.append(Criterion(MIN_CGPA, K_MIN, float(m.group(1)), m.group(0), src))
    if ACTIVE_BACKLOGS not in seen_fields:
        m = _PROSE_BACKLOG_MAX.search(prose)
        if m and parse_number(m.group(1)) is not None:
            out.append(Criterion(ACTIVE_BACKLOGS, K_MAX, int(parse_number(m.group(1))), m.group(0), src))
        elif _PROSE_BACKLOG_NONE.search(prose):
            out.append(Criterion(ACTIVE_BACKLOGS, K_MAX, 0, _PROSE_BACKLOG_NONE.search(prose).group(0), src))
    if HISTORY_BACKLOGS not in seen_fields and _PROSE_HIST_FREE.search(prose):
        out.append(Criterion(HISTORY_BACKLOGS, K_UNRESTRICTED, None, _PROSE_HIST_FREE.search(prose).group(0), src))
    if BRANCH not in seen_fields:
        m = _PROSE_BRANCH.search(prose)
        if m:
            br = parse_branches(next(g for g in m.groups() if g))
            if br:
                out.append(Criterion(BRANCH, K_IN_SET, br, m.group(0).strip(), src))
    return out


ELIGIBILITY_DOC_TYPES = ("eligibility_circular", "company_jd", "unknown")


def company_chunks(chunks: Iterable[Chunk], company: str,
                   doc_types: Sequence[str] = ELIGIBILITY_DOC_TYPES) -> list[Chunk]:
    """Metadata-filtered chunks that belong to a company (company-less documents excluded)."""
    want = normalize_company(company)
    return [c for c in chunks if c.company and normalize_company(c.company) == want and c.doc_type in doc_types]


def extract_company_criteria(chunks: Iterable[Chunk], company: str) -> CompanyCriteria:
    """Build :class:`CompanyCriteria` from every eligibility-relevant chunk of ``company``."""
    crit = CompanyCriteria(company=company)
    relevant = company_chunks(chunks, company)
    for ch in relevant:
        for c in extract_from_chunk(ch):
            crit.add(c)
    # The generated eligibility summary can span chunk boundaries. Parse the
    # complete document once so labels and values remain associated.
    by_document: dict[str, list[Chunk]] = {}
    for ch in relevant:
        by_document.setdefault(ch.doc_id, []).append(ch)
    for document_chunks in by_document.values():
        ordered_chunks = sorted(document_chunks, key=lambda c: (c.page_start, c.chunk_id))
        if any(ch.synthetic for ch in ordered_chunks):
            continue
        combined = "\n".join(ch.text for ch in ordered_chunks)
        if "Eligibility Details Summary" not in combined and "Candidate Eligibility Details" not in combined:
            continue
        source_chunk = next(ch for ch in ordered_chunks if "Eligibility Details Summary" in ch.text
                            or "Candidate Eligibility Details" in ch.text)
        source = _source(source_chunk)
        for c in _extract_eligibility_table(combined, source):
            if c.field not in crit.by_field:
                crit.add(c)
    return crit
