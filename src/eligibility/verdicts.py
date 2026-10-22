"""Result types shared by the eligibility engine and the UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    """Overall outcome for one company."""

    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT ELIGIBLE"
    UNCLEAR = "UNCLEAR"


class Outcome(str, Enum):
    """Outcome for a single criterion."""

    PASS = "Pass"
    FAIL = "Fail"
    UNCLEAR = "Unclear"
    NOT_STATED = "Not stated"  # optional criterion absent from documents; does not affect verdict


@dataclass(frozen=True)
class SourceRef:
    """Pointer to a real indexed chunk that supports a requirement."""

    chunk_id: str
    filename: str
    page: str
    section: str = ""
    effective_date: Optional[str] = None
    version: Optional[str] = None
    passage: str = ""

    def label(self) -> str:
        eff = f", effective {self.effective_date}" if self.effective_date else ""
        return f"{self.filename} (p. {self.page}{eff})"


@dataclass
class CriterionResult:
    """One row of the criterion-by-criterion breakdown."""

    field: str
    title: str
    requirement: str
    student_value: str
    outcome: Outcome
    detail: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    mandatory: bool = True


@dataclass
class EligibilityReport:
    """Full result for one company."""

    company: str
    verdict: Verdict
    reason: str
    results: list[CriterionResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[CriterionResult]:
        return [r for r in self.results if r.outcome == Outcome.FAIL]

    @property
    def unclear(self) -> list[CriterionResult]:
        return [r for r in self.results if r.outcome == Outcome.UNCLEAR]

    def all_sources(self) -> list[SourceRef]:
        seen: dict[str, SourceRef] = {}
        for r in self.results:
            for s in r.sources:
                seen.setdefault(s.chunk_id, s)
        return list(seen.values())
