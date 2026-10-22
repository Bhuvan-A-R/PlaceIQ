"""Side-by-side company comparison (missing values are shown as 'Not stated', never as zero)."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.companies.company_catalog import NOT_STATED, CompanyProfile

ATTRIBUTES: list[tuple[str, str]] = [
    ("Job role", "role"), ("Employment type", "employment_type"), ("Eligible branches", "branches_text"),
    ("Minimum CGPA", "min_cgpa_text"), ("Active backlogs", "backlog_text"), ("10th percentage", "tenth_text"),
    ("12th percentage", "twelfth_text"), ("CTC", "ctc_text"), ("Fixed component", "fixed_text"),
    ("Variable component", "variable_text"), ("Work location", "location"), ("Bond period", "bond_text"),
]


def _selection_rounds(p: CompanyProfile) -> str:
    return "; ".join(f"{i}. {r}" for i, r in enumerate(p.rounds, 1)) if p.rounds else NOT_STATED


def comparison_table(profiles: Sequence[CompanyProfile]) -> pd.DataFrame:
    """Attribute x company table. Index = attribute names, columns = company names."""
    data: dict[str, list[str]] = {}
    for p in profiles:
        col = []
        for _label, attr in ATTRIBUTES:
            v = getattr(p, attr)
            col.append(v if v not in (None, "") else NOT_STATED)
        col.append(_selection_rounds(p))
        col.append(p.last_verified or NOT_STATED)
        col.append(", ".join(p.documents) if p.documents else NOT_STATED)
        data[p.name] = col
    index = [a for a, _ in ATTRIBUTES] + ["Selection rounds", "Last verified (latest doc date)", "Source documents"]
    return pd.DataFrame(data, index=index)


def directory_table(profiles: Sequence[CompanyProfile]) -> pd.DataFrame:
    """One row per company for the explorer list."""
    rows = []
    for p in profiles:
        rows.append({
            "Company": p.name, "Role": p.role or NOT_STATED, "Type": p.employment_type or NOT_STATED,
            "Branches": p.branches_text, "Min CGPA": p.min_cgpa_text, "Backlogs": p.backlog_text,
            "CTC": p.ctc_text or NOT_STATED, "Location": p.location or NOT_STATED, "Bond": p.bond_text or NOT_STATED,
            "Rounds": str(len(p.rounds)) if p.rounds else NOT_STATED,
            "Last verified": p.last_verified or NOT_STATED,
        })
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8-sig")  # BOM so Excel opens UTF-8 correctly
