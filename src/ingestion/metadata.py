"""Document-level metadata detection (type, company, effective date, version).

Priority: explicit header lines in the document ("Document Type: ...") > filename
convention (``<doctype>__<company>.pdf``-style keywords) > ``unknown``. Nothing is guessed
beyond what the document or filename actually states.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.schemas import DOC_TYPES, PageText

_TYPE_ALIASES = {
    "placement policy": "placement_policy",
    "policy": "placement_policy",
    "company job description": "company_jd",
    "job description": "company_jd",
    "jd": "company_jd",
    "company eligibility circular": "eligibility_circular",
    "eligibility circular": "eligibility_circular",
    "circular": "eligibility_circular",
    "placement faq": "placement_faq",
    "faq": "placement_faq",
    "recruitment process": "recruitment_process",
    "recruitment process document": "recruitment_process",
}

_FILENAME_HINTS = [
    ("policy", "placement_policy"),
    ("faq", "placement_faq"),
    ("circular", "eligibility_circular"),
    ("eligibility", "eligibility_circular"),
    ("recruitment", "recruitment_process"),
    ("process", "recruitment_process"),
    ("jd", "company_jd"),
    ("job", "company_jd"),
]

_COMPANY_FILENAME_HINTS = [
    ("nimbus_soft", "Nimbus Soft (synthetic)"),
    ("orbit_analytics", "Orbit Analytics (synthetic)"),
    ("quanta_grid", "Quanta Grid Systems (synthetic)"),
    ("control tower", "Duroflex"),
    ("demandnxt", "DemandNXT Business Services Pvt. Ltd."),
    ("sfl mt", "Shriram Finance"),
    ("gammastack", "GammaStack"),
    ("wipfli", "Wipfli India LLP"),
    ("ibm-cio", "IBM CIO"),
]


def _header_value(text: str, key: str) -> Optional[str]:
    m = re.search(rf"^\s*{key}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def normalize_doc_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower().replace("_", " ")
    if v in _TYPE_ALIASES:
        return _TYPE_ALIASES[v]
    if v.replace(" ", "_") in DOC_TYPES:
        return v.replace(" ", "_")
    return None


def detect_metadata(pages: list[PageText], filename: str) -> dict:
    """Return ``doc_type``, ``company``, ``effective_date``, ``version``, ``synthetic``."""
    head = "\n".join(p.text for p in pages[:2])[:3000]
    doc_type = normalize_doc_type(_header_value(head, "document type"))
    if doc_type is None:
        stem = Path(filename).stem.lower()
        for hint, dt in _FILENAME_HINTS:
            if hint in stem:
                doc_type = dt
                break
    company = _header_value(head, "company")
    if not company:
        lowered_filename = Path(filename).stem.lower()
        for hint, name in _COMPANY_FILENAME_HINTS:
            if hint in lowered_filename:
                company = name
                break
    if company and company.lower() in {"n/a", "na", "none", "all", "-"}:
        company = None
    effective = _header_value(head, "effective date")
    version = _header_value(head, "version")
    synthetic = bool(re.search(r"synthetic", head, re.IGNORECASE))
    return {
        "doc_type": doc_type or "unknown",
        "company": company,
        "effective_date": effective,
        "version": version,
        "synthetic": synthetic,
    }
