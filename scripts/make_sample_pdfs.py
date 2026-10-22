"""Generate the SYNTHETIC sample placement PDFs in data/sample/.

Every company, number and rule below is invented for demonstration and testing. None of it
is real or official. Each PDF states this in its header.

Usage:
    python -m scripts.make_sample_pdfs
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
DISCLAIMER = "Synthetic sample document for PlaceBot demos and tests - not an official placement document."

PAGEBREAK = "<<PAGEBREAK>>"


def header(doc_type: str, company: str, effective: str, version: str) -> list[str]:
    return [
        DISCLAIMER,
        f"Document Type: {doc_type}",
        f"Company: {company}",
        f"Effective Date: {effective}",
        f"Version: {version}",
        "",
    ]


def company_jd(name: str, role: str, rows: list[tuple[str, str]], about: str, rounds: list[str], extra: str) -> list[str]:
    lines = header("Company Job Description", name, "2025-09-01", "1.0")
    lines += ["1. About the Company", about, ""]
    lines += ["2. Job Role and Compensation", "Item | Details"]
    lines += [f"{k} | {v}" for k, v in rows]
    lines += ["", "3. Selection Rounds"]
    lines += [f"Round {i} | {r}" for i, r in enumerate(rounds, 1)]
    lines += ["", "4. Additional Notes", extra]
    return lines


def eligibility(name: str, version: str, effective: str, rows: list[tuple[str, str, str]], note: str) -> list[str]:
    lines = header("Company Eligibility Circular", name, effective, version)
    lines += ["1. Eligibility Criteria", "Criterion | Requirement | Notes"]
    lines += [f"{a} | {b} | {c}" for a, b, c in rows]
    lines += ["", "2. Circular Notes", note]
    return lines


DOCS: dict[str, list[str]] = {}

DOCS["placement_policy_SYNTHETIC.pdf"] = (
    header("Placement Policy", "N/A", "2025-08-01", "1.0")
    + [
        "1. Purpose and Scope",
        "This synthetic policy describes example rules for a fictional campus placement cell. It is used only to demonstrate "
        "PlaceBot and must not be treated as the rules of any real institution.",
        "",
        "2. Company Categories",
        "Category | CTC band | Notes",
        "Regular | Below 8 LPA | Open to all eligible students",
        "Dream | 8 LPA to below 15 LPA | Students with a Regular offer may attend",
        "Super Dream | 15 LPA and above | Students with a Regular or Dream offer may attend",
        "",
        PAGEBREAK,
        "3. One-Offer Rule",
        "3.1 A student who has accepted a Regular category offer may attend Dream and Super Dream drives only.",
        "3.2 A student who has accepted a Dream category offer may attend Super Dream drives only and may not attend "
        "Regular or other Dream drives.",
        "3.3 A student who has accepted a Super Dream offer is not eligible for any further drive.",
        "",
        "4. Offer Acceptance",
        "4.1 An offer is considered accepted when the student signs the offer acknowledgement issued by the placement cell.",
        "4.2 Once accepted, an offer cannot be exchanged for another offer of the same or lower category.",
        "",
        "5. Debarment",
        "5.1 A student who is absent from a company drive after registering, without written permission from the placement "
        "cell, may be debarred from the next two drives.",
        "5.2 A student who withdraws from a selection process after the first technical round may be debarred from "
        "the remainder of the placement season.",
        "",
        "6. Mandatory Drive Attendance",
        "6.1 Registration for a drive makes attendance mandatory for all rounds of that drive.",
        "",
        "7. Withdrawal from a Process",
        "7.1 A student may withdraw before the first round by informing the placement cell in writing at least 24 hours "
        "before the drive.",
        "7.2 Withdrawals after the first round are governed by clause 5.2.",
        "",
        "8. Company-Specific Exceptions",
        "8.1 A company circular may impose stricter conditions than this policy. Where a company circular and this policy "
        "differ, the company circular for that drive applies.",
    ]
)

DOCS["nimbus_soft_JD_SYNTHETIC.pdf"] = company_jd(
    "Nimbus Soft (synthetic)",
    "Software Development Engineer",
    [
        ("Job role", "Software Development Engineer"),
        ("Employment type", "Full-time"),
        ("CTC", "12 LPA"),
        ("Fixed component", "10 LPA"),
        ("Variable component", "2 LPA"),
        ("Work location", "Bengaluru"),
        ("Bond period", "12 months"),
    ],
    "Nimbus Soft is a fictional cloud software company created for PlaceBot demonstrations.",
    ["Online coding assessment", "Technical interview 1", "Technical interview 2", "HR interview"],
    "Candidates must carry a valid college ID and a laptop for the online assessment.",
)
DOCS["nimbus_soft_eligibility_SYNTHETIC.pdf"] = eligibility(
    "Nimbus Soft (synthetic)",
    "1.0",
    "2025-09-01",
    [
        ("Eligible branches", "CSE, ISE, AIML", "Other branches are not eligible"),
        ("Minimum CGPA", "7.0", "On a 10 point scale, 7.0 qualifies"),
        ("Active backlogs", "0", "No active backlogs allowed"),
        ("10th percentage", "70", "Minimum"),
        ("12th percentage", "70", "Minimum"),
    ],
    "This synthetic circular does not mention history of backlogs.",
)

DOCS["orbit_analytics_JD_SYNTHETIC.pdf"] = company_jd(
    "Orbit Analytics (synthetic)",
    "Data Analyst",
    [
        ("Job role", "Data Analyst"),
        ("Employment type", "Full-time"),
        ("CTC", "7.5 LPA"),
        ("Work location", "Hyderabad"),
        ("Bond period", "No bond"),
    ],
    "Orbit Analytics is a fictional analytics consultancy created for PlaceBot demonstrations.",
    ["Aptitude test", "Case study round", "Technical and HR interview"],
    "The fixed and variable split of the CTC is not specified in this synthetic document.",
)
DOCS["orbit_analytics_eligibility_SYNTHETIC.pdf"] = eligibility(
    "Orbit Analytics (synthetic)",
    "1.0",
    "2025-09-05",
    [
        ("Eligible branches", "CSE, ISE, AIML, ECE, EEE", "All listed branches are eligible"),
        ("Minimum CGPA", "6.5", "On a 10 point scale, 6.5 qualifies"),
        ("Active backlogs", "1", "Maximum of one active backlog"),
        ("10th percentage", "60", "Minimum"),
        ("12th percentage", "60", "Minimum"),
    ],
    "History of backlogs is not restricted.",
)

DOCS["quanta_grid_JD_SYNTHETIC.pdf"] = company_jd(
    "Quanta Grid Systems (synthetic)",
    "Embedded Systems Engineer",
    [
        ("Job role", "Embedded Systems Engineer"),
        ("Employment type", "6 month internship leading to full-time"),
        ("CTC", "6.8 LPA"),
        ("Work location", "Pune"),
        ("Bond period", "24 months"),
    ],
    "Quanta Grid Systems is a fictional power electronics company created for PlaceBot demonstrations.",
    ["Written test", "Technical interview", "HR interview"],
    "Selected students begin with a six month paid internship.",
)
DOCS["quanta_grid_eligibility_SYNTHETIC.pdf"] = eligibility(
    "Quanta Grid Systems (synthetic)",
    "1.0",
    "2025-09-10",
    [
        ("Eligible branches", "ECE, EEE, EIE", "Other branches are not eligible"),
        ("Minimum CGPA", "6.0", "On a 10 point scale, 6.0 qualifies"),
        ("Active backlogs", "0", "No active backlogs allowed"),
        ("10th percentage", "60", "Minimum"),
        ("12th percentage", "60", "Minimum"),
    ],
    "Original circular.",
)
# Deliberately conflicting newer circular, used to test conflict handling.
DOCS["quanta_grid_eligibility_revised_SYNTHETIC.pdf"] = eligibility(
    "Quanta Grid Systems (synthetic)",
    "2.0",
    "2025-10-15",
    [
        ("Eligible branches", "ECE, EEE, EIE", "Unchanged"),
        ("Minimum CGPA", "6.5", "Revised upward from the earlier circular"),
        ("Active backlogs", "0", "No active backlogs allowed"),
        ("10th percentage", "60", "Minimum"),
        ("12th percentage", "60", "Minimum"),
    ],
    "Revised circular. This synthetic revision exists to test conflicting-criteria handling.",
)

DOCS["placement_faq_SYNTHETIC.pdf"] = (
    header("Placement FAQ", "N/A", "2025-08-15", "1.0")
    + [
        "1. General Questions",
        "Q: Who can register for placements?",
        "A: Final year students who complete the registration form on the placement portal can register.",
        "Q: Can I attend a drive without registering?",
        "A: No. Registration before the deadline is required to attend any drive.",
        "",
        "2. Offers",
        "Q: Can I attend another drive after receiving an offer?",
        "A: It depends on the category of the offer you accepted. See clause 3 of the placement policy.",
        "Q: Will the placement cell publish the offer letter?",
        "A: The company issues the offer letter. The placement cell only shares the acknowledgement form.",
        "",
        "3. Documents",
        "Q: What documents should I carry to a drive?",
        "A: A valid college ID card, a printed resume and marks cards.",
    ]
)

DOCS["recruitment_process_SYNTHETIC.pdf"] = (
    header("Recruitment Process", "N/A", "2025-08-20", "1.0")
    + [
        "1. Typical Stages",
        "Stage | Description",
        "Pre-placement talk | The company presents its roles and expectations",
        "Online test | Aptitude or coding assessment held on campus",
        "Interviews | One or more technical interviews",
        "HR round | Discussion of the offer and joining details",
        "",
        "2. Communication",
        "Shortlists are announced on the placement portal. Students must check the portal for venue and timing updates.",
    ]
)


def render(path: Path, lines: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    margin, top, leading = 50, h - 60, 14
    y = top
    for raw in lines:
        if raw == PAGEBREAK:
            c.showPage()
            y = top
            continue
        is_heading = bool(raw[:2].strip(". ").isdigit()) and raw[1:2] in ".)" and len(raw) < 60 and "|" not in raw
        wrapped = [raw] if "|" in raw or not raw else textwrap.wrap(raw, 100)
        for seg in wrapped or [""]:
            if y < 60:
                c.showPage()
                y = top
            c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 11 if is_heading else 9.5)
            c.drawString(margin, y, seg)
            y -= leading
    c.showPage()
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, lines in DOCS.items():
        render(OUT_DIR / name, lines)
        print("wrote", OUT_DIR / name)


if __name__ == "__main__":
    main()
