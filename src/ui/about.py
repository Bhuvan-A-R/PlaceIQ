"""About and Disclaimer."""
from __future__ import annotations

import streamlit as st

from src.ui.common import hero


def render() -> None:
    hero("About and Disclaimer", "What PlaceBot is, and what it is not.")
    st.markdown("""
**PlaceBot** answers campus-placement questions from indexed placement documents using retrieval-augmented
generation, and checks eligibility with deterministic Python rules.

### Disclaimer
* PlaceBot is a decision aid. **Its verdicts are not official institutional decisions.** The official placement rules
  and the placement cell always take precedence.
* Answers and verdicts depend on the documents that were indexed. If a circular is missing, outdated or conflicting,
  PlaceBot says so and returns *Information not found* or *UNCLEAR* rather than guessing.
* PlaceBot never silently substitutes an older circular for a newer one; conflicting versions are shown together.

### Synthetic data
Documents named `*_SYNTHETIC.pdf` in `data/sample` are invented examples. Their companies, CGPA cut-offs, packages,
bonds and policies are **not real** and must not be used for real decisions.

### Privacy
No names, USNs or phone numbers are requested. Student inputs stay in your browser session. The provenance log stores
only a short hash of each question, the status and the source documents used. API keys are read from environment variables.

### Known limitations
* Table extraction depends on how a PDF encodes tables; real documents may need parser adjustments.
* Without an LLM key, answers are extractive demo output, not generated text.
* Scanned PDFs need OCR before they can be indexed.
* Eligibility covers CGPA, branch, active and past backlogs, 10th/12th percentages and academic gap. Other
  company-specific conditions must be checked in the source circular.
""")
