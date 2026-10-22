"""Rule-based query rewriting for follow-ups (deterministic, no LLM required)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from src.retrieval.reranker import detect_companies
from src.retrieval.text_utils import informative_terms

_NUMWORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
_NUM_BEFORE = re.compile(r"\b(zero|one|two|three|four|five)\b(?=\s+(?:active\s+|current\s+)?(?:backlogs?|arrears?)\b)", re.I)
_POLICY_WORDS = re.compile(r"\b(offer|debar\w*|withdraw\w*|polic\w*|clause|rule|rules|attendance|dream|regular)\b", re.I)
_FOLLOW_UP = re.compile(r"\b(it|its|that|this|they|their|there|same|those)\b|^(and|also|what about|how about)\b", re.I)


@dataclass
class Rewrite:
    query: str
    company: Optional[str]
    note: str = ""


def normalize_numbers(q: str) -> str:
    """'one active backlog' -> '1 active backlog' so it matches table values."""
    return _NUM_BEFORE.sub(lambda m: _NUMWORDS[m.group(1).lower()], q)


def rewrite_query(question: str, history: Sequence[tuple[str, str]], companies: Sequence[str]) -> Rewrite:
    """Resolve follow-ups using the most recent user turns that named a company or topic."""
    q = normalize_numbers(question.strip())
    notes: list[str] = []
    if q != question.strip():
        notes.append("number words converted to digits")
    named = detect_companies(q, companies)
    if len(named) == 1:
        return Rewrite(q, named[0], "; ".join(notes))
    if named:  # several companies named: leave to cross-company handling
        return Rewrite(q, None, "; ".join(notes))
    user_turns = [t for r, t in history if r == "user"]
    prev_company: Optional[str] = None
    prev_q: Optional[str] = None
    for t in reversed(user_turns):
        found = detect_companies(t, companies)
        if len(found) == 1:
            prev_company, prev_q = found[0], t
            break
    if prev_company and not _POLICY_WORDS.search(q) and (_FOLLOW_UP.search(q) or len(informative_terms(q)) <= 3):
        notes.append(f"follow-up resolved to {prev_company}")
        q = f"{q} ({re.sub(r'\s*\(.*?\)', '', prev_company).strip()})"
        return Rewrite(q, prev_company, "; ".join(notes))
    return Rewrite(q, None, "; ".join(notes))
