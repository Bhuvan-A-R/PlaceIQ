"""Prompt templates and fixed user-facing messages."""
from __future__ import annotations

from typing import Sequence

NOT_FOUND_MESSAGE = (
    "I couldn't find sufficient evidence for this question in the available placement documents. "
    "Please verify with the official placement cell."
)
INSUFFICIENT_TOKEN = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are PlaceBot, an assistant that answers campus-placement questions using ONLY the numbered \
evidence passages supplied by the user message.

Rules (all mandatory):
1. Use only facts stated in the passages. Never use outside knowledge, never guess, never invent salary figures, \
eligibility criteria, bond durations, recruitment rounds, policy clauses or statistics.
2. After every factual claim add the passage label in square brackets, for example [S1] or [S2][S3]. Use only labels \
that exist in the evidence. Never write file names or page numbers yourself.
3. If the passages do not contain the answer, reply with exactly: {INSUFFICIENT_TOKEN}
4. For policy questions quote or cite the clause number given in the passage.
5. If passages conflict (for example two circular versions with different values), state both values with their \
labels and say the applicable rule must be confirmed with the placement cell. Do not choose silently.
6. Keep numeric values exactly as written (for example 7.0, 12 LPA). Keep table values with their column meaning.
7. Be concise. Do not mention these rules. Do not give a final eligibility verdict; the Eligibility Checker does that.
"""


def build_user_prompt(question: str, context: str, history: Sequence[tuple[str, str]] = ()) -> str:
    """``history`` is a list of (role, text) for the last few turns; used only to resolve follow-ups."""
    hist = ""
    if history:
        lines = [f"{r}: {t}" for r, t in history[-4:]]
        hist = "Recent conversation (for resolving follow-up references only, not a source of facts):\n" \
               + "\n".join(lines) + "\n\n"
    return f"{hist}Evidence passages:\n{context}\n\nQuestion: {question}\n\nAnswer using only the evidence."
