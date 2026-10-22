"""Text cleaning that preserves structure (headings, table rows, numbers)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from src.schemas import PageText

_WS = re.compile(r"[ \t ]+")
_PAGE_NUM = re.compile(r"^\s*(page\s+)?\d+(\s*(/|of)\s*\d+)?\s*$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Unicode-normalise, fix hyphenated line breaks, collapse spaces, keep newlines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated words
    lines = [_WS.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines)


def _detect_repeated_lines(pages: list[PageText], min_pages: int = 3) -> set[str]:
    """Lines repeated on most pages are headers/footers. Only applied to >= min_pages docs."""
    if len(pages) < min_pages:
        return set()
    counter: Counter[str] = Counter()
    for pg in pages:
        seen = {ln.strip() for ln in pg.text.split("\n") if ln.strip()}
        counter.update(seen)
    threshold = max(2, int(0.6 * len(pages)))
    return {ln for ln, c in counter.items() if c >= threshold and len(ln) < 100}


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Return cleaned copies of pages: normalised, headers/footers and page numbers removed."""
    normalised = [PageText(page=p.page, text=normalize_text(p.text)) for p in pages]
    repeated = _detect_repeated_lines(normalised)
    out: list[PageText] = []
    for p in normalised:
        kept = []
        for ln in p.text.split("\n"):
            if ln in repeated or _PAGE_NUM.match(ln):
                continue
            kept.append(ln)
        # collapse 3+ blank lines to a single blank line
        txt = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        out.append(PageText(page=p.page, text=txt))
    return out
