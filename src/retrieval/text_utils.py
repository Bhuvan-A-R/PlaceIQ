"""Shared tokenisation helpers for retrieval (BM25 and term-coverage checks)."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z][a-z0-9]*")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our should so than that the their them then there these they this to was we were what when
    where which who whom why will with would you your us am any all some may might must shall also about after
    again am being both each few more most other such only own same too very just get got""".split()
)


def stem(token: str) -> str:
    """Very light stemmer: plural/-ing trimming so 'backlogs' matches 'backlog'."""
    if token.replace(".", "").isdigit():
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Lowercase, keep decimals (6.5) as single tokens, stem lightly."""
    toks = _TOKEN.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


def informative_terms(query: str) -> list[str]:
    """Distinct non-stopword stems from the query, order preserved."""
    seen: dict[str, None] = {}
    for t in tokenize(query, remove_stopwords=True):
        if len(t) > 1 or t.isdigit():
            seen.setdefault(t, None)
    return list(seen)


def index_text(company: str | None, section: str, text: str) -> str:
    """Text actually indexed for a chunk: company + section heading + body."""
    return f"{company or ''}\n{section}\n{text}"
