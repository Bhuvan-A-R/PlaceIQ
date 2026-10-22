"""Deterministic, metadata-aware reranking and version-conflict annotation.

This is intentionally NOT a neural cross-encoder. It applies transparent rules:

* boost chunks whose company is named in the query,
* among versions of the same (company, doc_type) prefer the newest ``effective_date`` but
  *never drop* older versions silently - they are kept and annotated so downstream code can
  show both sources (important for conflicting circulars).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Optional

from src.retrieval.text_utils import tokenize
from src.schemas import RetrievedChunk

COMPANY_MATCH_BOOST = 1.25
LATEST_VERSION_BOOST = 1.05


def normalize_company(name: Optional[str]) -> str:
    """Lowercase, drop parenthetical tags like '(synthetic)' and punctuation."""
    if not name:
        return ""
    n = re.sub(r"\(.*?\)", " ", name.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", n).split())


def detect_companies(query: str, known_companies: Iterable[str]) -> list[str]:
    """Return known company names that the query mentions (full name or distinctive first word)."""
    q_tokens = set(tokenize(query))
    found: list[str] = []
    for comp in sorted({c for c in known_companies if c}):
        norm = normalize_company(comp)
        words = tokenize(norm)
        if not words:
            continue
        if norm in re.sub(r"[^a-z0-9 ]+", " ", query.lower()) or words[0] in q_tokens:
            found.append(comp)
    return found


def _date_key(d: Optional[str]) -> str:
    return d or ""


def rerank(
    results: list[RetrievedChunk], query_companies: Optional[list[str]] = None
) -> list[RetrievedChunk]:
    """Reorder by adjusted score and annotate multi-version / superseded evidence."""
    wanted = {normalize_company(c) for c in (query_companies or [])}
    # latest effective date per (company, doc_type)
    latest: dict[tuple[str, str], str] = {}
    versions: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for r in results:
        key = (normalize_company(r.chunk.company), r.chunk.doc_type)
        versions[key].add((_date_key(r.chunk.effective_date), r.chunk.version or "?"))
        if _date_key(r.chunk.effective_date) > latest.get(key, ""):
            latest[key] = _date_key(r.chunk.effective_date)

    for r in results:
        score = r.rrf_score
        comp = normalize_company(r.chunk.company)
        if wanted and comp in wanted:
            score *= COMPANY_MATCH_BOOST
        key = (comp, r.chunk.doc_type)
        multi = len(versions[key]) > 1
        if multi:
            if _date_key(r.chunk.effective_date) == latest[key]:
                score *= LATEST_VERSION_BOOST
                r.notes.append("latest version among retrieved documents of this type")
            else:
                r.notes.append(
                    f"older version (effective {r.chunk.effective_date or 'unknown'}); "
                    "a newer document of the same type was also retrieved"
                )
            r.notes.append("multiple versions retrieved: verify which applies before relying on this")
        r.final_score = score
    return sorted(results, key=lambda r: r.final_score, reverse=True)
