"""Configurable LLM access. Provider and key come from environment variables only.

Providers: ``mock`` (default, never calls a model), ``anthropic``, ``openai`` (also any
OpenAI-compatible server via PLACEBOT_LLM_BASE_URL) and ``ollama`` (local).

Mock mode is a transparent *extractive* demo: it selects the most relevant lines from the
retrieved passages. It is always labelled as such and is never presented as a model response.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from config.settings import Settings, get_settings
from src.generation.prompts import INSUFFICIENT_TOKEN
from src.retrieval.text_utils import informative_terms, tokenize
from src.schemas import RetrievedChunk


class LLMConfigError(RuntimeError):
    """Raised when a provider is selected but not configured."""


class LLMClient(Protocol):
    name: str
    is_mock: bool

    def generate(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    is_mock = False

    def __init__(self, s: Settings) -> None:
        if not s.llm_api_key:
            raise LLMConfigError("PLACEBOT_LLM_API_KEY is not set.")
        if not s.llm_model:
            raise LLMConfigError("PLACEBOT_LLM_MODEL is not set (choose a model from your Anthropic account).")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError("Install the 'anthropic' package: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=s.llm_api_key, timeout=s.llm_timeout_s)
        self._s = s
        self.name = f"anthropic:{s.llm_model}"

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - network
        msg = self._client.messages.create(model=self._s.llm_model, max_tokens=self._s.llm_max_tokens, temperature=0,
                                           system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


class OpenAILLM:
    is_mock = False

    def __init__(self, s: Settings) -> None:
        if not s.llm_api_key:
            raise LLMConfigError("PLACEBOT_LLM_API_KEY is not set.")
        if not s.llm_model:
            raise LLMConfigError("PLACEBOT_LLM_MODEL is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError("Install the 'openai' package: pip install openai") from exc
        kwargs = {"api_key": s.llm_api_key, "timeout": s.llm_timeout_s}
        if s.llm_base_url:
            kwargs["base_url"] = s.llm_base_url
        self._client = OpenAI(**kwargs)
        self._s = s
        self.name = f"openai:{s.llm_model}"

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - network
        r = self._client.chat.completions.create(
            model=self._s.llm_model, temperature=0, max_tokens=self._s.llm_max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()


class OllamaLLM:
    is_mock = False

    def __init__(self, s: Settings) -> None:
        if not s.llm_model:
            raise LLMConfigError("PLACEBOT_LLM_MODEL is not set (e.g. a model you pulled with 'ollama pull').")
        self._s = s
        self._url = (s.llm_base_url or "http://localhost:11434").rstrip("/") + "/api/chat"
        self.name = f"ollama:{s.llm_model}"

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - network
        body = json.dumps({"model": self._s.llm_model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self._url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._s.llm_timeout_s) as resp:
            return json.loads(resp.read())["message"]["content"].strip()


class MockLLM:
    """Placeholder client: ``generate`` is never used for answers (see :func:`extractive_answer`)."""

    name = "mock (extractive demo, no LLM)"
    is_mock = True

    def generate(self, system: str, user: str) -> str:  # pragma: no cover
        raise RuntimeError("MockLLM does not generate text; use extractive_answer().")


def get_llm_client(settings: Optional[Settings] = None) -> tuple[LLMClient, Optional[str]]:
    """Return (client, fallback_reason). On misconfiguration fall back to mock and say why."""
    s = settings or get_settings()
    provider = s.llm_provider
    if provider in ("", "mock"):
        return MockLLM(), None
    try:
        if provider == "anthropic":
            return AnthropicLLM(s), None
        if provider == "openai":
            return OpenAILLM(s), None
        if provider == "ollama":
            return OllamaLLM(s), None
        return MockLLM(), f"Unknown provider '{provider}'; using mock mode."
    except LLMConfigError as exc:
        return MockLLM(), f"{provider} is not configured ({exc}) Using mock mode."


# ---------------------------------------------------------------- extractive demo
_HEADER_WORDS = {"criterion", "criteria", "item", "category", "stage", "round", "parameter", "field", "particulars",
                 "attribute", "requirement", "description", "details", "detail"}
_NEW_ITEM = re.compile(r"^(\d+(?:\.\d+)*[.)]?\s|[-*\u2022]\s|Q:|A:)")


def _row_phrase(header: Optional[list[str]], cells: list[str]) -> str:
    """'Minimum CGPA | 7.0 | note' + header -> 'Minimum CGPA - Requirement: 7.0; Notes: note'."""
    if header and len(header) == len(cells) and len(cells) >= 2:
        rest = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], cells[1:]) if v)
        return f"{cells[0]} - {rest}" if rest else cells[0]
    parts = [c for c in cells if c]
    return f"{parts[0]}: {'; '.join(parts[1:])}" if len(parts) > 1 else " ".join(parts)


def _is_header(cells: Optional[list[str]]) -> bool:
    """A header row uses known header words and is not a numbered data row such as 'Round 1'."""
    if not cells or re.match(r"^(round|stage)\s*\d", cells[0], re.I):
        return False
    return cells[0].strip().lower() in _HEADER_WORDS


def _merge_wrapped(lines: list[str]) -> list[str]:
    """Join PDF-wrapped continuation lines so a clause is one unit."""
    out: list[str] = []
    for l in lines:
        if out and out[-1].startswith("Q:") and l.startswith("A:"):  # keep FAQ question and answer together
            out[-1] = f"{out[-1]} {l}"
        elif out and "|" not in l and "|" not in out[-1] and not _NEW_ITEM.match(l) and \
                not re.search(r"[.:?!]$", out[-1]):
            out[-1] = f"{out[-1]} {l}"
        else:
            out.append(l)
    return out


def extractive_lines(query: str, results: Sequence[RetrievedChunk], max_chunks: int = 3,
                     exclude_terms: Sequence[str] = ()) -> list[tuple[int, str]]:
    """Pick the most query-relevant lines from the top chunks. Returns (result_index, phrase).

    A chunk is used only if its relevance is at least 60% of the best chunk's, so weakly related
    passages do not add noise. Table rows are rendered together with their column headers.
    """
    exclude = set(exclude_terms) | {"synthetic"}
    terms = [t for t in informative_terms(query) if t not in exclude]
    word_terms = [t for t in terms if not t.replace(".", "").isdigit()]
    cands: list[tuple[float, int, list[tuple[str, Optional[list[str]]]], Optional[list[str]]]] = []
    for idx, rc in enumerate(results[:max_chunks]):
        lines = _merge_wrapped([l.strip() for l in rc.chunk.text.splitlines() if l.strip()])
        body = [(l, [c.strip() for c in l.split("|")] if "|" in l else None) for l in lines]
        hdr = next((cells for _, cells in body if cells), None)
        hdr = hdr if _is_header(hdr) else None
        scored = [(sum(1 for t in terms if t in set(tokenize(l))), l, cells) for l, cells in body
                  if not (cells is not None and cells is hdr)]
        best = max((s for s, _, _ in scored), default=0)
        sec_hits = sum(1 for t in word_terms if t in set(tokenize(rc.chunk.section)))
        if best > 0:
            keep = [(l, c) for s, l, c in scored if s >= max(1, best * 0.5)][:4]
        elif sec_hits:
            keep = [(l, c) for _, l, c in scored][:8]
        else:
            continue
        cands.append((best + 2 * sec_hits, idx, keep, hdr))
    if not cands:
        return []
    top = max(c[0] for c in cands)
    picked: list[tuple[int, str]] = []
    for score, idx, keep, hdr in cands:
        if score >= 0.6 * top:
            picked += [(idx, _row_phrase(hdr, cells) if cells else l) for l, cells in keep]
    return picked


def extractive_answer(query: str, results: Sequence[RetrievedChunk], labels: dict[str, int],
                      exclude_terms: Sequence[str] = ()) -> str:
    """Demo-mode answer: relevant lines + their [S#] labels. ``labels`` maps chunk_id -> label number."""
    lines = extractive_lines(query, results, exclude_terms=exclude_terms)
    out = []
    for idx, phrase in lines:
        n = labels.get(results[idx].chunk.chunk_id)
        if n is None:
            continue
        out.append(f"- {phrase} [S{n}]")
    return "\n".join(out)
