"""Context assembly: dedupe, budget, and label retrieved chunks for generation."""
from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.chunker import count_tokens
from src.schemas import RetrievedChunk


@dataclass
class ContextBundle:
    """Chunks selected for the prompt, each addressable by a stable label like ``S1``."""

    items: list[tuple[str, RetrievedChunk]]
    text: str
    tokens_used: int
    dropped_for_budget: int

    def label_to_chunk_id(self) -> dict[str, str]:
        return {label: rc.chunk.chunk_id for label, rc in self.items}


def assemble_context(results: list[RetrievedChunk], token_budget: int = 1800) -> ContextBundle:
    """Pick chunks in rank order, skipping duplicates, until the token budget is reached.

    Each block carries its source filename, company, type, section and page so that table
    values stay next to their headers and citations can be validated later.
    """
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    items: list[tuple[str, RetrievedChunk]] = []
    blocks: list[str] = []
    used = 0
    dropped = 0
    for rc in results:
        c = rc.chunk
        norm = " ".join(c.text.split()).lower()
        if c.chunk_id in seen_ids or norm in seen_text:
            continue
        label = f"S{len(items) + 1}"
        header = (
            f"[{label}] file={c.source_filename} | company={c.company or 'n/a'} | type={c.doc_type} | "
            f"section={c.section} | page={c.page_label} | effective={c.effective_date or 'unknown'}"
        )
        block = f"{header}\n{c.text}"
        t = count_tokens(block)
        if items and used + t > token_budget:
            dropped += 1
            continue
        seen_ids.add(c.chunk_id)
        seen_text.add(norm)
        items.append((label, rc))
        blocks.append(block)
        used += t
    return ContextBundle(items=items, text="\n\n".join(blocks), tokens_used=used, dropped_for_budget=dropped)
