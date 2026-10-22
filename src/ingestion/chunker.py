"""Section-aware chunking with page tracking and table-header preservation.

Strategy
--------
1. Walk all cleaned pages as a stream of (page, line) pairs.
2. Start a new *section* whenever a heading line is detected.
3. Within a section, group consecutive pipe-delimited lines into *table blocks* and
   everything else into *text blocks*.
4. Pack blocks into chunks up to ``max_tokens`` (aiming for ``target_tokens``). Chunk
   boundaries only fall between lines. Long tables are split by rows and the header row is
   repeated in every piece so numbers never lose their column names.
5. Text chunks carry ``overlap_tokens`` worth of trailing lines from the previous chunk of
   the same section.

Tokens are approximated as ``words * 1.3`` (no tokenizer dependency).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.schemas import Chunk, PageText

_HEADING_NUM = re.compile(r"^(\d+(\.\d+)*[.)]?|[A-Z][.)]|Section\s+\d+[.:]?|Clause\s+\d+(\.\d+)*[.:]?)\s+\S")
_METADATA_LINE = re.compile(r"^(document type|company|effective date|version|role|status|source)\s*:", re.I)


def count_tokens(text: str) -> int:
    """Cheap token estimate: ~1.3 tokens per whitespace-delimited word."""
    return int(len(text.split()) * 1.3) + (1 if text.strip() else 0)


def is_table_line(line: str) -> bool:
    """A pipe-delimited row (two or more columns means at least one separator)."""
    return line.count("|") >= 1


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90 or is_table_line(s):
        return False
    if _METADATA_LINE.match(s):
        return False
    if s.endswith((".", ",", ";")) and not _HEADING_NUM.match(s):
        return False
    if _HEADING_NUM.match(s) and len(s.split()) <= 12 and not s.endswith((".", ";", ",")):
        return True
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters) and len(s.split()) <= 10:
        return True
    return False


@dataclass
class _Line:
    page: int
    text: str


@dataclass
class _Section:
    heading: str
    lines: list[_Line]


def _split_sections(pages: list[PageText]) -> list[_Section]:
    sections: list[_Section] = [_Section(heading="Document header", lines=[])]
    for pg in pages:
        for raw in pg.text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if is_heading(line):
                sections.append(_Section(heading=line, lines=[]))
            else:
                sections[-1].lines.append(_Line(pg.page, line))
    return [s for s in sections if s.lines]


@dataclass
class _Block:
    kind: str  # "text" | "table"
    lines: list[_Line]


def _to_blocks(lines: list[_Line]) -> list[_Block]:
    blocks: list[_Block] = []
    for ln in lines:
        kind = "table" if is_table_line(ln.text) else "text"
        if blocks and blocks[-1].kind == kind:
            blocks[-1].lines.append(ln)
        else:
            blocks.append(_Block(kind, [ln]))
    return blocks


def _split_table(block: _Block, max_tokens: int, heading_tokens: int) -> list[list[_Line]]:
    """Split a table into row groups, repeating the header row (first line) in each."""
    header, rows = block.lines[0], block.lines[1:]
    if not rows:
        return [block.lines]
    budget = max(max_tokens - heading_tokens, 60)
    groups: list[list[_Line]] = []
    current: list[_Line] = [header]
    used = count_tokens(header.text)
    for row in rows:
        t = count_tokens(row.text)
        if len(current) > 1 and used + t > budget:
            groups.append(current)
            current, used = [header], count_tokens(header.text)
        current.append(row)
        used += t
    groups.append(current)
    return groups


def _make_chunk(
    lines: list[_Line],
    heading: str,
    doc_meta: dict,
    seq: int,
) -> Chunk:
    text = "\n".join(l.text for l in lines)
    pages = [l.page for l in lines]
    return Chunk(
        chunk_id=f"{doc_meta['doc_id']}::c{seq:03d}",
        doc_id=doc_meta["doc_id"],
        source_filename=doc_meta["filename"],
        doc_type=doc_meta.get("doc_type", "unknown"),
        company=doc_meta.get("company"),
        section=heading,
        page_start=min(pages),
        page_end=max(pages),
        text=text,
        effective_date=doc_meta.get("effective_date"),
        version=doc_meta.get("version"),
        synthetic=bool(doc_meta.get("synthetic", False)),
        token_count=count_tokens(text),
    )


def _tail_overlap(lines: list[_Line], overlap_tokens: int) -> list[_Line]:
    """Trailing text lines (never table rows) worth ~overlap_tokens, for the next chunk."""
    if overlap_tokens <= 0:
        return []
    picked: list[_Line] = []
    used = 0
    for ln in reversed(lines):
        if is_table_line(ln.text):
            break
        t = count_tokens(ln.text)
        if used + t > overlap_tokens:
            break
        picked.append(ln)
        used += t
    return list(reversed(picked))


def chunk_document(
    pages: list[PageText],
    doc_meta: dict,
    target_tokens: int = 300,
    max_tokens: int = 400,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Chunk a cleaned document into section-aware, metadata-rich chunks.

    Args:
        pages: cleaned pages (see :func:`src.ingestion.cleaner.clean_pages`).
        doc_meta: must include ``doc_id`` and ``filename``; may include ``doc_type``,
            ``company``, ``effective_date``, ``version``, ``synthetic``.
    """
    if max_tokens < target_tokens:
        max_tokens = target_tokens
    chunks: list[Chunk] = []
    seq = 0
    for section in _split_sections(pages):
        heading = section.heading
        heading_tokens = count_tokens(heading)
        pending: list[_Line] = []
        pending_tokens = heading_tokens
        carried: list[_Line] = []  # overlap lines already inside `pending`

        def flush() -> None:
            nonlocal pending, pending_tokens, seq, carried
            if not pending or all(l in carried for l in pending):
                pending, pending_tokens, carried = [], heading_tokens, []
                return
            seq += 1
            chunks.append(_make_chunk(pending, heading, doc_meta, seq))
            carried = _tail_overlap(pending, overlap_tokens)
            pending = list(carried)
            pending_tokens = heading_tokens + sum(count_tokens(l.text) for l in pending)

        for block in _to_blocks(section.lines):
            if block.kind == "table":
                groups = (
                    _split_table(block, max_tokens, heading_tokens)
                    if sum(count_tokens(l.text) for l in block.lines) + heading_tokens > max_tokens
                    else [block.lines]
                )
                for g in groups:
                    g_tokens = sum(count_tokens(l.text) for l in g)
                    if pending and pending_tokens + g_tokens > max_tokens:
                        flush()
                        # never carry text overlap into the front of a table chunk
                        pending, pending_tokens, carried = [], heading_tokens, []
                    pending.extend(g)
                    pending_tokens += g_tokens
                    if len(groups) > 1:  # each split table piece stands alone
                        flush()
                        pending, pending_tokens, carried = [], heading_tokens, []
            else:
                for ln in block.lines:
                    t = count_tokens(ln.text)
                    if pending and pending_tokens + t > max_tokens:
                        flush()
                    elif pending and pending_tokens >= target_tokens and pending_tokens + t > target_tokens:
                        flush()
                    pending.append(ln)
                    pending_tokens += t
        flush()
    return chunks
