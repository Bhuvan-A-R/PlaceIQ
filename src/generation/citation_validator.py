"""Citation verification: every citation must point at a retrieved chunk that exists in the corpus.

Labels like ``[S2]`` are resolved only through the label -> chunk_id map built from the context
actually sent to the model. Filenames and page numbers are never taken from model text; they are
always read back from the indexed chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from src.schemas import Chunk

CITE_RE = re.compile(r"\[S(\d+)\]")
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


@dataclass
class ValidationReport:
    cited_labels: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)       # label not in this answer's context
    missing_chunks: list[str] = field(default_factory=list)       # label maps to an id absent from the corpus
    uncited_numeric: list[str] = field(default_factory=list)      # sentences with numbers but no citation
    unsupported_numbers: list[tuple[str, str]] = field(default_factory=list)  # (number, sentence)

    @property
    def citations_ok(self) -> bool:
        return not self.invalid_labels and not self.missing_chunks

    @property
    def claims_ok(self) -> bool:
        return not self.uncited_numeric and not self.unsupported_numbers

    @property
    def ok(self) -> bool:
        return self.citations_ok and self.claims_ok and bool(self.cited_labels)


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        out.extend(p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if p.strip())
    return out


def _numbers(s: str) -> list[str]:
    return _NUM_RE.findall(CITE_RE.sub("", s))


def _norm_num(n: str) -> str:
    return n.rstrip("0").rstrip(".") if "." in n else n


def validate_answer(text: str, label_to_chunk_id: Mapping[str, str], corpus: Mapping[str, Chunk]) -> ValidationReport:
    """Check labels, chunk existence and that numbers in cited sentences appear in the cited chunks."""
    rep = ValidationReport()
    for m in CITE_RE.finditer(text):
        lab = f"S{m.group(1)}"
        if lab not in rep.cited_labels:
            rep.cited_labels.append(lab)
    for lab in rep.cited_labels:
        cid = label_to_chunk_id.get(lab)
        if cid is None:
            rep.invalid_labels.append(lab)
        elif cid not in corpus:
            rep.missing_chunks.append(lab)
    for sent in _sentences(text):
        labs = [f"S{n}" for n in CITE_RE.findall(sent)]
        nums = _numbers(sent)
        if not nums:
            continue
        if not labs:
            rep.uncited_numeric.append(sent)
            continue
        evidence_nums: set[str] = set()
        for lab in labs:
            ch = corpus.get(label_to_chunk_id.get(lab, ""))
            if ch:
                evidence_nums.update(_norm_num(n) for n in _NUM_RE.findall(ch.text))
        for n in nums:
            if _norm_num(n) not in evidence_nums:
                rep.unsupported_numbers.append((n, sent))
    return rep


def strip_invalid_citations(text: str, invalid_labels: list[str]) -> str:
    """Remove citation markers that do not resolve (the claim then counts as uncited)."""
    bad = {l[1:] for l in invalid_labels}
    return CITE_RE.sub(lambda m: "" if m.group(1) in bad else m.group(0), text)


def verify_citation_record(chunk_id: str, filename: str, page_label: str, corpus: Mapping[str, Chunk]) -> bool:
    """True iff the chunk exists and filename/page match the indexed record (guards against fabrication)."""
    ch: Optional[Chunk] = corpus.get(chunk_id)
    return bool(ch and ch.source_filename == filename and ch.page_label == page_label)
