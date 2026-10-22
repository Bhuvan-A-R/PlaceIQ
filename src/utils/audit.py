"""Minimal provenance log: which source versions supplied an answer.

Privacy: no student profile values and no raw question text are written; only a short hash of
the question, the status, and the source documents used.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from config.settings import Settings, get_settings
from src.schemas import Chunk


def log_answer(kind: str, query: str, status: str, mode: str, chunks: Iterable[Chunk],
               settings: Optional[Settings] = None, extra: Optional[dict] = None) -> None:
    s = settings or get_settings()
    if not s.log_answers:
        return
    try:
        s.processed_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind, "query_sha": hashlib.sha256(query.encode()).hexdigest()[:12],
            "status": status, "mode": mode,
            "sources": [{"doc_id": c.doc_id, "filename": c.source_filename, "chunk_id": c.chunk_id,
                         "version": c.version, "effective_date": c.effective_date} for c in chunks],
            **(extra or {}),
        }
        with open(Path(s.processed_dir) / "answer_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:  # logging must never break the app
        pass


def read_log(settings: Optional[Settings] = None, limit: int = 200) -> list[dict]:
    s = settings or get_settings()
    p = Path(s.processed_dir) / "answer_log.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out
