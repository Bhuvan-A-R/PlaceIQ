"""Ingest PDFs and write the processed corpus (chunks.jsonl + registry.json).

Usage:
    python -m scripts.build_index --source data/sample
    python -m scripts.build_index --source data/raw
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from config.settings import get_settings
from src.ingestion.pdf_loader import backend_name
from src.ingestion.pipeline import ingest_directory, save_corpus


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(s.sample_dir), help="Directory containing PDFs")
    ap.add_argument("--out", default=str(s.processed_dir), help="Output directory for the processed corpus")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_dir():
        print(f"Source directory not found: {src}")
        return 1
    records, chunks = ingest_directory(src, s)
    save_corpus(records, chunks, args.out)

    print(f"PDF backend: {backend_name()}")
    print(f"Documents: {len(records)}  Chunks: {len(chunks)}  -> {args.out}")
    by_status = Counter(r.status for r in records)
    print("Status:", dict(by_status))
    for r in records:
        flag = "" if r.status == "indexed" else f"  <-- {r.status}: {r.error}"
        print(f"  {r.filename:<48} {r.doc_type:<22} chunks={r.num_chunks:<3}{flag}")
    return 0 if all(r.status in ("indexed", "duplicate") for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
