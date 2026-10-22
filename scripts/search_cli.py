"""Try hybrid retrieval from the terminal.

Usage:
    python -m scripts.search_cli "What is the minimum CGPA for Nimbus Soft?"
    python -m scripts.search_cli "bond period" --company "Quanta Grid Systems (synthetic)" --top-k 3
"""
from __future__ import annotations

import argparse

from config.settings import get_settings
from src.ingestion.pipeline import load_corpus
from src.retrieval.hybrid_search import HybridRetriever


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query")
    ap.add_argument("--company", default=None)
    ap.add_argument("--doc-type", action="append", default=None)
    ap.add_argument("--top-k", type=int, default=s.top_k)
    ap.add_argument("--corpus", default=str(s.processed_dir))
    args = ap.parse_args()

    _, chunks = load_corpus(args.corpus)
    r = HybridRetriever(chunks, s)
    res = r.search(args.query, company=args.company, doc_types=args.doc_type, top_k=args.top_k)

    print(f"embedder={res.embedder}  bm25={r.bm25.backend}  vectors={r.vectors.backend}")
    print(f"company filter: {res.filters['company']}")
    status = "EVIDENCE FOUND" if res.evidence_sufficient else "INFORMATION NOT FOUND"
    print(f"status: {status} ({res.reason})\n")
    for i, x in enumerate(res.results, 1):
        c = x.chunk
        print(f"[{i}] {c.source_filename} | {c.company or 'n/a'} | {c.doc_type} | {c.section} | page {c.page_label}")
        print(f"    rrf={x.rrf_score:.4f} bm25={x.bm25_score} vector={x.vector_score}")
        for n in x.notes:
            print(f"    note: {n}")
        print("    " + c.text.replace("\n", "\n    ") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
