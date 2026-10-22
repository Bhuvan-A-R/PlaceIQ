"""Evaluation harness: retrieval hit, answer correctness, citation correctness, unsupported claims.

Usage:
    python -m src.evaluation.evaluate                       # writes data/processed/eval_results.csv
    python -m src.evaluation.evaluate --sweep-threshold     # not-found accuracy vs coverage threshold
    python -m src.evaluation.evaluate --embedder hashing    # force the offline embedder

All numbers are computed from the run; nothing is hardcoded. In mock mode, "answer correctness"
measures the extractive demo output, not a language model.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import Settings, get_settings
from src.generation.chatbot import PlaceBot
from src.ingestion.pipeline import load_corpus
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.vector_search import HashingEmbedder

DEFAULT_QUESTIONS = Path(__file__).with_name("test_questions.csv")
COLUMNS = ["id", "category", "query", "answerable", "expected_source", "expected_evidence", "status",
           "retrieved_top_k", "retrieval_hit", "answer_correct", "citation_correct", "unsupported_claims",
           "mode", "notes"]


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(";") if x.strip()]


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def evaluate_one(bot: PlaceBot, q: dict) -> dict:
    answerable = q["answerable"].strip().lower() == "yes"
    ans = bot.ask(q["query"], company=q.get("company") or None)
    retrieved = ans.retrieval.results if ans.retrieval else []
    top = [f"{r.chunk.source_filename}:p{r.chunk.page_label}" for r in retrieved]
    notes: list[str] = []
    exp_sources, exp_evidence = _split(q["expected_source"]), q["expected_evidence"].strip()
    answer_sources = {c.filename for c in ans.citations}

    if answerable:
        files = {r.chunk.source_filename for r in retrieved} | answer_sources  # cross-company answers use criteria chunks
        in_text = any(_norm(exp_evidence) in _norm(r.chunk.text) for r in retrieved) or \
            any(_norm(exp_evidence) in _norm(c.passage) for c in ans.citations)
        hit = all(s in files for s in exp_sources) and (in_text or not exp_evidence)
        if not hit:
            notes.append("expected source/evidence not in retrieved or cited passages")
        needles = _split(q["expected_answer_contains"])
        correct = ans.found and all(_norm(n) in _norm(ans.text) for n in needles)
        if not ans.found:
            notes.append("answerable question returned not-found (threshold too strict or retrieval miss)")
        elif not correct:
            notes.append("answer text missing expected content")
        cite_ok = bool(ans.citations) and (ans.validation.citations_ok if ans.validation else False) and \
            all(s in answer_sources for s in exp_sources)
        if ans.found and not cite_ok:
            notes.append("citations missing, invalid, or not covering all expected sources")
    else:
        hit = None  # no expected source exists
        correct = ans.status == "not_found"
        cite_ok = not ans.citations
        if not correct:
            notes.append("unanswerable question was answered (possible fabrication)")
    unsupported = bool(ans.validation and (ans.validation.unsupported_numbers or ans.validation.uncited_numeric))
    return {"id": q["id"], "category": q["category"], "query": q["query"], "answerable": q["answerable"],
            "expected_source": q["expected_source"], "expected_evidence": exp_evidence, "status": ans.status,
            "retrieved_top_k": " | ".join(top), "retrieval_hit": hit, "answer_correct": correct,
            "citation_correct": cite_ok, "unsupported_claims": unsupported, "mode": ans.mode,
            "notes": "; ".join(notes)}


def run_evaluation(bot: PlaceBot, questions: Optional[list[dict]] = None) -> pd.DataFrame:
    qs = questions or load_questions()
    return pd.DataFrame([evaluate_one(bot, q) for q in qs], columns=COLUMNS)


def summarize(df: pd.DataFrame) -> dict:
    ans = df[df["answerable"].str.lower() == "yes"]
    un = df[df["answerable"].str.lower() == "no"]
    pct = lambda s: float(s.astype(bool).mean()) if len(s) else float("nan")
    return {
        "questions": len(df), "answerable": len(ans), "unanswerable": len(un),
        "retrieval_hit_rate": pct(ans["retrieval_hit"]), "answer_correct_rate": pct(ans["answer_correct"]),
        "citation_correct_rate": pct(ans["citation_correct"]),
        "unanswerable_handled_rate": pct(un["answer_correct"]),
        "unsupported_claim_count": int(df["unsupported_claims"].astype(bool).sum()),
    }


def sweep_threshold(chunks, base: Settings, embedder, values=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Re-run the evaluation at several min-term-coverage thresholds."""
    rows = []
    for v in values:
        s = dataclasses.replace(base, min_term_coverage=v, log_answers=False)
        bot = PlaceBot(chunks, s, retriever=HybridRetriever(chunks, s, embedder=embedder))
        sm = summarize(run_evaluation(bot))
        rows.append({"min_term_coverage": v, "answerable_found_rate": sm["answer_correct_rate"],
                     "unanswerable_handled_rate": sm["unanswerable_handled_rate"]})
    return pd.DataFrame(rows)


def main() -> int:
    s = dataclasses.replace(get_settings(), log_answers=False)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(s.processed_dir))
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    ap.add_argument("--out", default=str(s.processed_dir / "eval_results.csv"))
    ap.add_argument("--embedder", choices=["auto", "hashing"], default="auto")
    ap.add_argument("--sweep-threshold", action="store_true")
    args = ap.parse_args()

    _, chunks = load_corpus(args.corpus)
    emb = HashingEmbedder() if args.embedder == "hashing" else None
    retr = HybridRetriever(chunks, s, embedder=emb)
    bot = PlaceBot(chunks, s, retriever=retr)
    df = run_evaluation(bot, load_questions(Path(args.questions)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"mode={bot.llm.name}  embedder={retr.embedder.name}")
    for k, v in summarize(df).items():
        print(f"  {k}: {v:.0%}" if isinstance(v, float) and "rate" in k else f"  {k}: {v}")
    print(f"Results written to {args.out}")
    for _, r in df[df["notes"] != ""].iterrows():
        print(f"  [{r['id']}] {r['notes']}")
    if args.sweep_threshold:
        print("\nThreshold sweep (min_term_coverage):")
        print(sweep_threshold(chunks, s, retr.embedder).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
