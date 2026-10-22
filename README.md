# PlaceBot — Your Smart Companion for Campus Placements

A RAG-based placement assistant for engineering students: source-grounded answers with citations, a
**deterministic** eligibility checker, company explorer/comparison, a policy assistant, document management
and an evaluation suite, all in a Streamlit app.

## Disclosure: synthetic data

Everything in `data/sample/` is **synthetic** (`*_SYNTHETIC.pdf`). Nimbus Soft, Orbit Analytics and Quanta Grid
Systems are invented; every CGPA, CTC, bond, round and policy clause is made up for demos and tests. **None of it
is an official rule of any institution or company.** `quanta_grid_eligibility_revised_SYNTHETIC.pdf`
deliberately conflicts with the original Quanta Grid circular (CGPA 6.5 vs 6.0) to test conflict handling.
To use real data, put your PDFs in `data/raw/` (see below).

## Features

| Area | What it does |
|---|---|
| Ask PlaceBot | Chat with history, follow-up rewriting, company filter, citation cards, expandable retrieved passages, explicit *Information not found* state |
| Eligibility Checker | Anonymous form → criteria **extracted from indexed documents** → deterministic Python rules → ELIGIBLE / NOT ELIGIBLE / UNCLEAR with per-criterion sources |
| Company Explorer / Comparison | Directory built from documents, filters (branch, CGPA, backlogs, CTC, location, role type), side-by-side table, CSV export. Missing ≠ zero ("Not stated") |
| Placement Policies | Policy/FAQ-only retrieval, clause citations, preset topics |
| Documents and Sources | Upload (type/size/magic-byte validation), duplicate detection, versioned replacement (old file kept, marked superseded), re-index, ingestion errors, counts, history, provenance log |
| Evaluation Dashboard | Runs `src/evaluation`, shows retrieval hit / answer / citation / unsupported-claim metrics, threshold sweep |

## Architecture

```
PDF -> load (PyMuPDF | pypdf) -> clean -> metadata -> section-aware chunks -> data/processed/{chunks.jsonl, registry.json}
query -> rewrite (follow-ups, number words) -> metadata filter -> BM25 + vector -> RRF -> rerank
      -> evidence check -> context assembly (dedupe, token budget, S1..Sn labels)
      -> generation (LLM or extractive demo) -> citation + number verification -> UI
criteria: chunks -> criteria_extractor (table rows, prose fallback) -> rules_engine (pure functions) -> verdict
```

Key design choices:

* **Eligibility never uses an LLM.** Thresholds are inclusive; branches match by exact normalised name (no inferred
  equivalence); active backlogs, past backlogs and academic gap are separate criteria.
  Any confirmed mandatory failure → NOT ELIGIBLE. Else any mandatory criterion missing / ambiguous / conflicting /
  student value not entered → UNCLEAR. Else ELIGIBLE. Core criteria (branch, CGPA, active backlogs, 10th, 12th) must be
  stated in the documents; silence is never treated as "satisfied". Two documents with different values → UNCLEAR with both sources shown.
* **Citations are verified.** Labels (`[S1]`) resolve only through the context actually sent; filename/page/passage are read
  back from the indexed chunk. Numbers in a cited sentence must appear in the cited chunk, otherwise the LLM answer is
  rejected and replaced by extractive lines (with a visible warning).
* **Not-found** requires retrieval signals above thresholds *and* enough of the question's key terms in the top passages.
  Scores are retrieval signals, not probabilities. Tune `PLACEBOT_MIN_TERM_COVERAGE` using `--sweep-threshold`.
* **Mock mode is honest.** Without an LLM, answers are verbatim lines from retrieved passages, labelled
  "mock (extractive demo, no LLM)".
* **Cross-company questions** ("Which companies allow EEE with 6.5 CGPA?") are computed from extracted criteria, per company,
  and say they cover only the indexed companies.

## Install and run

```bash
cd placebot
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # defaults run in mock/demo mode, no API key needed

python -m scripts.build_index --source data/sample       # ingest + chunk -> data/processed/
streamlit run app.py                                     # http://localhost:8501
```

(If you skip `build_index`, the app offers a **Build index from sample data** button.)

Other commands:

```bash
python -m scripts.search_cli "What is the minimum CGPA for Nimbus Soft?"      # retrieval only
pytest -q                                                                      # all tests
python -m src.evaluation.evaluate                                              # writes data/processed/eval_results.csv
python -m src.evaluation.evaluate --sweep-threshold                            # threshold sensitivity
python -m scripts.make_sample_pdfs                                             # regenerate synthetic PDFs
```

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `PLACEBOT_LLM_PROVIDER` | `mock` (default), `anthropic`, `openai`, `ollama` |
| `PLACEBOT_LLM_MODEL`, `PLACEBOT_LLM_API_KEY`, `PLACEBOT_LLM_BASE_URL` | provider settings; never hardcoded |
| `PLACEBOT_EMBEDDING_MODEL` | sentence-transformers model; falls back to a hashing embedder (and says so) |
| `PLACEBOT_TOP_K`, `CANDIDATE_K`, `RRF_K`, `CONTEXT_TOKEN_BUDGET` | retrieval/context sizing |
| `PLACEBOT_MIN_BM25_SCORE`, `MIN_VECTOR_SCORE`, `MIN_TERM_COVERAGE` | evidence thresholds |
| `PLACEBOT_CHUNK_*` | target/max tokens and overlap |
| `PLACEBOT_MAX_UPLOAD_MB`, `PLACEBOT_LOG_ANSWERS` | uploads; provenance log toggle |

To use an LLM: `pip install anthropic` (or `openai`), set the provider, model and key. If the provider is misconfigured,
PlaceBot falls back to mock mode and shows why.

## Adding your own documents

1. Put PDFs in `data/raw/` (or use the *Documents and Sources* page).
2. Start each PDF with header lines the metadata detector reads, for example:
   ```
   Document Type: Company Eligibility Circular      (also: Company Job Description, Placement Policy, Placement FAQ, Recruitment Process)
   Company: <name>            (N/A for general documents)
   Effective Date: 2025-09-10
   Version: 1.0
   ```
   Without headers the type is guessed from the filename and company/date stay unknown.
3. Eligibility/JD tables should have rows like `Minimum CGPA | 7.0 | note` and `Round 1 | description`
   (see `data/sample` for the layout). Extraction reads these rows and a few prose patterns.
4. Re-index: `python -m scripts.build_index --source data/raw`, or *Documents → Re-index* (which also skips superseded files).

## How to verify each acceptance criterion

| Criterion | How to check |
|---|---|
| Sample PDF ingested and indexed | `python -m scripts.build_index --source data/sample` → 10 documents; `tests/test_ingestion.py` |
| Source-grounded answer | *Ask PlaceBot* → "What is the minimum CGPA for Nimbus Soft?" → answer ends with `[S1]` |
| Source card shows real passage + page | Expand `[S1]` under the answer: filename, page, section, passage (`tests/test_citations.py`) |
| Deterministic eligibility verdict | *Eligibility Checker*: AIML, CGPA 7.2, 1 backlog, 80/80, Nimbus Soft → NOT ELIGIBLE (active backlogs); Orbit Analytics → ELIGIBLE |
| Missing/contradictory → UNCLEAR | Quanta Grid, EEE, CGPA 6.2, 0 backlogs, 70/70 → UNCLEAR, both circulars shown; leave a field empty → UNCLEAR |
| Multi-company comparison | *Company Comparison* → select 2–3 companies; "Fixed component" for Orbit shows *Not stated* |
| Unanswerable not fabricated | Ask "What was the average package of last year's CSE batch?" → *Information not found* |
| Evaluation records results | `python -m src.evaluation.evaluate` → `data/processed/eval_results.csv` |
| Runs locally | commands above |

## Tests

`pytest -q` runs ingestion, chunk metadata, BM25/vector/RRF retrieval, eligibility boundaries, missing and conflicting
criteria, citation integrity, unanswerable questions, company comparison, document management and the evaluation harness.

## Known limitations (please read)

* **Verification status.** The code was developed in a sandbox without network access, so PyMuPDF, rank-bm25,
  sentence-transformers, FAISS, Streamlit, pytest and the LLM SDKs could not be installed there. The tests ran with a small
  stand-in runner on the `pypdf` fallback, the built-in BM25 and the hashing embedder; the Streamlit pages were only
  smoke-tested against a stub, never in a real browser. **Run `pytest -q` and click through the app on your machine.**
  The real-package paths and all LLM providers are untested.
* The hashing embedder is lexical, not semantic. Install `sentence-transformers` for real semantic search; then re-tune
  thresholds with `--sweep-threshold`.
* The evaluation set (12 questions) is small and written for the synthetic corpus; high scores here do not predict
  performance on real documents. In mock mode it evaluates extractive output, not an LLM.
* Table handling assumes pipe-delimited rows (as in the sample PDFs). Real PDFs may extract cells one per line;
  expect to adapt `chunker.py` / `criteria_extractor.py`.
* Criteria extraction recognises CGPA, branch, active/past backlogs, 10th/12th and academic gap. Other company conditions
  (aptitude cut-offs, gender, graduation year) are not evaluated; the UI says so.
* Query rewriting is rule-based. Cross-company questions are recognised by pattern.
* Scanned PDFs are reported as needing OCR; no OCR is bundled.
* "Last verified" is the latest document effective date, not a manual verification record.

## Privacy

No names/USNs/phones requested. Student inputs live in Streamlit session state only. The provenance log
(`data/processed/answer_log.jsonl`) stores a 12-character question hash, status, mode and source files/versions;
set `PLACEBOT_LOG_ANSWERS=false` to disable. API keys come from environment variables; `.env` is git-ignored.

## Disclaimer

PlaceBot is a decision aid. Its verdicts are **not** official institutional decisions. Official placement rules and the
placement cell always take precedence.
