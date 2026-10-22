"""Grounded question answering: rewrite -> retrieve -> evidence check -> generate -> verify citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from config.settings import Settings, get_settings
from src.companies.company_catalog import CompanyProfile, build_catalog, filter_companies
from src.eligibility.criteria_extractor import ACTIVE_BACKLOGS, BRANCH, MIN_CGPA, parse_number
from src.eligibility.rules_engine import normalize_branch
from src.eligibility.verdicts import SourceRef
from src.generation.citation_validator import (
    ValidationReport, strip_invalid_citations, validate_answer,
)
from src.generation.llm_client import LLMClient, extractive_answer, get_llm_client
from src.generation.prompts import INSUFFICIENT_TOKEN, NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from src.generation.query_rewriter import rewrite_query
from src.retrieval.context import assemble_context
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import normalize_company
from src.retrieval.text_utils import tokenize
from src.schemas import Chunk, RetrievalResult
from src.utils.audit import log_answer

POLICY_DOC_TYPES = ("placement_policy", "placement_faq")
_CROSS = re.compile(r"\b(which|what|list|show)\b.*\bcompan(y|ies)\b|\bcompanies\b.*\b(allow|accept|hire|eligible|open)\b", re.I)
_DEMO_NOTE = ("Demo mode: no language model was used. These are the most relevant lines extracted "
              "verbatim from the retrieved passages.")


@dataclass
class Citation:
    """A source card: always read back from a real indexed chunk."""

    label: str
    chunk_id: str
    filename: str
    company: Optional[str]
    doc_type: str
    section: str
    page: str
    effective_date: Optional[str]
    version: Optional[str]
    passage: str

    @classmethod
    def from_chunk(cls, label: str, c: Chunk) -> "Citation":
        return cls(label, c.chunk_id, c.source_filename, c.company, c.doc_type, c.section, c.page_label,
                   c.effective_date, c.version, c.text)


@dataclass
class Answer:
    question: str
    status: str  # "answered" | "not_found"
    text: str
    mode: str  # e.g. "mock (extractive demo, no LLM)" | "anthropic:..." | "deterministic (rules over extracted criteria)"
    citations: list[Citation] = field(default_factory=list)
    rewritten_query: str = ""
    warnings: list[str] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    retrieval: Optional[RetrievalResult] = None
    table: Optional[pd.DataFrame] = None
    company: Optional[str] = None

    @property
    def found(self) -> bool:
        return self.status == "answered"


class PlaceBot:
    """Owns the retriever, the company catalog and the LLM client."""

    def __init__(self, chunks: Sequence[Chunk], settings: Optional[Settings] = None,
                 retriever: Optional[HybridRetriever] = None, llm: Optional[LLMClient] = None) -> None:
        self.settings = settings or get_settings()
        self.chunks = list(chunks)
        self.corpus = {c.chunk_id: c for c in self.chunks}
        self.retriever = retriever or HybridRetriever(self.chunks, self.settings)
        self.catalog = build_catalog(self.chunks)
        self.llm_fallback_reason: Optional[str] = None
        if llm is None:
            llm, self.llm_fallback_reason = get_llm_client(self.settings)
        self.llm = llm

    # ------------------------------------------------------------------ public
    def ask(self, question: str, history: Sequence[tuple[str, str]] = (), company: Optional[str] = None,
            doc_types: Optional[Sequence[str]] = None, kind: str = "chat") -> Answer:
        question = question.strip()
        if not question:
            return Answer(question, "not_found", "Please enter a question.", self.llm.name)
        rw = rewrite_query(question, history, self.retriever.companies)
        company = company or rw.company
        if company is None and not doc_types and _CROSS.search(question):
            cross = self._cross_company(question, rw.query)
            if cross is not None:
                return cross
        res = self.retriever.search(rw.query, company=company, doc_types=doc_types)
        if not res.evidence_sufficient:
            ans = Answer(question, "not_found", NOT_FOUND_MESSAGE, self.llm.name, rewritten_query=rw.query,
                         retrieval=res, company=company, warnings=[f"Evidence check: {res.reason}."])
            log_answer(kind, question, "not_found", self.llm.name, [], self.settings)
            return ans
        ans = self._generate(question, rw.query, res, company, history)
        if rw.note:
            ans.warnings.append(f"Query rewritten: {rw.note}.")
        log_answer(kind, question, ans.status, ans.mode, [self.corpus[c.chunk_id] for c in ans.citations],
                   self.settings)
        return ans

    def ask_policy(self, question: str, history: Sequence[tuple[str, str]] = ()) -> Answer:
        """Policy assistant: restricts retrieval to policy and FAQ documents and asks for clause citations."""
        ans = self.ask(question, history, doc_types=POLICY_DOC_TYPES, kind="policy")
        if ans.found:
            ans.warnings.append("Company circulars may impose stricter conditions than the general policy; "
                                "check the company's own circular and confirm with the placement cell.")
        return ans

    # --------------------------------------------------------------- internals
    def _generate(self, question: str, rq: str, res: RetrievalResult, company: Optional[str],
                  history: Sequence[tuple[str, str]]) -> Answer:
        bundle = assemble_context(res.results, self.settings.context_token_budget)
        label_to_id = bundle.label_to_chunk_id()
        id_to_num = {cid: int(l[1:]) for l, cid in label_to_id.items()}
        exclude = (tokenize(normalize_company(company)) + tokenize(company)) if company else []
        warnings: list[str] = []
        versions = [n for r in res.results for n in r.notes if "multiple versions" in n]
        if versions:
            warnings.append("Multiple versions of a document were retrieved. Both are shown; "
                            "verify which one applies with the placement cell.")
        if bundle.dropped_for_budget:
            warnings.append(f"{bundle.dropped_for_budget} passage(s) were left out to respect the context budget.")

        mode = self.llm.name
        text = ""
        if not self.llm.is_mock:
            try:
                raw = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, bundle.text, list(history)))
            except Exception as exc:  # network/provider errors must not crash the UI
                raw = ""
                warnings.append(f"LLM call failed ({type(exc).__name__}); showing extractive lines instead.")
            if INSUFFICIENT_TOKEN in raw:
                return Answer(question, "not_found", NOT_FOUND_MESSAGE, mode, rewritten_query=rq, retrieval=res,
                              company=company, warnings=warnings + ["The model reported insufficient evidence."])
            text = raw
        report: Optional[ValidationReport] = None
        if text:
            report = validate_answer(text, label_to_id, self.corpus)
            if report.invalid_labels or report.missing_chunks:
                text = strip_invalid_citations(text, report.invalid_labels + report.missing_chunks)
                report = validate_answer(text, label_to_id, self.corpus)
            if not report.ok:
                warnings.append("The model's answer failed citation/number verification "
                                f"(uncited numeric claims: {len(report.uncited_numeric)}, unsupported numbers: "
                                f"{len(report.unsupported_numbers)}); showing extractive lines instead.")
                text = ""
        if not text:
            mode = "mock (extractive demo, no LLM)" if self.llm.is_mock else f"{self.llm.name} -> extractive fallback"
            text = extractive_answer(rq, res.results, id_to_num, exclude_terms=exclude)
            if self.llm.is_mock:
                warnings.insert(0, _DEMO_NOTE)
                if self.llm_fallback_reason:
                    warnings.insert(0, self.llm_fallback_reason)
            if not text:  # retrieval looked sufficient but no line matched: stay honest
                return Answer(question, "not_found", NOT_FOUND_MESSAGE, mode, rewritten_query=rq, retrieval=res,
                              company=company, warnings=warnings + ["No passage line matched the question terms."])
            report = validate_answer(text, label_to_id, self.corpus)
        used = [l for l in report.cited_labels] if report else []
        cites = [Citation.from_chunk(l, self.corpus[label_to_id[l]]) for l in used if l in label_to_id]
        return Answer(question, "answered", text, mode, cites, rq, warnings, report, res, company=company)

    # ---------------------------------------------------- cross-company questions
    def _parse_profile_from_query(self, q: str) -> tuple[Optional[str], Optional[float], Optional[int]]:
        branches: dict[str, str] = {}
        for p in self.catalog.values():
            if p.criteria:
                for c in p.criteria.by_field.get(BRANCH, []):
                    if isinstance(c.value, tuple):
                        for b in c.value:
                            branches[normalize_branch(b)] = b
        qtoks = {normalize_branch(t) for t in re.findall(r"[A-Za-z]+", q)}
        branch = next((branches[k] for k in branches if k in qtoks), None)
        m = re.search(r"(\d+(?:\.\d+)?)\s*cgpa|cgpa\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)", q, re.I)
        cgpa = float(next(g for g in m.groups() if g)) if m else None
        m2 = re.search(r"(\w+)\s+(?:active\s+)?backlogs?", q, re.I)
        backlogs = int(parse_number(m2.group(1))) if m2 and parse_number(m2.group(1)) is not None else None
        return branch, cgpa, backlogs

    def _cross_company(self, question: str, rq: str) -> Optional[Answer]:
        branch, cgpa, backlogs = self._parse_profile_from_query(rq)
        if branch is None and cgpa is None and backlogs is None:
            return None
        profiles = list(self.catalog.values())
        sure = filter_companies(profiles, branch=branch, cgpa=cgpa, active_backlogs=backlogs, include_unknown=False)
        maybe = [p for p in filter_companies(profiles, branch=branch, cgpa=cgpa, active_backlogs=backlogs,
                                             include_unknown=True) if p not in sure]
        used = ", ".join(x for x in [f"branch {branch}" if branch else "", f"CGPA {cgpa:g}" if cgpa is not None else "",
                                     f"{backlogs} active backlog(s)" if backlogs is not None else ""] if x)
        cites: list[Citation] = []
        seen: dict[str, str] = {}

        def cite(src: SourceRef) -> str:
            if src.chunk_id not in seen:
                seen[src.chunk_id] = f"S{len(seen) + 1}"
                cites.append(Citation.from_chunk(seen[src.chunk_id], self.corpus[src.chunk_id]))
            return f"[{seen[src.chunk_id]}]"

        def refs(p: CompanyProfile) -> str:
            marks = []
            for f in (BRANCH, MIN_CGPA, ACTIVE_BACKLOGS):
                res = p.criteria.resolve(f) if p.criteria else None
                if res and res.candidates and ((f == BRANCH and branch) or (f == MIN_CGPA and cgpa is not None)
                                               or (f == ACTIVE_BACKLOGS and backlogs is not None)):
                    for c in res.candidates:
                        marks.append(cite(c.source))
            return "".join(dict.fromkeys(marks))

        lines = [f"Checked criteria for: {used}. Only the criteria you mentioned were applied."]
        if sure:
            lines.append("Companies whose stated criteria you meet:")
            lines += [f"- {p.name}: branches {p.branches_text}; minimum CGPA {p.min_cgpa_text}; "
                      f"backlogs {p.backlog_text} {refs(p)}" for p in sure]
        else:
            lines.append("No indexed company was confirmed as matching.")
        if maybe:
            lines.append("Could not be determined (missing, ambiguous or conflicting documents):")
            lines += [f"- {p.name}: branches {p.branches_text}; minimum CGPA {p.min_cgpa_text} {refs(p)}" for p in maybe]
        lines.append(f"This covers only the {len(profiles)} companies in the indexed documents, not all employers, "
                     "and does not check 10th/12th percentages. Use the Eligibility Checker for a full verdict.")
        text = "\n".join(lines)
        table = pd.DataFrame([{"Company": p.name, "Branches": p.branches_text, "Min CGPA": p.min_cgpa_text,
                               "Backlogs": p.backlog_text,
                               "Match": "Yes" if p in sure else ("Undetermined" if p in maybe else "No")}
                              for p in profiles])
        report = validate_answer(text, {c.label: c.chunk_id for c in cites}, self.corpus)
        # Numbers here echo the user's own inputs or Python-computed counts, not retrieved claims.
        report.uncited_numeric = []
        report.unsupported_numbers = []  # dates come from chunk metadata, values from structured extraction
        ans = Answer(question, "answered", text, "deterministic (rules over extracted criteria; no LLM)", cites,
                     rq, ["Counts and matches are computed in Python from indexed criteria, not generated by a model."],
                     report, None, table)
        log_answer("chat", question, "answered", ans.mode, [self.corpus[c.chunk_id] for c in cites], self.settings)
        return ans
