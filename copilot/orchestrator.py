"""The answer pipeline. The order of steps is fixed in code, not chosen by the model.

LLM tool calling happens inside step 6 only; it is NOT an autonomous agent.
Every step is its own small function so it can be tested alone.
"""
import logging
from dataclasses import dataclass, field

import config
from copilot import llm
from copilot.brief import build_brief
from copilot.compare import compare_threads
from copilot.persona import extract_persona, plan_queries
from copilot.prompts import ANSWER_ASSEMBLY, TOOL_CALLING_SYSTEM
from copilot.retrieval import retrieve
from copilot.schemas import (Answer, AnswerDraft, Citation, Finding, MissingItem, Persona,
                             RoadmapStep)
from copilot.text import plain
from copilot.tools import (days_in_india_calculator, fetch_community, scrub, search_community,
                           search_official_docs)
from copilot.updates import compare_updates, fetch_official_updates, last_checked
from copilot.web_queries import write_web_queries
from guardrails.advice import classify_scope, scope_response
from guardrails.escalation import escalation_reasons
from guardrails.grounding import check_grounding
from guardrails.output_checks import run_output_checks
from guardrails.pii import check_pii


log = logging.getLogger(__name__)


def _record(trace: list, step: str, detail: str) -> None:
    """Keep a step outcome in the run trace and the log. Only guard names, outcomes and
    scrubbed tool arguments go here, never raw user input."""
    trace.append((step, detail))
    log.info("%s: %s", step, detail)


@dataclass
class PipelineResult:
    # pii_blocked | refused | us_side_tax | abstained | answered
    kind: str
    message: str = ""
    answer: Answer | None = None
    persona: Persona | None = None
    chunks: list = field(default_factory=list)
    trace: list = field(default_factory=list)   # (step, detail): never raw user input
    links: list = field(default_factory=list)
    model_used: str = ""


# --- steps ------------------------------------------------------------------
def retrieve_merged(queries: list[str]):
    """Run every planned query, keep the best-scoring distinct passages."""
    seen, merged = set(), []
    for q in queries:
        for c in retrieve(q, config.TOP_K):
            key = (c.metadata["doc_id"], c.metadata["page"], c.text[:80])
            if key not in seen:
                seen.add(key)
                merged.append(c)
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged[: config.MERGED_TOP_N]


def grounding_for(question: str):
    """Is the user's own question supported by the documents? Judged on the question alone:
    the planned queries add search phrases built from the profile, and those match the
    corpus whatever was asked, so they must not decide whether to answer."""
    return check_grounding(retrieve(question, config.TOP_K))


def merge_chunks(chunks, extra):
    """Add passages found by a tool call, keeping the best-scoring distinct ones."""
    seen = {(c.metadata["doc_id"], c.metadata["page"], c.text[:80]) for c in chunks}
    merged = list(chunks) + [c for c in extra
                             if (c.metadata["doc_id"], c.metadata["page"], c.text[:80]) not in seen]
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged[: config.MERGED_TOP_N]


def numbered_sources(chunks) -> str:
    return "\n\n".join(f"[{i}] {c.metadata['title']} ({c.metadata['issuer']}), page {c.metadata['page']}:\n{c.text}"
                       for i, c in enumerate(chunks, start=1))


def citation_for(chunk) -> Citation:
    m = chunk.metadata
    return Citation(document_title=m["title"], issuer=m["issuer"], page=m["page"], doc_id=m["doc_id"],
                    page_url=m["page_url"], last_verified=m.get("last_verified", ""),
                    published=m.get("date_published", ""))


def build_answer(draft: AnswerDraft, chunks) -> tuple[Answer, list[str]]:
    """Turn the model's draft (cites by number) into an Answer with real citations."""
    problems = []

    def cites(ids, text):
        out = []
        for i in ids:
            if 1 <= i <= len(chunks):
                out.append(citation_for(chunks[i - 1]))
            else:
                problems.append(f"source number {i} does not exist: {text[:40]}")
        return out

    findings, steps = [], []
    for f in draft.findings:
        c = cites(f.source_ids, f.statement)
        if c:
            findings.append(Finding(statement=plain(f.statement), citations=c))
    for s in draft.roadmap:
        c = cites(s.source_ids, s.step)
        if c:
            steps.append(RoadmapStep(stage=s.stage, step=plain(s.step), citations=c))
    missing = [MissingItem(field=m.field, why_it_matters=plain(m.why_it_matters),
                           if_this_then_that=plain(m.if_this_then_that))
               for m in draft.missing_information]
    answer = Answer(summary=plain(draft.summary), official_findings=findings, roadmap=steps,
                    missing_information=missing, confidence=draft.confidence,
                    confidence_reason=plain(draft.confidence_reason))
    return answer, problems


def find_newer_publications(question: str, trace: list) -> list[dict]:
    """Official-site items published after the last document check. A failed or
    unavailable search never blocks the answer; it only shows up in the trace."""
    since = last_checked()
    if since is None:
        _record(trace, "official_updates", "skipped: no check date on record")
        return []
    try:
        found = fetch_official_updates(question, since)
    except Exception as e:
        _record(trace, "official_updates", f"skipped: {type(e).__name__}")
        return []
    _record(trace, "official_updates", f"{len(found)} published since {since.isoformat()}")
    return found


def tool_calling_step(question: str, persona: Persona, trace: list, community: bool = True,
                      community_query: str = ""):
    """The model chooses among its tools. Returns (days result, community threads, extra passages).

    Official retrieval has already run, so grounding never depends on this step; an extra
    official search by the model only adds passages. Community search is left out when the
    user opted out. The model decides whether to search the community; the query itself is
    the one written from what the user meant (community_query), not the user's raw wording."""
    days, threads, extra = None, [], []
    community_searched = False
    tools = [search_official_docs, days_in_india_calculator] + ([search_community] if community else [])
    try:
        msg = llm.tool_call_step(TOOL_CALLING_SYSTEM,
                                 f"Question: {question}\nContext: {persona.model_dump_json()}", tools)
    except Exception as e:
        _record(trace, "tool_calling", f"skipped: {type(e).__name__}")
        msg = None
    for call in getattr(msg, "tool_calls", []) or []:
        name, args = call["name"], call["args"]
        _record(trace, "tool_call", f"{name}({scrub(str(args))})")
        try:
            if name == "days_in_india_calculator":
                days = days_in_india_calculator.invoke(args)
            elif name == "search_community" and community:
                community_searched = True
                threads += fetch_community(community_query or args.get("query", ""))
            elif name == "search_official_docs":
                extra += retrieve(args.get("query", ""), config.TOP_K)
        except Exception as e:
            _record(trace, "tool_error", f"{name}: {type(e).__name__}")
    if community and not community_searched:
        # Smaller models often skip this tool. The user asked for community results, so search anyway.
        _record(trace, "community_search", "model did not call it, searching anyway")
        try:
            threads += fetch_community(community_query or question)
        except Exception as e:
            _record(trace, "tool_error", f"community_search: {type(e).__name__}")
    return days, threads[: config.COMMUNITY_MAX_RESULTS], extra


def _finish(answer: Answer, persona, reasons: list[str]) -> Answer:
    answer.needs_human_review = bool(reasons) or answer.abstained
    answer.review_reasons = reasons
    if answer.needs_human_review:
        answer.ca_brief = build_brief(persona, answer)
    return answer


def abstain(persona, chunks_note: str, trace) -> PipelineResult:
    reasons = ["No official text supports an answer: " + chunks_note]
    ans = Answer(summary=("I couldn't find this in the official documents, so I'm not going to guess. "
                          "A chartered accountant is the best person to check this with."),
                 abstained=True, confidence="low", confidence_reason=chunks_note)
    ans = _finish(ans, persona, reasons)
    return PipelineResult("abstained", ans.summary, ans, persona, trace=trace, model_used=llm.last_model_used)


# --- the pipeline -------------------------------------------------------------
def run_pipeline(question: str, persona: Persona | None = None, community: bool = True) -> PipelineResult:
    trace: list = []

    # 1. PII guard: nothing reaches a model before this passes.
    typed = question + " " + (persona.concerns if persona else "")
    pii = check_pii(typed)
    _record(trace, "pii_guard", "blocked" if pii.blocked else "passed")
    if pii.blocked:
        return PipelineResult("pii_blocked", pii.message, trace=trace)

    # 2. Scope guard
    scope = classify_scope(question)
    _record(trace, "scope_guard", scope.label)
    resp = scope_response(scope.label)
    if resp["action"] == "refuse":
        return PipelineResult("refused", resp["message"], trace=trace, model_used=llm.last_model_used)
    if resp["action"] == "escalate_us_tax":
        p = persona or Persona()
        ans = Answer(summary=resp["message"], confidence="low", confidence_reason="US-side question: not answered")
        ans = _finish(ans, p, escalation_reasons(p, scope_label="us_side_tax", confidence="low"))
        return PipelineResult("us_side_tax", resp["message"], ans, p, trace=trace, links=resp["links"],
                              model_used=llm.last_model_used)

    # 3. Persona
    persona = persona or extract_persona(question)
    _record(trace, "persona", "ready")
    queries = plan_queries(persona, question)
    _record(trace, "queries", f"{len(queries)} planned")
    chunks = retrieve_merged(queries)
    grounding = grounding_for(question)
    _record(trace, "grounding_guard", "passed" if grounding.passed else "abstain")
    if not grounding.passed:
        return abstain(persona, grounding.reason, trace)

    # 5. Anything newer than our last document check? Searches use queries written from
    #    what the user meant, in the words each kind of site would use.
    web = write_web_queries(question, persona)
    _record(trace, "web_queries", scrub(f"community: {web.community} | official: {web.official}"))
    newer = find_newer_publications(web.official, trace)

    # 6. Tool calling (days calculator, community search)
    days, threads, extra = tool_calling_step(question, persona, trace, community, web.community)
    chunks = merge_chunks(chunks, extra)

    # 7. Compare community and newer publications with the official passages
    community_findings = compare_threads(threads, chunks)
    updates = [u for u in compare_updates(newer, chunks) if u.verdict != "unrelated"]
    if newer:
        _record(trace, "official_updates_compared", f"{len(updates)} of {len(newer)} relevant")

    # 8 + 9. Assemble, check, regenerate once, then abstain
    context = (f"Persona: {persona.model_dump_json()}\nQuestion: {question}\n"
               f"Days per year computed by tool (information only): {days['days_per_year'] if days else 'none supplied'}\n\n"
               f"OFFICIAL SOURCES:\n{numbered_sources(chunks)}")
    problems: list[str] = []
    answer = None
    for attempt in range(1 + config.MAX_REGENERATIONS):
        extra = ("\n\nYour previous answer failed these checks, fix them:\n- " + "\n- ".join(problems)) if problems else ""
        draft = llm.structured_call(AnswerDraft, ANSWER_ASSEMBLY, context + extra)
        answer, build_problems = build_answer(draft, chunks)
        problems = build_problems + run_output_checks(persona, answer, chunks)
        _record(trace, "output_checks", "passed" if not problems else f"failed ({len(problems)})")
        if not problems:
            break
    if problems:
        return abstain(persona, "the answer failed its safety checks twice", trace)

    # 10. Escalation and CA brief
    answer.community_findings = community_findings
    answer.official_updates = updates
    reasons = escalation_reasons(persona, "in_scope", answer.confidence, community_findings)
    if any(u.verdict == "changes" for u in updates):
        reasons.append("A newer official publication may change this answer, so check it before acting")
    answer = _finish(answer, persona, reasons)
    return PipelineResult("answered", answer.summary, answer, persona, chunks, trace,
                          model_used=llm.last_model_used)
