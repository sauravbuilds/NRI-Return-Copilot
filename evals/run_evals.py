"""Run the golden set through the real pipeline and write evals/results.md and results.csv.

Run:  python evals/run_evals.py

Needs at least one LLM key for the full run. Without one, only the metrics that do not
need a model are measured and the rest say NOT RUN (a metric is never filled with an estimate).
Never edit the golden set to improve a result.
"""
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot import llm  # noqa: E402
from copilot.retrieval import Chunk, retrieve  # noqa: E402
from copilot.schemas import JudgeVerdict, Persona  # noqa: E402
from guardrails.grounding import check_grounding  # noqa: E402
from guardrails.output_checks import check_citations  # noqa: E402
from guardrails.pii import check_pii  # noqa: E402

GOLDEN = config.EVALS_DIR / "golden_set.json"
PAIRS = config.EVALS_DIR / "contradiction_pairs.json"


def total_llm_calls() -> int:
    return sum(llm.calls_by_model.values())


def hit(chunks, sources) -> bool:
    return any(c.metadata["doc_id"] == s["doc_id"] and s["pages"][0] <= c.metadata["page"] <= s["pages"][1]
               for c in chunks for s in sources)


# --- one question ----------------------------------------------------------------
def run_question(q: dict, full: bool) -> dict:
    """Returns {'ok': bool|None, 'detail': str, ...}. ok=None means not measured."""
    kind = q["type"]
    if kind == "pii":
        before = total_llm_calls()
        res = check_pii(q["question"])
        return {"ok": res.blocked and total_llm_calls() == before, "detail": f"categories={res.categories}"}

    if not full:
        if kind == "answerable":
            if check_pii(q["question"]).blocked:
                return {"ok": False, "detail": "falsely blocked as PII"}
            chunks = retrieve(q["question"])
            ok = hit(chunks, q["expected_source"])
            return {"ok": ok, "detail": "hit" if ok else "no expected chunk in top results"}
        if kind == "out_of_corpus":
            g = check_grounding(retrieve(q["question"]))
            return {"ok": not g.passed, "detail": f"retrieval guard only: {g.reason}"}
        return {"ok": None, "detail": "needs an LLM"}

    from copilot import orchestrator
    persona = Persona.model_validate(q["persona"]) if "persona" in q else None
    try:
        r = orchestrator.run_pipeline(q["question"], persona)
    except Exception as e:
        return {"ok": None, "detail": f"error: {type(e).__name__}: {e}"}
    base = {"result": r, "detail": f"kind={r.kind}"}

    if kind == "answerable":
        ok = r.kind == "answered" and hit(r.chunks, q["expected_source"])
        why = f", reason={r.answer.review_reasons[0]}" if r.kind == "abstained" and r.answer.review_reasons else ""
        return {**base, "ok": ok,
                "detail": f"kind={r.kind}, retrieval_hit={hit(r.chunks, q['expected_source'])}{why}"}
    if kind == "out_of_corpus":
        return {**base, "ok": r.kind in ("abstained", "refused")}
    if kind == "investment_advice":
        return {**base, "ok": r.kind == "refused"}
    if kind == "us_side_tax":
        return {**base, "ok": r.kind == "us_side_tax" and bool(r.links)}
    if kind == "escalation":
        return {**base, "ok": bool(r.answer and r.answer.needs_human_review)}
    if kind == "missing_information":
        listed = {m.field for m in r.answer.missing_information} if r.answer and r.kind == "answered" else set()
        missing = [f for f in q["expected_missing"] if f not in listed]
        return {**base, "ok": r.kind == "answered" and not missing, "detail": f"kind={r.kind}, not listed={missing}"}
    return {**base, "ok": None}


# --- metrics that need extra model calls ----------------------------------------------
def passage_text(result, citation) -> str:
    return "\n".join(c.text for c in result.chunks
                     if c.metadata["doc_id"] == citation.doc_id and c.metadata["page"] == citation.page)


def judge_claims(results: list) -> tuple[list[dict], float | None]:
    from copilot.prompts import EVAL_JUDGE
    rows = []
    for qid, r in results:
        if r.kind != "answered":
            continue
        for item, text in [*[(f, f.statement) for f in r.answer.official_findings],
                           *[(s, s.step) for s in r.answer.roadmap]]:
            passage = "\n---\n".join(passage_text(r, c) for c in item.citations)
            try:
                v = llm.structured_call(JudgeVerdict, EVAL_JUDGE, f"CLAIM:\n{text}\n\nCITED PASSAGE TEXT:\n{passage}")
            except Exception as e:
                rows.append({"question": qid, "claim": text, "passage": passage, "judge": "error", "reason": str(e)})
                continue
            rows.append({"question": qid, "claim": text, "passage": passage, "judge": v.label, "reason": v.reason})
            time.sleep(config.EVAL_SLEEP_SECONDS)
    judged = [x for x in rows if x["judge"] != "error"]
    rate = (sum(x["judge"] != "supported" for x in judged) / len(judged)) if judged else None
    return rows, rate


def page_text(doc_id: str, page: int) -> str:
    from pypdf import PdfReader
    return PdfReader(config.OFFICIAL_DIR / f"{doc_id}.pdf").pages[page - 1].extract_text()


def contradiction_accuracy() -> tuple[float | None, int, list]:
    from copilot.compare import compare_snippet
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    correct, compared, wrong = 0, 0, []
    for p in pairs:
        chunk = Chunk(page_text(p["doc_id"], p["page"]),
                      {"title": p["doc_id"], "page": p["page"], "doc_id": p["doc_id"]}, 1.0)
        try:
            label = compare_snippet(p["snippet"], [chunk]).label
        except Exception as e:  # no model answered: not measured, so it is neither right nor wrong
            wrong.append((p["id"], f"not compared, {type(e).__name__}"))
            continue
        compared += 1
        if label == p["label"]:
            correct += 1
        else:
            wrong.append((p["id"], f"expected {p['label']}, got {label}"))
        time.sleep(config.EVAL_SLEEP_SECONDS)
    return (correct / compared if compared else None), compared, wrong


def retrieval_stage_metrics() -> dict:
    """What the document search and the grounding check do with each golden question, without
    any model call: the share of answerable questions whose expected page is retrieved, the
    share that clear the abstention threshold, and the share of out-of-corpus questions that
    abstain. evals/tune.py uses this to compare chunk sizes and thresholds."""
    from copilot.orchestrator import grounding_for, retrieve_merged
    from copilot.persona import plan_queries
    hits, passed, abstained = [], [], []
    for q in json.loads(GOLDEN.read_text(encoding="utf-8")):
        if q["type"] not in ("answerable", "out_of_corpus"):
            continue
        persona = Persona.model_validate(q["persona"]) if "persona" in q else Persona()
        chunks = retrieve_merged(plan_queries(persona, q["question"]))
        grounded = grounding_for(q["question"]).passed
        if q["type"] == "answerable":
            hits.append(hit(chunks, q["expected_source"]))
            passed.append(grounded)
        else:
            abstained.append(not grounded)

    def share(flags):
        return sum(flags) / len(flags) if flags else None
    return {"retrieval_hit_rate": share(hits), "answerable_passing": share(passed),
            "out_of_corpus_abstained": share(abstained),
            "n_answerable": len(hits), "n_out_of_corpus": len(abstained)}


# --- reporting ----------------------------------------------------------------------------
def fmt(x) -> str:
    return "NOT RUN" if x is None else f"{x:.2f}"


def rate(results, types):
    rows = [r for r in results if r["type"] in types and r["ok"] is not None]
    total = len([r for r in results if r["type"] in types])
    return (sum(r["ok"] for r in rows) / len(rows) if rows else None), len(rows), total


def main() -> dict:
    """Run everything and return the metrics (evals/tune.py calls this after changing config)."""
    full = llm.llm_available()
    questions = json.loads(GOLDEN.read_text(encoding="utf-8"))
    results, pipeline_results = [], []
    for q in questions:
        out = run_question(q, full)
        results.append({"id": q["id"], "type": q["type"], "ok": out["ok"], "detail": out["detail"]})
        if out.get("result") is not None:
            pipeline_results.append((q["id"], out["result"]))
        if full and q["type"] != "pii":
            time.sleep(config.EVAL_SLEEP_SECONDS)

    m = {}
    m["retrieval_hit_rate"], hit_n, hit_t = rate(results, {"answerable"})
    m["pii_catch_rate"], pii_n, pii_t = rate(results, {"pii"})
    m["refusal_accuracy"], ref_n, ref_t = rate(results, {"out_of_corpus", "investment_advice"}) if full else (None, 0, 8)
    ooc_rate, ooc_n, ooc_t = rate(results, {"out_of_corpus"})
    m["missing_info_rate"], mi_n, mi_t = rate(results, {"missing_information"}) if full else (None, 0, 2)

    # Retrieval hit rate in full mode counts retrieval only (answer kind is reported separately).
    if full:
        ans_q = [(q, r) for q, r in zip(questions, results) if q["type"] == "answerable"]
        hits = [hit(dict(pipeline_results)[q["id"]].chunks, q["expected_source"])
                for q, _ in ans_q if q["id"] in dict(pipeline_results)]
        m["retrieval_hit_rate"] = sum(hits) / len(hits) if hits else None
        hit_n = len(hits)

    audit_rows, m["unsupported_claim_rate"] = ([], None)
    if full:
        audit_rows, m["unsupported_claim_rate"] = judge_claims(pipeline_results)

    # Escalation precision/recall: abstentions are excluded (they always get a brief).
    m["escalation_precision"] = m["escalation_recall"] = None
    if full:
        expected = {q["id"] for q in questions if q["type"] in ("us_side_tax", "escalation")}
        predicted = {qid for qid, r in pipeline_results
                     if r.kind in ("answered", "us_side_tax") and r.answer and r.answer.needs_human_review}
        tp = len(expected & predicted)
        m["escalation_precision"] = tp / len(predicted) if predicted else None
        m["escalation_recall"] = tp / len(expected) if expected else None
        answered = [(qid, r) for qid, r in pipeline_results if r.kind == "answered"]
        ok_cites = sum(1 for _, r in answered if not check_citations(r.answer, r.chunks))
        m["citation_completeness"] = ok_cites / len(answered) if answered else None
        m["contradiction_accuracy"], n_pairs, wrong_pairs = contradiction_accuracy()
    else:
        m["citation_completeness"] = m["contradiction_accuracy"] = None
        n_pairs, wrong_pairs = 0, []
    total_pairs = len(json.loads(PAIRS.read_text(encoding="utf-8")))

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    models = ", ".join(f"{k} x{v}" for k, v in llm.calls_by_model.items()) or "none (no LLM key set)"
    lines = [
        "# Eval results", "", f"Run: {now:%Y-%m-%d %H:%M}",
        f"Config: chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}, top_k={config.TOP_K}, "
        f"relevance_threshold={config.RELEVANCE_THRESHOLD}, embeddings={config.EMBEDDING_MODEL}",
        f"Chat models actually used: {models}",
        f"Golden set: {len(questions)} questions. Mode: {'full pipeline' if full else 'retrieval-only (no LLM key)'}",
        "", "## Metrics", "", "| Metric | Value | Measured on |", "|---|---|---|",
        f"| Retrieval hit rate | {fmt(m['retrieval_hit_rate'])} | {hit_n} of {hit_t} answerable questions |",
        f"| Unsupported-claim rate (lower is better) | {fmt(m['unsupported_claim_rate'])} | "
        f"{len([x for x in audit_rows if x['judge'] != 'error'])} claims judged |",
        f"| Refusal accuracy | {fmt(m['refusal_accuracy'])} | {ref_n} of {ref_t} out-of-corpus and advice questions |",
        f"| PII catch rate | {fmt(m['pii_catch_rate'])} | {pii_n} of {pii_t} PII inputs |",
        f"| Missing-information handling rate | {fmt(m['missing_info_rate'])} | {mi_n} of {mi_t} questions |",
        f"| Citation completeness | {fmt(m['citation_completeness'])} | answered questions |",
        f"| Escalation precision | {fmt(m['escalation_precision'])} | abstentions excluded |",
        f"| Escalation recall | {fmt(m['escalation_recall'])} | 5 questions expected to escalate |",
        f"| Contradiction-detection accuracy | {fmt(m['contradiction_accuracy'])} | {n_pairs} of {total_pairs} hand-labelled pairs compared |",
        f"| (Out-of-corpus abstention alone) | {fmt(ooc_rate)} | {ooc_n} of {ooc_t} |",
        "", "## Failed questions", ""]
    failed = [r for r in results if r["ok"] is False]
    lines += [f"- {r['id']} ({r['type']}): {r['detail']}" for r in failed] or ["- none"]
    lines += [f"- {pid} (contradiction pair): {why}" for pid, why in wrong_pairs]
    lines += ["", "## Not measured", ""]
    lines += [f"- {r['id']} ({r['type']}): {r['detail']}" for r in results if r["ok"] is None] or ["- none"]
    report = "\n".join(lines) + "\n"

    (config.EVALS_DIR / "results.md").write_text(report, encoding="utf-8")
    with open(config.EVALS_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "type", "ok", "detail"])
        w.writeheader()
        w.writerows(results)
    if audit_rows:  # 10 sampled claims for manual review
        random.Random(0).shuffle(audit_rows)
        with open(config.EVALS_DIR / "manual_audit.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["question", "claim", "passage", "judge", "reason", "human_verdict"])
            w.writeheader()
            w.writerows({**r, "human_verdict": ""} for r in audit_rows[:10])
    history = config.EVALS_DIR / "history"
    history.mkdir(exist_ok=True)
    shutil.copy(config.EVALS_DIR / "results.md", history / f"results_{stamp}.md")
    shutil.copy(config.EVALS_DIR / "results.csv", history / f"results_{stamp}.csv")
    print(report)
    return m


if __name__ == "__main__":
    main()
