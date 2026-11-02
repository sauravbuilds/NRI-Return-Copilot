"""Plumbing test for the eval runner with a fake pipeline. These numbers mean nothing;
this only proves the runner computes metrics, writes its files and never edits the golden set."""
import json

import config
from copilot import llm, orchestrator
from copilot.orchestrator import PipelineResult
from copilot.retrieval import Chunk
from copilot.schemas import (Answer, Citation, Finding, JudgeVerdict, MissingItem, Persona, VerdictResult)
from evals import run_evals


def test_runner_retrieval_only_mode_marks_llm_metrics_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "llm_available", lambda: False)
    monkeypatch.setattr(run_evals, "retrieve", lambda q: [])
    monkeypatch.setattr(config, "EVALS_DIR", tmp_path)
    (tmp_path / "x").mkdir()
    monkeypatch.setattr(run_evals, "GOLDEN", config.ROOT / "evals" / "golden_set.json")
    m = run_evals.main()
    report = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert m["unsupported_claim_rate"] is None and m["pii_catch_rate"] == 1.0
    assert "NOT RUN" in report and "retrieval-only" in report


def test_runner_full_mode_with_fake_pipeline_writes_all_metrics(tmp_path, monkeypatch):
    golden_before = (config.ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8")
    meta = {"doc_id": "it_guide_fsi_tr_fa_2026", "title": "T", "issuer": "I", "page": 2,
            "page_url": "http://x#page=2", "last_verified": "2026-01-01"}
    chunk = Chunk("passage", meta, 0.9)
    cite = Citation(document_title="T", issuer="I", page=2, doc_id=meta["doc_id"], page_url="http://x#page=2",
                    last_verified="2026-01-01")

    def fake_pipeline(question, persona=None, community=True):
        ans = Answer(summary="s", official_findings=[Finding(statement="claim", citations=[cite])],
                     missing_information=[MissingItem(field=f, why_it_matters="w", if_this_then_that="i")
                                          for f in config.REQUIRED_PERSONA_FIELDS])
        return PipelineResult("answered", "s", ans, Persona(), [chunk])

    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(orchestrator, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(llm, "structured_call", lambda schema, s, u: (
        JudgeVerdict(label="supported", reason="r") if schema is JudgeVerdict else VerdictResult(label="agrees", reason="r")))
    monkeypatch.setattr(config, "EVAL_SLEEP_SECONDS", 0)
    monkeypatch.setattr(run_evals, "page_text", lambda d, p: "official text")
    monkeypatch.setattr(config, "EVALS_DIR", tmp_path)
    m = run_evals.main()
    assert m["unsupported_claim_rate"] == 0.0 and m["citation_completeness"] == 1.0
    assert (tmp_path / "results.md").exists() and (tmp_path / "results.csv").exists()
    assert (tmp_path / "manual_audit.csv").exists()
    assert (config.ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8") == golden_before


def test_golden_set_has_required_composition():
    qs = json.loads((config.ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
    counts = {}
    for q in qs:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    assert counts == {"answerable": 11, "out_of_corpus": 4, "investment_advice": 4, "pii": 4,
                      "us_side_tax": 3, "escalation": 2, "missing_information": 2}
    assert len({q["id"] for q in qs}) == 30


def test_ten_contradiction_pairs_with_valid_labels():
    pairs = json.loads((config.ROOT / "evals" / "contradiction_pairs.json").read_text(encoding="utf-8"))
    assert len(pairs) == 10 and {p["label"] for p in pairs} <= {"agrees", "contradicts", "unrelated"}
