"""Prepared answers for common questions.

Each answer is produced by the real pipeline from the official documents and stored together
with a fingerprint of the documents it was built from. When any document changes, the stored
answers no longer match and are ignored, so the app answers live again until they are
prepared afresh.
"""
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import config
from copilot.schemas import Answer

PREPARED_PATH = config.ROOT / "data" / "prepared_answers.json"
QUESTIONS_PATH = config.ROOT / "data" / "common_questions.json"


@dataclass
class PreparedAnswer:
    id: str
    question: str
    answer: Answer
    prepared_on: str


MIN_CITED_ITEMS = 2  # findings plus roadmap steps; each one carries a citation


def good_enough(result) -> bool:
    """Only a sourced answer with reasonable confidence and some substance is worth keeping.
    A definition ("What is Schedule FA?") may be all findings and no steps, so either counts."""
    answer = result.answer
    return bool(result.kind == "answered" and answer and answer.confidence != "low"
                and len(answer.official_findings) + len(answer.roadmap) >= MIN_CITED_ITEMS)


def documents_fingerprint(manifest_path=None) -> str:
    """A short hash of every document's id and content hash: it changes when any document does."""
    with open(manifest_path or config.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["doc_id"])
    return hashlib.sha256("|".join(f"{r['doc_id']}:{r['sha256']}" for r in rows).encode()).hexdigest()[:16]


def save(entries: list[dict], path=None, manifest_path=None, today: str | None = None) -> None:
    """entries: [{"id", "question", "answer": Answer as JSON-ready dict}]."""
    payload = {"documents": documents_fingerprint(manifest_path),
               "prepared_on": today or date.today().isoformat(), "answers": entries}
    Path(path or PREPARED_PATH).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")


def available(path=None, manifest_path=None) -> list[PreparedAnswer]:
    """The prepared answers that still match the current documents, or [] if there are none."""
    path = Path(path or PREPARED_PATH)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["documents"] != documents_fingerprint(manifest_path):
            return []
        return [PreparedAnswer(e["id"], e["question"], Answer.model_validate(e["answer"]), payload["prepared_on"])
                for e in payload["answers"]]
    except (KeyError, ValueError, OSError):
        return []
