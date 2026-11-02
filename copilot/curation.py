"""Answers written from the official documents and checked against them.

Each answer in data/curated_answers.json names, for every statement, the document, the page and
the exact words that support it. Nothing is stored unless every quote is found on its page of
the downloaded PDF, so a statement cannot drift from its source. Run again after a refresh:
a document that changed makes any quote that moved or disappeared fail loudly.
"""
import csv
import json
import re
from pathlib import Path

import config
from copilot.brief import build_brief
from copilot.schemas import Answer, Citation, Finding, MissingItem, Persona, RoadmapStep

CURATED_PATH = config.ROOT / "data" / "curated_answers.json"
QUESTIONS_PATH = config.ROOT / "data" / "common_questions.json"


class QuoteNotFound(ValueError):
    """A quote is not on the page it is attributed to."""


def _letters_and_digits(text: str) -> str:
    """Lowercase letters and digits only. PDF text extraction splits words, swaps quote marks and
    dashes and breaks lines, so quotes are compared on their letters and digits alone."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def page_text(doc_id: str, page: int, official_dir=None) -> str:
    from pypdf import PdfReader
    pdf = PdfReader(Path(official_dir or config.OFFICIAL_DIR) / f"{doc_id}.pdf")
    if not 1 <= page <= len(pdf.pages):
        raise QuoteNotFound(f"{doc_id} has no page {page}")
    return pdf.pages[page - 1].extract_text() or ""


def verify_quote(doc_id: str, page: int, quote: str, official_dir=None) -> None:
    if _letters_and_digits(quote) not in _letters_and_digits(page_text(doc_id, page, official_dir)):
        raise QuoteNotFound(f"{doc_id} page {page}: quote not found: {quote[:70]}")


def _manifest(manifest_path=None) -> dict:
    with open(manifest_path or config.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def build_citation(spec: dict, manifest: dict, official_dir=None) -> Citation:
    verify_quote(spec["doc_id"], spec["page"], spec["quote"], official_dir)
    doc = manifest[spec["doc_id"]]
    return Citation(document_title=doc["title"], issuer=doc["issuer"], page=spec["page"], doc_id=spec["doc_id"],
                    page_url=f"{doc['url']}#page={spec['page']}", last_verified=doc.get("last_verified", ""),
                    published=doc.get("date_published", ""), quote=spec["quote"])


def build_answer(entry: dict, manifest: dict, official_dir=None) -> Answer:
    def cites(specs):
        return [build_citation(s, manifest, official_dir) for s in specs]
    answer = Answer(
        summary=entry["summary"],
        official_findings=[Finding(statement=f["statement"], citations=cites(f["citations"]))
                           for f in entry["findings"]],
        roadmap=[RoadmapStep(stage=s["stage"], step=s["step"], citations=cites(s["citations"]))
                 for s in entry["roadmap"]],
        missing_information=[MissingItem(**m) for m in entry.get("missing_information", [])],
        confidence=entry["confidence"], confidence_reason=entry.get("confidence_reason", ""))
    reasons = entry.get("review_reasons", [])
    if reasons:
        answer.needs_human_review, answer.review_reasons = True, reasons
        answer.ca_brief = build_brief(Persona(), answer)
    return answer


def build_entries(curated_path=None, questions_path=None, official_dir=None, manifest_path=None) -> list[dict]:
    """One stored entry per question in the question list, in its order. Raises if a question
    has no curated answer or any quote cannot be verified."""
    curated = {e["id"]: e for e in json.loads(Path(curated_path or CURATED_PATH).read_text(encoding="utf-8"))}
    manifest = _manifest(manifest_path)
    entries = []
    for q in json.loads(Path(questions_path or QUESTIONS_PATH).read_text(encoding="utf-8")):
        if q["id"] not in curated:
            raise KeyError(f"no curated answer for question '{q['id']}'")
        answer = build_answer(curated[q["id"]], manifest, official_dir)
        entries.append({"id": q["id"], "question": q["question"], "community_query": q.get("community_query", ""),
                        "answer": answer.model_dump(mode="json")})
    return entries
