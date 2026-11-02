"""Prepared answers: stored with the documents' fingerprint and ignored once the documents change."""
import csv
import json

from copilot import prepared
from copilot.schemas import Answer


def manifest(tmp_path, sha="aaa"):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "sha256"])
        w.writeheader()
        w.writerow({"doc_id": "d1", "sha256": sha})
    return path


def entry():
    return {"id": "q1", "question": "What happens to my NRO account?", "community_query": "nro account return",
            "answer": Answer(summary="Plain answer.").model_dump(mode="json")}


def test_answers_prepared_from_the_current_documents_are_available(tmp_path):
    store, docs = tmp_path / "prepared.json", manifest(tmp_path)
    prepared.save([entry()], store, docs, today="2026-11-02")
    items = prepared.available(store, docs)
    assert [(i.id, i.answer.summary, i.prepared_on) for i in items] == [("q1", "Plain answer.", "2026-11-02")]


def test_a_changed_document_makes_the_stored_answers_unavailable(tmp_path):
    store = tmp_path / "prepared.json"
    prepared.save([entry()], store, manifest(tmp_path, "aaa"))
    assert prepared.available(store, manifest(tmp_path, "bbb")) == []


def test_a_missing_or_damaged_file_means_no_prepared_answers(tmp_path):
    docs = manifest(tmp_path)
    assert prepared.available(tmp_path / "none.json", docs) == []
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert prepared.available(tmp_path / "bad.json", docs) == []
    (tmp_path / "odd.json").write_text(json.dumps({"documents": "x"}), encoding="utf-8")
    assert prepared.available(tmp_path / "odd.json", docs) == []


def test_the_fingerprint_changes_with_any_document_hash(tmp_path):
    assert prepared.documents_fingerprint(manifest(tmp_path, "a")) != prepared.documents_fingerprint(manifest(tmp_path, "b"))


def test_common_questions_file_is_well_formed():
    items = json.loads(prepared.QUESTIONS_PATH.read_text(encoding="utf-8"))
    assert len(items) == 20 and len({i["id"] for i in items}) == len(items)
    assert all(i["question"].endswith("?") for i in items)
