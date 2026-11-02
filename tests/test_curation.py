"""Answers written from the documents: every quote must be on its page, every question answered."""
import json

import pytest

from copilot import curation, prepared
from guardrails.pii import check_pii

QUESTIONS = json.loads(curation.QUESTIONS_PATH.read_text(encoding="utf-8"))
CURATED = json.loads(curation.CURATED_PATH.read_text(encoding="utf-8"))


def test_every_question_has_an_answer_and_every_answer_has_a_question():
    assert [q["id"] for q in QUESTIONS] == [a["id"] for a in CURATED]


def test_every_quote_is_on_the_page_it_is_attributed_to():
    entries = curation.build_entries()  # raises QuoteNotFound for the first quote that is not on its page
    assert len(entries) == len(QUESTIONS) == 20


def test_every_cited_statement_carries_its_quote_document_issuer_and_page_link():
    for entry in curation.build_entries():
        answer = entry["answer"]
        items = answer["official_findings"] + answer["roadmap"]
        assert len(answer["official_findings"]) >= 2 and answer["roadmap"], entry["id"]
        for item in items:
            for c in item["citations"]:
                assert c["quote"] and c["document_title"] and c["issuer"], entry["id"]
                assert c["page_url"].endswith(f"#page={c['page']}") and c["page"] >= 1


def test_the_stored_answers_match_the_curated_ones_and_the_current_documents():
    stored = prepared.available()
    assert [p.id for p in stored] == [q["id"] for q in QUESTIONS], "run scripts/build_prepared_answers.py"
    assert [p.answer.model_dump(mode="json") for p in stored] == [e["answer"] for e in curation.build_entries()]


def test_each_question_has_a_ready_community_search_in_everyday_words():
    for q in QUESTIONS:
        words = q["community_query"].split()
        assert 3 <= len(words) <= 15, q["id"]
        assert not check_pii(q["community_query"]).blocked


def test_no_curated_text_has_em_or_en_dashes():
    text = curation.CURATED_PATH.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text


def test_an_answer_says_what_it_means_in_plain_words_and_lists_a_confidence_reason():
    for a in CURATED:
        assert 20 <= len(a["summary"].split()) <= 90, a["id"]
        assert a["confidence"] in ("high", "medium", "low") and a["confidence_reason"], a["id"]


# --- the verifier itself --------------------------------------------------------------
def test_a_real_quote_passes_even_with_different_spacing_quote_marks_and_dashes():
    curation.verify_quote("it_guide_fsi_tr_fa_2026", 2,
                          'Schedule FA need not be completed if the taxpayer is classified as "not ordinarily resident" or a "non-resident."')
    curation.verify_quote("rbi_faq_accounts_nonresidents", 3, "Balances in an NRO account of NRIs/ PIOs are remittable up to USD 1 (one) million")


def test_an_invented_quote_is_rejected():
    with pytest.raises(curation.QuoteNotFound):
        curation.verify_quote("rbi_faq_accounts_nonresidents", 3, "NRO balances may be sent abroad without any limit")


def test_a_real_quote_on_the_wrong_page_is_rejected():
    with pytest.raises(curation.QuoteNotFound):
        curation.verify_quote("rbi_faq_accounts_nonresidents", 1, "Balances in an NRO account of NRIs/ PIOs are remittable up to USD 1")


def test_a_page_that_does_not_exist_is_rejected():
    with pytest.raises(curation.QuoteNotFound, match="no page"):
        curation.verify_quote("rbi_faq_accounts_nonresidents", 99, "anything")


def test_a_question_without_a_curated_answer_stops_the_build(tmp_path):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"id": "unanswered", "question": "Q?", "community_query": "a b c"}]), encoding="utf-8")
    with pytest.raises(KeyError, match="unanswered"):
        curation.build_entries(questions_path=questions)


def test_a_review_reason_sets_the_flag_and_builds_a_brief():
    entry = next(a for a in CURATED if a.get("review_reasons"))
    answer = curation.build_answer(entry, curation._manifest())
    assert answer.needs_human_review and answer.review_reasons and answer.ca_brief
