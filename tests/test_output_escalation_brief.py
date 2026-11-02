import pytest

from copilot.brief import build_brief
from copilot.retrieval import Chunk
from copilot.schemas import (Answer, Citation, CommunityFinding, Finding, MissingItem,
                             Persona, RoadmapStep)
from guardrails import escalation as esc
from guardrails import output_checks as oc
from guardrails.pii import check_pii


def cite(doc="d1", page=2):
    return Citation(document_title="Doc One", issuer="RBI", page=page, doc_id=doc,
                    page_url=f"https://x.org/{doc}.pdf#page={page}", last_verified="2026-11-01")


CHUNKS = [Chunk("t", {"doc_id": "d1", "page": 2}, 0.9)]


def answer(**kw):
    base = dict(summary="Info only.", official_findings=[Finding(statement="s", citations=[cite()])],
                roadmap=[RoadmapStep(stage="first_year", step="do x", citations=[cite()])])
    base.update(kw)
    return Answer(**base)


# --- output checks: one test per check -------------------------------------
def test_citations_pass_when_all_retrieved():
    assert oc.check_citations(answer(), CHUNKS) == []


def test_citation_not_in_retrieved_set_rejected():
    a = answer(roadmap=[RoadmapStep(stage="first_year", step="x", citations=[cite("d1", 9)])])
    assert any("not in retrieved" in p for p in oc.check_citations(a, CHUNKS))


def test_uncited_claim_cannot_even_be_built():
    with pytest.raises(Exception):
        Finding(statement="s", citations=[])


def test_disclaimer_must_be_present_and_unaltered():
    assert oc.check_disclaimer(answer()) == []
    assert oc.check_disclaimer(answer(disclaimer="")) != []


@pytest.mark.parametrize("text", ["You should buy index funds.", "I recommend investing in US stocks.",
                                  "Consider buying more shares."])
def test_advice_language_rejected(text):
    assert oc.check_no_advice_language(answer(summary=text)) != []


def test_rules_language_about_investments_is_allowed():
    assert oc.check_no_advice_language(answer(summary="Moving shares needs paperwork described on page 2.")) == []


def test_usernames_rejected():
    cf = CommunityFinding(url="u", title="t", snippet="posted by u/someone", verdict="unrelated", reason="r")
    assert oc.check_no_usernames(answer(community_findings=[cf])) != []


def test_missing_information_must_list_every_empty_required_field():
    p = Persona(relocation_status="planning")
    problems = oc.check_missing_information(p, answer())
    assert any("move_date" in x for x in problems) and not any("relocation_status" in x for x in problems)
    listed = [MissingItem(field=f, why_it_matters="w", if_this_then_that="i")
              for f in ["move_date", "indian_accounts", "foreign_assets", "days_in_india_by_year"]]
    assert oc.check_missing_information(p, answer(missing_information=listed)) == []


# --- escalation: one test per rule ------------------------------------------
def test_rule_us_side_tax():
    assert "US-side" in esc.escalation_reasons(Persona(), scope_label="us_side_tax")[0]


def test_rule_complex_assets_three_or_more():
    three = Persona(foreign_assets=["us_stocks", "vested_rsus_esops", "property"])
    two = Persona(foreign_assets=["us_stocks", "property"])
    assert esc.has_complex_assets(three) and not esc.has_complex_assets(two)


def test_rule_low_confidence():
    assert esc.escalation_reasons(Persona(), confidence="low")
    assert not esc.escalation_reasons(Persona(), confidence="high")


def test_rule_community_contradicts():
    bad = CommunityFinding(url="u", title="t", snippet="s", verdict="contradicts", reason="r")
    ok = CommunityFinding(url="u", title="t", snippet="s", verdict="agrees", reason="r")
    assert esc.community_contradicts([bad]) and not esc.community_contradicts([ok])


# --- CA brief ----------------------------------------------------------------
def test_brief_has_no_pii_and_lists_every_escalation_reason():
    p = Persona(relocation_status="planning", foreign_assets=["us_stocks", "vested_rsus_esops", "property"],
                concerns="my number is 9876543210")  # must never be copied into the brief
    reasons = esc.escalation_reasons(p, confidence="low")
    a = answer(review_reasons=reasons, needs_human_review=True)
    text = build_brief(p, a)
    assert not check_pii(text).blocked and "9876543210" not in text
    assert all(r in text for r in reasons)
    assert "Doc One" in text and "#page=2" in text


def test_brief_is_produced_on_abstention():
    a = Answer(summary="Could not find official text.", abstained=True, needs_human_review=True,
               review_reasons=["No grounded answer"])
    text = build_brief(Persona(), a)
    assert "No grounded answer" in text and "no official text" in text.lower()
