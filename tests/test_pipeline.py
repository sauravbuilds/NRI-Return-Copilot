"""Pipeline tests with a fake LLM (no keys, no network)."""
import pytest

import config
from copilot import llm, orchestrator as orch
from copilot.persona import persona_from_intake, plan_queries
from copilot.retrieval import Chunk
from copilot.schemas import (AnswerDraft, DraftFinding, DraftStep, MissingItem, NumberedUpdateVerdict,
                             NumberedVerdict, Persona, ScopeResult, SearchPlan, ThreadVerdicts,
                             UpdateVerdicts)
from copilot.tools import fetch_community
from guardrails import advice

META = {"doc_id": "d1", "title": "Doc One", "issuer": "RBI", "page": 2,
        "page_url": "https://x.org/d1.pdf#page=2", "last_verified": "2026-11-01"}
GOOD = [Chunk("NRO accounts may be re-designated.", dict(META), 0.9)]
ALL_MISSING = [MissingItem(field=f, why_it_matters="w", if_this_then_that="if x then y")
               for f in ["relocation_status", "move_date", "indian_accounts", "foreign_assets", "days_in_india_by_year"]]


class Fake:
    """Replaces llm.structured_call / tool_call_step. Counts calls."""
    def __init__(self, scope="in_scope", draft=None, verdict="agrees", update_verdict="changes", us_side=False):
        self.scope, self.verdict, self.update_verdict, self.us_side, self.calls = (
            scope, verdict, update_verdict, us_side, 0)
        self.draft = draft or AnswerDraft(
            summary="Info.", findings=[DraftFinding(statement="s", source_ids=[1])],
            roadmap=[DraftStep(stage="first_30_days", step="do x", source_ids=[1])],
            missing_information=ALL_MISSING, confidence="high")

    def structured(self, schema, system, user, **kwargs):
        self.calls += 1
        if schema is ScopeResult:
            return ScopeResult(label=self.scope, has_us_side=self.us_side)
        if schema is AnswerDraft:
            return self.draft
        if schema is ThreadVerdicts:
            return ThreadVerdicts(verdicts=[NumberedVerdict(number=i, label=self.verdict, reason="r")
                                            for i in range(1, 6)])
        if schema is UpdateVerdicts:
            return UpdateVerdicts(verdicts=[NumberedUpdateVerdict(number=i, label=self.update_verdict,
                                                                  reason="newer rule") for i in range(1, 6)])
        if schema is SearchPlan:
            return SearchPlan(community_query="moved back what to do with old bank account",
                              official_query="NRO account re-designation returning NRI circular")
        if schema is Persona:
            return Persona()
        raise AssertionError(schema)

    def tools(self, system, user, tools, **kwargs):
        self.calls += 1
        class Msg:
            tool_calls = [{"name": "search_community", "args": {"query": "nro u/someone"}},
                          {"name": "search_official_docs", "args": {"query": "extra"}}]
        return Msg()


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(llm, "structured_call", f.structured)
    monkeypatch.setattr(llm, "tool_call_step", f.tools)
    monkeypatch.setattr(orch, "retrieve", lambda q, k=5: list(GOOD))
    monkeypatch.setattr(orch, "fetch_community", lambda q: [
        {"url": "http://f.com/1", "title": "t", "snippet": "s", "date": "", "site": "f.com"}])
    monkeypatch.setattr(orch, "fetch_official_updates", lambda q, since: [])
    return f


def test_no_model_is_called_when_pii_guard_blocks(fake):
    r = orch.run_pipeline("my PAN is ABCDE1234F, what happens to my NRO account?")
    assert r.kind == "pii_blocked" and fake.calls == 0 and "ABCDE1234F" not in r.message


def test_pii_in_guided_free_text_also_blocks(fake):
    r = orch.run_pipeline("What applies?", Persona(concerns="call 9876543210"))
    assert r.kind == "pii_blocked" and fake.calls == 0


@pytest.mark.parametrize("label", ["investment_advice", "out_of_scope"])
def test_refusals_stop_before_retrieval(fake, monkeypatch, label):
    fake.scope = label
    monkeypatch.setattr(orch, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieved")))
    r = orch.run_pipeline("Which stock should I buy?")
    assert r.kind == "refused" and "official documents" in r.message  # says what the tool can do


def test_us_tax_escalates_with_irs_link_and_no_answer(fake):
    fake.scope = "us_side_tax"
    r = orch.run_pipeline("Will the IRS tax my 401(k)?")
    assert r.kind == "us_side_tax" and r.answer.needs_human_review and r.answer.ca_brief
    assert any("irs.gov" in l["url"] for l in r.links)


def test_a_question_with_a_us_side_is_still_answered_for_the_indian_side(fake):
    fake.us_side = True
    r = orch.run_pipeline("How do I move my 401(k) to India?")
    assert r.kind == "answered" and r.answer.roadmap
    assert any("US side" in x for x in r.answer.review_reasons) and r.answer.ca_brief
    assert any("irs.gov" in l["url"] for l in r.links)  # where to look for the US part


def test_the_model_is_told_when_there_is_a_us_side(fake, monkeypatch):
    fake.us_side = True
    prompts, original = [], fake.structured
    monkeypatch.setattr(llm, "structured_call", lambda schema, s, u, **k: prompts.append(u) or original(schema, s, u))
    orch.run_pipeline("How do I move my 401(k) to India?")
    assert any("US side: yes" in u for u in prompts) and not any("US side: no" in u for u in prompts)


def test_without_a_us_side_nothing_about_the_us_is_shown(fake):
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.links == [] and not any("US side" in x for x in r.answer.review_reasons)


def test_a_purely_us_question_still_offers_community_discussions(fake):
    fake.scope = "us_side_tax"
    r = orch.run_pipeline("Will the IRS tax my 401(k)?")
    assert r.kind == "us_side_tax" and r.answer.community_links and r.links
    assert r.answer.community_links[0].url == "http://f.com/1"


def test_community_discussions_for_a_us_question_can_be_switched_off(fake, monkeypatch):
    fake.scope = "us_side_tax"
    monkeypatch.setattr(orch, "fetch_community", lambda q: (_ for _ in ()).throw(AssertionError("searched")))
    r = orch.run_pipeline("Will the IRS tax my 401(k)?", community=False)
    assert r.answer.community_links == [] and r.links


def test_when_the_documents_do_not_cover_it_community_discussions_are_still_offered(fake, monkeypatch):
    monkeypatch.setattr(orch, "retrieve", lambda q, k=5: [Chunk("x", dict(META), 0.01)])
    r = orch.run_pipeline("How do I transfer my 401(k) to India?")
    assert r.kind == "abstained" and r.answer.community_links
    assert any(s == "community_search" for s, _ in r.trace)


def test_an_abstention_with_a_us_side_also_points_to_the_official_us_sources(fake, monkeypatch):
    fake.us_side = True
    monkeypatch.setattr(orch, "retrieve", lambda q, k=5: [Chunk("x", dict(META), 0.01)])
    r = orch.run_pipeline("How do I transfer my 401(k) to India?")
    assert r.kind == "abstained" and any("irs.gov" in l["url"] for l in r.links)
    assert any("US side" in x for x in r.answer.review_reasons)


def test_a_failed_community_search_never_blocks_an_abstention(fake, monkeypatch):
    monkeypatch.setattr(orch, "retrieve", lambda q, k=5: [Chunk("x", dict(META), 0.01)])
    monkeypatch.setattr(orch, "fetch_community", lambda q: (_ for _ in ()).throw(RuntimeError("down")))
    r = orch.run_pipeline("What is stamp duty in Mumbai?")
    assert r.kind == "abstained" and r.answer.community_links == []


def test_newer_publications_can_be_left_out_of_a_run(fake, monkeypatch):
    monkeypatch.setattr(orch, "fetch_official_updates", lambda q, since: (_ for _ in ()).throw(AssertionError("searched")))
    assert orch.run_pipeline("What happens to my NRO account?", check_updates=False).kind == "answered"


def test_abstains_when_nothing_relevant(fake, monkeypatch):
    monkeypatch.setattr(orch, "retrieve", lambda q, k=5: [Chunk("x", dict(META), 0.01)])
    r = orch.run_pipeline("What is stamp duty in Mumbai?")
    assert r.kind == "abstained" and r.answer.abstained and r.answer.ca_brief


def test_grounding_is_judged_on_the_question_not_on_added_search_phrases(fake, monkeypatch):
    unrelated = [Chunk("x", dict(META), 0.10)]
    # Search phrases added from the profile match the corpus whatever was asked.
    monkeypatch.setattr(orch, "retrieve",
                        lambda q, k=5: unrelated if q == "capital gains rate in Japan?" else list(GOOD))
    r = orch.run_pipeline("capital gains rate in Japan?")
    assert r.kind == "abstained"


def test_happy_path_answers_with_real_citation_and_community_tier(fake):
    r = orch.run_pipeline("What happens to my NRO account?")
    a = r.answer
    assert r.kind == "answered" and a.roadmap[0].citations[0].page_url.endswith("#page=2")
    assert a.community_findings and a.community_findings[0].verdict == "agrees"
    assert all("http://f.com" not in f.statement for f in a.official_findings)  # tiers separate
    assert any(s == "tool_call" for s, _ in r.trace)


def test_tool_call_trace_scrubs_usernames(fake):
    r = orch.run_pipeline("What happens to my NRO account?")
    detail = next(d for s, d in r.trace if s == "tool_call")
    assert "u/someone" not in detail and "search_community" in detail


def test_contradicting_thread_flags_human_review(fake):
    fake.verdict = "contradicts"
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.answer.needs_human_review and any("contradicts" in x for x in r.answer.review_reasons)
    assert r.answer.ca_brief


NEWER = [{"url": "https://www.rbi.org.in/new", "title": "New circular", "snippet": "limit changed",
          "date": "2026-11-02", "site": "rbi.org.in"}]


def test_newer_official_publication_is_surfaced_and_can_flag_review(fake, monkeypatch):
    seen = {}
    monkeypatch.setattr(orch, "fetch_official_updates", lambda q, since: seen.update(since=since) or NEWER)
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.answer.official_updates[0].verdict == "changes"
    assert any("newer official publication" in x for x in r.answer.review_reasons)
    assert seen["since"] == orch.last_checked()  # search starts at the last document check, not from scratch
    assert any(s == "official_updates" for s, _ in r.trace)


def test_unrelated_newer_publication_is_not_shown(fake, monkeypatch):
    fake.update_verdict = "unrelated"
    monkeypatch.setattr(orch, "fetch_official_updates", lambda q, since: NEWER)
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.answer.official_updates == [] and not r.answer.needs_human_review


def test_update_search_failure_never_blocks_the_answer(fake, monkeypatch):
    def boom(q, since):
        raise RuntimeError("search down")
    monkeypatch.setattr(orch, "fetch_official_updates", boom)
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.kind == "answered" and ("official_updates", "skipped: RuntimeError") in r.trace


def test_dashes_in_model_text_are_removed_from_the_answer(fake):
    fake.draft = AnswerDraft(
        summary="Good news — you can keep it.", missing_information=ALL_MISSING, confidence="high",
        findings=[DraftFinding(statement="Limit is 2020–21 only", source_ids=[1])],
        roadmap=[DraftStep(stage="first_year", step="File it – on time", source_ids=[1])])
    a = orch.run_pipeline("What happens to my NRO account?").answer
    text = " ".join([a.summary, a.official_findings[0].statement, a.roadmap[0].step])
    assert "—" not in text and "–" not in text
    assert "2020-21" in text and a.summary == "Good news, you can keep it."


def test_citation_carries_release_date_and_document_link():
    c = orch.citation_for(Chunk("t", dict(META, date_published="2015-07-01"), 0.9))
    assert c.published == "2015-07-01" and c.document_url == "https://x.org/d1.pdf"


def test_invented_source_number_triggers_regeneration_then_abstain(fake):
    fake.draft = AnswerDraft(summary="x", findings=[DraftFinding(statement="s", source_ids=[99])],
                             missing_information=ALL_MISSING, confidence="high")
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.kind == "abstained"
    assert sum(1 for s, _ in r.trace if s == "output_checks") == 2  # tried once, regenerated once


def test_facts_the_model_forgot_to_list_are_added_by_code_not_failed(fake):
    fake.draft.missing_information = []
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.kind == "answered"
    assert {m.field for m in r.answer.missing_information} == set(config.REQUIRED_PERSONA_FIELDS)


# --- persona + query planning --------------------------------------------------
def test_different_personas_give_different_queries():
    q = "What applies to me?"
    a = plan_queries(Persona(indian_accounts=["NRE"], goals=["understand_reporting"]), q)
    b = plan_queries(Persona(foreign_assets=["foreign_retirement_account"], goals=["keep_abroad"]), q)
    assert a != b and 2 <= len(a) <= 4 and 2 <= len(b) <= 4 and a[0] == b[0] == q


def test_empty_persona_still_gets_two_queries():
    assert len(plan_queries(Persona(), "hello")) >= 2


def test_guided_intake_maps_to_valid_persona():
    p = persona_from_intake({"relocation": "I plan to relocate", "move_month": 6, "move_year": 2027,
                             "countries": ["USA"], "days": {2024: 40}, "accounts": ["NRE"],
                             "assets": ["US stocks", "401(k) or similar retirement account"],
                             "goals": ["Just understand the rules"], "concern": "reporting"})
    assert p.relocation_status == "planning" and p.move_date.year == 2027
    assert p.foreign_assets == ["us_stocks", "foreign_retirement_account"]
    assert Persona.model_validate(p.model_dump()) == p


# --- scope + community ---------------------------------------------------------
def test_scope_responses():
    assert advice.scope_response("investment_advice")["action"] == "refuse"
    assert advice.scope_response("out_of_scope")["action"] == "refuse"
    assert advice.scope_response("us_side_tax")["links"]
    assert advice.scope_response("in_scope")["action"] == "proceed"


def test_community_tool_returns_only_allowed_fields_and_no_usernames():
    class C:
        def invoke(self, _): return {"results": [{"url": "https://www.reddit.com/r/x/1", "title": "Hi u/bob",
                                                   "content": "ask @alice 9876543210123", "score": 0.9,
                                                   "raw_content": "SECRET", "author": "bob"}]}
    out = fetch_community("q", client=C())
    assert set(out[0]) == {"url", "title", "snippet", "date", "site"}
    blob = str(out)
    assert "bob" not in blob and "alice" not in blob and "9876543210123" not in blob and "SECRET" not in blob
    assert out[0]["site"] == "reddit.com"


def test_community_tool_caps_results():
    class C:
        def invoke(self, _): return {"results": [{"url": f"http://a.com/{i}", "title": "t", "content": "c"} for i in range(20)]}
    assert len(fetch_community("q", client=C())) == 5


def test_community_without_key_returns_nothing(monkeypatch):
    import config
    monkeypatch.setattr(config, "TAVILY_API_KEY", None)
    assert fetch_community("q") == []


def test_community_opt_out_skips_community_search(fake, monkeypatch):
    monkeypatch.setattr(orch, "fetch_community", lambda q: (_ for _ in ()).throw(AssertionError("searched")))
    r = orch.run_pipeline("What happens to my NRO account?", community=False)
    assert r.kind == "answered" and r.answer.community_findings == []


def test_all_threads_are_compared_in_one_model_call(fake, monkeypatch):
    monkeypatch.setattr(orch, "fetch_community", lambda q: [
        {"url": f"http://f.com/{i}", "title": "t", "snippet": "s", "date": "", "site": "f.com"} for i in range(4)])
    schemas, original = [], fake.structured
    monkeypatch.setattr(llm, "structured_call", lambda schema, s, u, **k: schemas.append(schema) or original(schema, s, u))
    r = orch.run_pipeline("What happens to my NRO account?")
    assert schemas.count(ThreadVerdicts) == 1 and len(r.answer.community_findings) == 4


def test_abstention_reasons_shown_to_users_carry_no_internal_detail(fake, monkeypatch):
    import re
    monkeypatch.setattr(orch, "retrieve", lambda q, k=5: [Chunk("x", dict(META), 0.01)])
    r = orch.run_pipeline("What is stamp duty in Mumbai?")
    shown = " ".join(r.answer.review_reasons) + r.answer.confidence_reason + r.message
    assert not re.search(r"threshold|relevance|score|safety", shown, re.I)
    assert any(step == "abstained" and "threshold" in detail for step, detail in r.trace)  # kept for the log


def test_a_draft_that_keeps_failing_is_reported_plainly_not_with_internals(fake):
    fake.draft = AnswerDraft(summary="x", findings=[DraftFinding(statement="s", source_ids=[99])],
                             missing_information=ALL_MISSING, confidence="high")
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.answer.review_reasons == ["I couldn't put together a reliable, fully sourced answer this time."]


def test_progress_is_reported_in_plain_language(fake):
    steps = []
    orch.run_pipeline("What happens to my NRO account?", on_step=steps.append)
    assert steps == ["Understanding your question", "Searching the official documents",
                     "Checking community discussions and recent circulars", "Writing your answer"]


def test_web_searches_use_the_written_queries_not_the_raw_question(fake, monkeypatch):
    sent = {}
    monkeypatch.setattr(orch, "fetch_official_updates", lambda q, since: sent.update(official=q) or [])
    monkeypatch.setattr(orch, "fetch_community", lambda q: sent.update(community=q) or [])
    orch.run_pipeline("im back from toronto, what about my old account??")
    assert sent["official"] == "NRO account re-designation returning NRI circular"
    assert sent["community"] == "moved back what to do with old bank account"  # even though the model asked differently


def test_community_is_searched_even_when_the_model_skips_the_tool(fake, monkeypatch):
    class NoCalls:
        tool_calls = []
    monkeypatch.setattr(llm, "tool_call_step", lambda *a, **k: NoCalls())
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.answer.community_findings and any(s == "community_search" for s, _ in r.trace)


def test_extra_official_search_by_the_model_only_adds_validated_passages(fake):
    r = orch.run_pipeline("What happens to my NRO account?")
    assert r.kind == "answered" and len(r.chunks) == 1  # same passage retrieved again is not duplicated
