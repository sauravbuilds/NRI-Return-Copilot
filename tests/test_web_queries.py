"""The web search queries are written from the user's meaning (fake model, no network)."""
from copilot import llm
from copilot.schemas import Persona, SearchPlan
from copilot.web_queries import fallback_queries, write_web_queries

GUIDED = Persona(indian_accounts=["NRO"], foreign_assets=["foreign_retirement_account"])


def plan(community="moved back from canada what to do with my old account",
         official="NRO account re-designation resident rupee account returning NRI RBI"):
    return lambda schema, system, user: SearchPlan(community_query=community, official_query=official)


def test_each_query_uses_the_words_of_its_own_kind_of_site(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", plan())
    q = write_web_queries("I just got back from Toronto, what happens to my bank account here?", Persona())
    assert "moved back" in q.community and "RBI" in q.official and q.community != q.official


def test_the_question_and_context_both_reach_the_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "structured_call", lambda s, sys, user: seen.update(user=user) or SearchPlan(
        community_query="a", official_query="b"))
    write_web_queries("what about my account?", GUIDED)
    assert "what about my account?" in seen["user"] and "NRO" in seen["user"]


def test_fallback_keeps_the_question_and_adds_the_persona(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(llm, "structured_call", boom)
    q = write_web_queries("Which rules apply to my situation?", GUIDED)
    assert q.community == q.official and q.community.startswith("Which rules apply to my situation?")
    assert "NRO" in q.community  # a generic question still searches for the real situation


def test_fallback_for_an_empty_persona_is_just_the_question():
    assert fallback_queries("what about my account?", Persona()).official == "what about my account?"


def test_queries_that_contain_identifiers_are_replaced_by_the_fallback(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", plan(community="my pan ABCDE1234F nro account"))
    q = write_web_queries("what about my account?", GUIDED)
    assert "ABCDE1234F" not in q.community and q.community.startswith("what about my account?")


def test_empty_model_output_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", plan(community="  "))
    assert write_web_queries("what about my account?", GUIDED).community.startswith("what about my account?")
