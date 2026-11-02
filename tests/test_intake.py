"""Guided intake: write-your-own answers, how they are understood, and what the questions ask."""
from copilot import llm, orchestrator as orch
from copilot.persona import (CITIZENSHIP_OPTIONS, OTHER_FIELDS, enrich_persona, persona_from_intake,
                             plan_queries)
from copilot.retrieval import Chunk
from copilot.schemas import Persona

ANSWERS = {"relocation": "I relocated recently (under a year ago)", "citizenship": "Overseas Citizen of India (OCI)",
           "accounts": ["NRO"], "assets": ["Bank accounts abroad"], "goals": [],
           "assets_other": "RRSP and a TFSA in Canada", "goals_other": "bring the money over slowly"}


def test_written_answers_are_kept_in_the_users_words():
    p = persona_from_intake(ANSWERS)
    assert "RRSP and a TFSA in Canada" in p.other_context and "bring the money over slowly" in p.other_context
    assert p.citizenship == "oci" and p.foreign_assets == ["foreign_bank_account"]


def test_an_unanswered_question_stays_empty():
    p = persona_from_intake({"relocation": None, "move_month": None, "move_year": None, "citizenship": None})
    assert p.relocation_status is None and p.move_date is None and p.citizenship is None
    assert p.other_context == ""


def test_every_write_in_box_has_a_label_in_the_combined_text():
    assert set(OTHER_FIELDS) == {"relocation_other", "citizenship_other", "accounts_other", "assets_other",
                                 "goals_other"}


def test_a_model_maps_written_answers_onto_the_option_fields(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", lambda *a, **k: Persona(
        foreign_assets=["foreign_retirement_account", "foreign_bank_account"], countries_lived=["Canada"]))
    merged = enrich_persona(persona_from_intake(ANSWERS))
    assert merged.foreign_assets == ["foreign_bank_account", "foreign_retirement_account"]  # chosen ones stay first
    assert merged.countries_lived == ["Canada"] and merged.citizenship == "oci"


def test_enrichment_never_overwrites_what_was_chosen(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", lambda *a, **k: Persona(citizenship="pio", relocation_status="planning"))
    merged = enrich_persona(persona_from_intake(ANSWERS))
    assert merged.citizenship == "oci" and merged.relocation_status == "relocated_recently"


def test_if_the_model_is_unavailable_the_written_text_is_still_kept(monkeypatch):
    def down(*a, **k):
        raise RuntimeError("no provider")
    monkeypatch.setattr(llm, "structured_call", down)
    merged = enrich_persona(persona_from_intake(ANSWERS))
    assert "RRSP" in merged.other_context and merged.foreign_assets == ["foreign_bank_account"]


def test_nothing_is_sent_to_the_model_when_there_is_no_written_text(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert enrich_persona(persona_from_intake({"accounts": ["NRE"]})).indian_accounts == ["NRE"]


def test_written_text_and_citizenship_shape_the_search():
    queries = plan_queries(persona_from_intake(ANSWERS), "What applies to me?")
    assert any("RRSP" in q for q in queries) and len(queries) <= 4
    assert any("Overseas Citizen of India" in q for q in plan_queries(Persona(citizenship="oci", other_context=""), "q"))


def test_identifiers_in_a_written_answer_block_the_request_before_any_model_call(monkeypatch):
    monkeypatch.setattr(llm, "structured_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    monkeypatch.setattr(orch, "retrieve", lambda *a, **k: [Chunk("x", {}, 0.9)])
    persona = persona_from_intake({"assets_other": "my account is 123456789012"})
    assert orch.run_pipeline("What applies to me?", persona).kind == "pii_blocked"


def test_citizenship_options_cover_the_main_cases():
    assert set(CITIZENSHIP_OPTIONS.values()) == {"indian_citizen", "oci", "pio", "other"}
