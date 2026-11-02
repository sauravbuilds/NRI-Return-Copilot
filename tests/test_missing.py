"""The 'what I could not determine' list is completed by code when the model leaves a fact out."""
from datetime import date

import config
from copilot.missing import FIELD_TEXT, complete_missing_information
from copilot.schemas import MissingItem, Persona
from guardrails.output_checks import check_missing_information


def item(field, why="model's reason", then="model's note"):
    return MissingItem(field=field, why_it_matters=why, if_this_then_that=then)


def test_every_required_fact_has_wording():
    assert set(FIELD_TEXT) == set(config.REQUIRED_PERSONA_FIELDS)


def test_an_empty_profile_gets_an_entry_for_every_required_fact():
    out = complete_missing_information(Persona(), [])
    assert {m.field for m in out} == set(config.REQUIRED_PERSONA_FIELDS)


def test_what_the_model_wrote_is_kept_and_not_duplicated():
    out = complete_missing_information(Persona(), [item("move_date")])
    dates = [m for m in out if m.field == "move_date"]
    assert len(dates) == 1 and dates[0].why_it_matters == "model's reason"


def test_field_names_written_loosely_by_the_model_still_count():
    out = complete_missing_information(Persona(), [item("Move date"), item("days-in-india-by-year")])
    assert sorted(m.field for m in out).count("move_date") == 1
    assert sorted(m.field for m in out).count("days_in_india_by_year") == 1


def test_facts_the_user_gave_are_not_listed_as_missing():
    persona = Persona(move_date=date(2026, 1, 1), indian_accounts=["NRO"], relocation_status="planning")
    assert {m.field for m in complete_missing_information(persona, [])} == {"foreign_assets", "days_in_india_by_year"}


def test_the_completed_list_always_passes_the_output_check():
    from copilot.schemas import Answer
    answer = Answer(summary="s", missing_information=complete_missing_information(Persona(), []))
    assert check_missing_information(Persona(), answer) == []


def test_the_added_wording_states_no_rule_and_has_no_dashes():
    for why, then in FIELD_TEXT.values():
        assert "—" not in why + then and "–" not in why + then
