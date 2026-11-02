"""The "what I could not determine" list, completed by code.

The model is asked to list every fact it was not given and how it would change the answer.
Models often leave one out, and a missing entry used to make the whole answer fail its checks.
Code now adds a plain entry for any fact that is still unlisted. The wording only asks for the
fact and says what it would be used for; it states no rule.
"""
import config
from copilot.schemas import MissingItem, Persona

FIELD_TEXT = {
    "relocation_status": (
        "The steps depend on whether you have already moved or are still planning to.",
        "If you have moved, the roadmap would focus on what to do now. If you are still planning, it "
        "would focus on what to do before you go."),
    "move_date": (
        "Many rules start counting from the date you become a resident.",
        "With your move date, the roadmap could say which documents and deadlines to look at first."),
    "indian_accounts": (
        "Different kinds of accounts are treated differently when you return.",
        "If you hold NRE, NRO or resident accounts, the steps for each would be listed separately."),
    "foreign_assets": (
        "Different foreign assets come with different reporting and transfer rules.",
        "If you hold shares, funds, bank accounts or retirement accounts abroad, each would be covered "
        "on its own."),
    "days_in_india_by_year": (
        "Residential status is worked out from the days you spend in India.",
        "With your day counts per year, the roadmap could point to the status the documents describe "
        "for you."),
}


def _key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def complete_missing_information(persona: Persona, listed: list[MissingItem]) -> list[MissingItem]:
    """Keep what the model wrote, and add an entry for each required fact that is empty in the
    profile and still unlisted."""
    out = [m.model_copy(update={"field": _key(m.field)}) for m in listed]
    present = {m.field for m in out}
    for field in config.REQUIRED_PERSONA_FIELDS:
        if not getattr(persona, field) and field not in present:
            why, then = FIELD_TEXT[field]
            out.append(MissingItem(field=field, why_it_matters=why, if_this_then_that=then))
    return out
