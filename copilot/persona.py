"""Persona: from the guided intake form, or extracted from a free-text question.
Also plans the retrieval queries, conditioned on the persona."""
from datetime import date

import config
from copilot import llm
from copilot.prompts import PERSONA_EXTRACTION
from copilot.schemas import DaysInIndia, Persona
from guardrails.pii import check_pii

# --- Guided intake: option labels shown to the user -> schema values ---------
RELOCATION_OPTIONS = {
    "I plan to relocate": "planning",
    "I relocated recently (under a year ago)": "relocated_recently",
    "I relocated over a year ago": "relocated_over_a_year_ago",
    "Not decided yet": "undecided",
}
CITIZENSHIP_OPTIONS = {
    "Indian citizen": "indian_citizen",
    "Overseas Citizen of India (OCI)": "oci",
    "Person of Indian Origin (PIO)": "pio",
    "Foreign citizen, no OCI or PIO card": "other",
}
ACCOUNT_OPTIONS = {"NRE": "NRE", "NRO": "NRO", "Resident savings": "resident_savings", "None": "none"}
ASSET_OPTIONS = {
    "US stocks": "us_stocks",
    "Vested RSUs or ESOPs": "vested_rsus_esops",
    "Mutual funds": "mutual_funds",
    "401(k) or similar retirement account": "foreign_retirement_account",
    "Bank accounts abroad": "foreign_bank_account",
    "Property": "property",
    "None": "none",
}
GOAL_OPTIONS = {
    "Move assets to India": "move_assets_to_india",
    "Keep assets abroad": "keep_abroad",
    "Just understand the rules": "understand_reporting",
}


# Questions that offer a "write your own" box: answer key -> how it reads in the combined text.
OTHER_FIELDS = {
    "relocation_other": "Move",
    "citizenship_other": "Citizenship",
    "accounts_other": "Indian accounts",
    "assets_other": "Assets abroad",
    "goals_other": "Goals",
}


def persona_from_intake(answers: dict) -> Persona:
    """answers: relocation (label), move_month/move_year (ints or None), citizenship (label),
    countries (list[str]), days (dict year -> days), accounts, assets and goals (labels),
    concern (str), and optional free-text answers under the OTHER_FIELDS keys."""
    move_date = None
    if answers.get("move_year") and answers.get("move_month"):
        move_date = date(int(answers["move_year"]), int(answers["move_month"]), 1)
    other = "; ".join(f"{label}: {answers[key].strip()}" for key, label in OTHER_FIELDS.items()
                      if answers.get(key, "").strip())
    return Persona(
        relocation_status=RELOCATION_OPTIONS.get(answers.get("relocation")),
        move_date=move_date,
        citizenship=CITIZENSHIP_OPTIONS.get(answers.get("citizenship")),
        countries_lived=[c.strip() for c in answers.get("countries", []) if c.strip()],
        days_in_india_by_year=[DaysInIndia(year=int(y), days=int(d))
                               for y, d in (answers.get("days") or {}).items()],
        indian_accounts=[ACCOUNT_OPTIONS[a] for a in answers.get("accounts", [])],
        foreign_assets=[ASSET_OPTIONS[a] for a in answers.get("assets", [])],
        goals=[GOAL_OPTIONS[g] for g in answers.get("goals", [])],
        concerns=answers.get("concern", "").strip(),
        other_context=other,
    )


def enrich_persona(persona: Persona) -> Persona:
    """Understand what was written in the free-text boxes: a model maps it onto the same
    fields as the answer options, and the result is merged in. Nothing already chosen is
    overwritten. If the model is unavailable the text is still kept in `other_context`, so
    it still shapes the search and the answer."""
    if not persona.other_context:
        return persona
    try:
        found = llm.structured_call(Persona, PERSONA_EXTRACTION, persona.other_context, fast=True)
    except Exception:
        return persona
    merged = persona.model_copy(deep=True)
    for name in ("indian_accounts", "foreign_assets", "goals", "countries_lived"):
        values = list(getattr(persona, name))
        values += [v for v in getattr(found, name) if v not in values]
        setattr(merged, name, values)
    for name in ("relocation_status", "move_date", "citizenship"):
        if getattr(merged, name) is None:
            setattr(merged, name, getattr(found, name))
    return merged


def extract_persona(text: str) -> Persona:
    """Direct mode. The PII guard has already run on `text`. Unknown fields stay empty."""
    persona = llm.structured_call(Persona, PERSONA_EXTRACTION, text, fast=True)
    if check_pii(persona.concerns).blocked:  # never carry identifiers forward
        persona.concerns = ""
    return persona


# --- Query planning ------------------------------------------------------------
# Search hints per persona field. These are retrieval keywords, not legal facts:
# what the documents say is only ever read from the retrieved text.
_ACCOUNT_HINTS = {
    "NRE": "NRE account change of residential status on return to India",
    "NRO": "NRO account re-designated resident rupee account on return",
    "resident_savings": "resident savings account non-resident returning to India",
}
_ASSET_HINTS = {
    "us_stocks": "foreign securities held by person resident in India acquired while non-resident",
    "vested_rsus_esops": "employee stock options shares of foreign company resident in India",
    "mutual_funds": "foreign mutual funds held by resident remittance",
    "foreign_retirement_account": "foreign pension retirement account held by returning resident",
    "foreign_bank_account": "foreign currency bank account held by resident individual",
    "property": "immovable property outside India held by resident",
}
_CITIZENSHIP_HINTS = {
    "oci": "Overseas Citizen of India cardholder accounts and investments in India",
    "pio": "Person of Indian Origin accounts and investments in India",
}
_GOAL_HINTS = {
    "understand_reporting": "Schedule FA reporting foreign assets in income tax return",
    "move_assets_to_india": "repatriation of foreign assets to India by returning NRI",
    "keep_abroad": "foreign currency account and assets held abroad by resident individual",
}


def persona_hints(persona: Persona) -> list[str]:
    """Search phrases for what the persona says. Empty when the persona says nothing."""
    hints: list[str] = []
    if persona.days_in_india_by_year or persona.relocation_status in ("planning", "relocated_recently"):
        hints.append("residential status change on return to India days of stay")
    if persona.citizenship in _CITIZENSHIP_HINTS:
        hints.append(_CITIZENSHIP_HINTS[persona.citizenship])
    hints += [_ACCOUNT_HINTS[a] for a in persona.indian_accounts if a in _ACCOUNT_HINTS]
    hints += [_ASSET_HINTS[a] for a in persona.foreign_assets if a in _ASSET_HINTS]
    hints += [_GOAL_HINTS[g] for g in persona.goals if g in _GOAL_HINTS]
    return hints


def plan_queries(persona: Persona, question: str) -> list[str]:
    """2 to 4 focused retrieval queries. Always includes the user's own question;
    the rest depend on what the persona says."""
    queries = [question.strip()]
    if persona.other_context:
        queries.append(persona.other_context[:config.OTHER_CONTEXT_QUERY_CHARS])
    for h in persona_hints(persona):
        if h not in queries:
            queries.append(h)
    if len(queries) < config.MIN_QUERIES:
        queries.append("rules for NRIs returning to India resident status accounts foreign assets")
    return queries[: config.MAX_QUERIES]
