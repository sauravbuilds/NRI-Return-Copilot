"""Scope guard: in scope / investment advice / US-side tax / out of scope.

The classification is done by the LLM (structured output, temperature 0).
What happens for each label is plain code, so it can be tested without a model.
"""
import csv

import config
from copilot import llm
from copilot.prompts import SCOPE_CLASSIFIER
from copilot.schemas import ScopeResult

CAPABILITIES = ("Here's what I can help with: which Indian rules and paperwork apply when you move back, "
                "like residential status, NRE and NRO accounts, holding foreign assets, and reporting "
                "them in your Indian tax return. I answer from official documents.")

REFUSAL_ADVICE = ("I can't tell you what to buy, sell or invest in. What I can do is tell you which rules "
                  "and paperwork apply, like what you'd need to do to move shares to India. Want to ask "
                  "it that way? " + CAPABILITIES)
REFUSAL_OUT_OF_SCOPE = "That one's outside what I can help with. " + CAPABILITIES
US_TAX_MESSAGE = ("That one is about US tax, so I can't answer it. The official sources are linked below, "
                  "and a qualified US tax professional is the right person to ask. Community "
                  "discussions that may help are listed too.")


def classify_scope(question: str) -> ScopeResult:
    return llm.structured_call(ScopeResult, SCOPE_CLASSIFIER, question, fast=True)


def escalation_links(path=None) -> list[dict]:
    with open(path or config.ESCALATION_LINKS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scope_response(label: str) -> dict:
    """What to do for each label. 'proceed' means carry on with the pipeline."""
    if label == "investment_advice":
        return {"action": "refuse", "message": REFUSAL_ADVICE}
    if label == "out_of_scope":
        return {"action": "refuse", "message": REFUSAL_OUT_OF_SCOPE}
    if label == "us_side_tax":
        return {"action": "escalate_us_tax", "message": US_TAX_MESSAGE, "links": escalation_links()}
    return {"action": "proceed", "message": ""}
