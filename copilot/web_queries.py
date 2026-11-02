"""Write the web search queries from what the user meant, not only what they typed.

People describe their situation in everyday words, and the official sites use formal
terms. One model call turns the question and the persona into a community query and
an official-site query. If that call fails, or returns anything that looks like a
personal identifier, fixed hints built from the persona are used instead.
"""
from dataclasses import dataclass

from copilot import llm
from copilot.persona import persona_hints
from copilot.prompts import SEARCH_QUERY_WRITER
from copilot.schemas import Persona, SearchPlan
from copilot.tools import scrub
from guardrails.pii import check_pii


@dataclass
class WebQueries:
    community: str
    official: str


def fallback_queries(question: str, persona: Persona) -> WebQueries:
    """The question plus the first persona-based hint, so a generic question still searches
    for the user's actual situation."""
    query = " ".join([question.strip()] + persona_hints(persona)[:1])
    return WebQueries(community=query, official=query)


def write_web_queries(question: str, persona: Persona) -> WebQueries:
    fallback = fallback_queries(question, persona)
    try:
        plan = llm.structured_call(SearchPlan, SEARCH_QUERY_WRITER,
                                   f"Question: {question}\nContext: {persona.model_dump_json()}")
    except Exception:
        return fallback
    community, official = scrub(plan.community_query).strip(), scrub(plan.official_query).strip()
    if not community or not official or check_pii(f"{community} {official}").blocked:
        return fallback
    return WebQueries(community=community, official=official)
