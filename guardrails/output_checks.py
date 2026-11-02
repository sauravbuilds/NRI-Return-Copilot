"""Checks on the finished Answer. Each returns a list of problems ([] = pass)."""
import re

import config
from copilot.schemas import DISCLAIMER

# Language that recommends buying, selling or investing.
ADVICE_PATTERNS = [
    r"\byou should (?:buy|sell|invest|purchase|put your money)",
    r"\bi (?:recommend|suggest|advise)\b[^.]{0,40}\b(?:buy|sell|invest|purchas)",
    r"\b(?:we|i) recommend (?:buying|selling|investing)",
    r"\b(?:best|better|good) (?:stock|fund|etf|investment)s? to (?:buy|pick)",
    r"\bconsider (?:buying|selling|investing)",
]
_HANDLE = re.compile(r"(?<![\w.])(?:u|r)/[\w-]+|(?<![\w.])@[\w.]{2,}")


def _texts(answer) -> list[str]:
    t = [answer.summary]
    t += [f.statement for f in answer.official_findings]
    t += [s.step for s in answer.roadmap]
    t += [m.if_this_then_that for m in answer.missing_information]
    return t


def check_citations(answer, retrieved_chunks) -> list[str]:
    """Every finding/step is cited and every cited (doc, page) was really retrieved."""
    allowed = {(c.metadata["doc_id"], c.metadata["page"]) for c in retrieved_chunks}
    problems = []
    items = [("finding", f.statement, f.citations) for f in answer.official_findings]
    items += [("roadmap step", s.step, s.citations) for s in answer.roadmap]
    for kind, text, cites in items:
        if not cites:
            problems.append(f"uncited {kind}: {text[:50]}")
        for c in cites:
            if (c.doc_id, c.page) not in allowed:
                problems.append(f"citation not in retrieved set: {c.doc_id} p.{c.page}")
    return problems


def check_disclaimer(answer) -> list[str]:
    return [] if answer.disclaimer == DISCLAIMER else ["disclaimer missing or altered"]


def check_no_advice_language(answer) -> list[str]:
    problems = []
    for text in _texts(answer):
        for p in ADVICE_PATTERNS:
            if re.search(p, text, re.IGNORECASE):
                problems.append(f"advice language: {text[:50]}")
                break
    return problems


def check_no_usernames(answer) -> list[str]:
    texts = _texts(answer) + [c.snippet + " " + c.title + " " + c.reason for c in answer.community_findings]
    return [f"username/handle found: {t[:40]}" for t in texts if _HANDLE.search(t)]


def check_missing_information(persona, answer) -> list[str]:
    """Every empty persona field the answer depends on must be listed."""
    listed = {m.field for m in answer.missing_information}
    problems = []
    for field in config.REQUIRED_PERSONA_FIELDS:
        if not getattr(persona, field) and field not in listed:
            problems.append(f"missing information not listed: {field}")
    return problems


def run_output_checks(persona, answer, retrieved_chunks) -> list[str]:
    return (check_citations(answer, retrieved_chunks) + check_disclaimer(answer)
            + check_no_advice_language(answer) + check_no_usernames(answer)
            + check_missing_information(persona, answer))
