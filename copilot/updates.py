"""Check for official publications newer than our last document check.

The document refresh only re-checks files already in the manifest and runs every few
days, so a new circular can appear in between. This module searches the official
sites only, and only for items published after the last check, then compares each
item with the passages an answer is based on.
"""
import csv
from datetime import date

import config
from copilot import llm
from copilot.compare import format_passages
from copilot.prompts import UPDATE_COMPARISON
from copilot.schemas import OfficialUpdate, UpdateVerdicts
from copilot.text import plain
from copilot.tools import scrub, site_of


class SearchUnavailable(RuntimeError):
    """No search key is configured."""


def _manifest_rows(manifest_path=None) -> list[dict]:
    with open(manifest_path or config.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last_checked(manifest_path=None) -> date | None:
    """The most recent day any document in the manifest was checked or downloaded."""
    days = []
    for row in _manifest_rows(manifest_path):
        for key in ("last_verified", "date_downloaded"):
            if row.get(key):
                days.append(date.fromisoformat(row[key]))
    return max(days) if days else None


def fetch_official_updates(query: str, since: date, client=None, manifest_path=None,
                           max_results: int | None = None) -> list[dict]:
    """Official-site pages published on or after `since` that match the query.
    Pages that are already in the manifest are left out."""
    max_results = max_results or config.OFFICIAL_UPDATE_MAX_RESULTS
    if client is None:
        if not config.TAVILY_API_KEY:
            raise SearchUnavailable("no search key configured")
        from langchain_tavily import TavilySearch
        client = TavilySearch(max_results=max_results, tavily_api_key=config.TAVILY_API_KEY)
    raw = client.invoke({"query": query, "include_domains": config.OFFICIAL_DOMAINS,
                         "start_date": since.isoformat()})
    known = {row["url"] for row in _manifest_rows(manifest_path)}
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return [{"url": r.get("url", ""), "title": plain(scrub(r.get("title", ""))),
             "snippet": plain(scrub(r.get("content", ""))), "date": r.get("published_date", "") or "",
             "site": site_of(r.get("url", ""))}
            for r in results if r.get("url") and r["url"] not in known][:max_results]


UNCHECKED_REASON = "I couldn't compare this one with the documents, so give it a quick read yourself."


def compare_updates(updates: list[dict], chunks) -> list[OfficialUpdate]:
    """What each newer publication means for the passages in use, judged in one model call.
    A failed or missing comparison is kept as 'unchecked' so a newer circular is never
    silently dropped."""
    if not updates:
        return []
    numbered = "\n\n".join(f"[{i}] {u['title']}\n{u['snippet']}" for i, u in enumerate(updates, start=1))
    user = f"OUR PASSAGES:\n{format_passages(chunks)}\n\nNEWER PUBLICATIONS:\n{numbered}"
    try:
        found = llm.structured_call(UpdateVerdicts, UPDATE_COMPARISON, user, fast=True).verdicts
    except Exception:
        found = []
    by_number = {v.number: v for v in found}
    out = []
    for i, u in enumerate(updates, start=1):
        v = by_number.get(i)
        out.append(OfficialUpdate(url=u["url"], title=u["title"], snippet=u["snippet"], date=u["date"],
                                  verdict=v.label if v else "unchecked",
                                  reason=plain(v.reason) if v else UNCHECKED_REASON))
    return out
