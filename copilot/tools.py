"""The three LLM tools. The model reads each docstring to decide when to call it.

The module also holds the pure helpers behind them: the day counter and the
community-text scrubber.
"""
import re
from datetime import date, timedelta

from langchain_core.tools import tool

import config
from copilot.retrieval import retrieve
from copilot.text import plain


# --- days in India ----------------------------------------------------------
def count_days_by_year(date_ranges: list[tuple[date, date]]) -> dict[int, int]:
    """Days per calendar year. Both start and end dates are COUNTED (inclusive).
    Overlapping ranges are merged so a day is never counted twice."""
    ranges = sorted((s, e) for s, e in date_ranges)
    for s, e in ranges:
        if e < s:
            raise ValueError("end date is before start date")
    merged: list[list[date]] = []
    for s, e in ranges:
        if merged and s <= merged[-1][1] + timedelta(days=1):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    totals: dict[int, int] = {}
    for s, e in merged:
        for year in range(s.year, e.year + 1):
            lo, hi = max(s, date(year, 1, 1)), min(e, date(year, 12, 31))
            totals[year] = totals.get(year, 0) + (hi - lo).days + 1
    return totals


@tool
def days_in_india_calculator(date_ranges: list[list[str]]) -> dict:
    """Add up days spent in India. Input: a list of [start, end] pairs as
    YYYY-MM-DD strings that the USER supplied. Returns days per calendar year.
    Both the start and end dates are counted. This only adds up days: it does
    NOT decide residential status or apply any legal threshold. Call it only
    when the user gave dates."""
    parsed = [(date.fromisoformat(s), date.fromisoformat(e)) for s, e in date_ranges]
    return {"days_per_year": count_days_by_year(parsed), "end_date_counted": True,
            "note": "Counts days only. Does not decide residential status."}


# --- community scrubber -----------------------------------------------------
_HANDLE = re.compile(r"(?<![\w.])(?:u|r)/[\w-]+|(?<![\w.])@[\w.]{2,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def scrub(text: str) -> str:
    """Remove usernames, handles, emails and long numbers from community text."""
    text = _EMAIL.sub("[removed]", text)
    text = _HANDLE.sub("[user]", text)
    return _LONG_NUMBER.sub("[removed]", text)


# --- official docs ----------------------------------------------------------
@tool
def search_official_docs(query: str) -> list[dict]:
    """Search the official RBI / Income Tax documents. Use this for ANY rule,
    threshold, form, deadline or procedure. Returns passages with title, issuer,
    page and a page link. Returns an empty list when nothing is relevant enough."""
    chunks = retrieve(query, config.TOP_K)
    if not chunks or max(c.score for c in chunks) < config.RELEVANCE_THRESHOLD:
        return []
    return [{"text": c.text, "score": round(c.score, 3), **{
        k: c.metadata[k] for k in ("doc_id", "title", "issuer", "page", "page_url", "last_verified")}}
        for c in chunks]


# --- community search (Tavily) ------------------------------------------------
def site_of(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc.removeprefix("www.")


def fetch_community(query: str, client=None, max_results: int | None = None) -> list[dict]:
    """Search public community pages. Keeps ONLY url, title, snippet, date, site,
    and scrubs usernames/identifiers from every text field before returning."""
    max_results = max_results or config.COMMUNITY_MAX_RESULTS
    if client is None:
        if not config.TAVILY_API_KEY:
            return []
        from langchain_tavily import TavilySearch
        client = TavilySearch(max_results=max_results, tavily_api_key=config.TAVILY_API_KEY)
    raw = client.invoke({"query": query, "include_domains": config.COMMUNITY_DOMAINS})
    results = raw.get("results", []) if isinstance(raw, dict) else []
    out = []
    for r in results[:max_results]:
        out.append({
            "url": r.get("url", ""),
            "title": plain(scrub(r.get("title", ""))),
            "snippet": plain(scrub(r.get("content", ""))),
            "date": r.get("published_date", "") or "",
            "site": site_of(r.get("url", "")),
        })
    return out


@tool
def search_community(query: str) -> list[dict]:
    """Search public community discussions (Reddit, X, Quora) for personal experiences.
    These are NOT authoritative. Use only after official documents were searched.
    Returns the link to each conversation with title, snippet, date and site only."""
    return fetch_community(query)
