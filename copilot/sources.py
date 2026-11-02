"""Source list for the app footer: last_verified (and effective_date when set)."""
import csv
from datetime import date

import config


def pretty_date(value: str) -> str:
    """'2015-07-01' becomes '1 July 2015'. Anything that is not an ISO date is returned as is."""
    try:
        d = date.fromisoformat(value)
    except (TypeError, ValueError):
        return value or ""
    return f"{d.day} {d.strftime('%B %Y')}"


def describe_citation(c) -> str:
    """Markdown for one source: the document, its release date when known, the page, and links
    to that page and to the full document."""
    released = f", released {pretty_date(c.published)}" if c.published else ""
    return (f"{c.document_title} ({c.issuer}){released}, page {c.page} "
            f"([open that page]({c.page_url}) or [get the full document]({c.document_url}))")


def footer_lines(manifest_path=None) -> list[str]:
    with open(manifest_path or config.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = []
    for r in rows:
        text = f"{r['title']} ({r['issuer']}): last verified {r['last_verified'] or 'unknown'}"
        if r.get("effective_date"):
            text += f", effective {r['effective_date']}"
        lines.append(text)
    return lines
