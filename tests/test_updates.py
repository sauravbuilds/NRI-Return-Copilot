"""Newer-publication check, citation wording and dash removal (no keys, no network)."""
import csv
from datetime import date

import pytest

import config
from copilot import llm
from copilot.retrieval import Chunk
from copilot.schemas import Citation, UpdateVerdictResult
from copilot.sources import describe_citation, pretty_date
from copilot.text import plain
from copilot.updates import SearchUnavailable, compare_updates, fetch_official_updates, last_checked
from guardrails import advice
from guardrails.pii import PIIResult

KNOWN = "https://www.rbi.org.in/known.pdf"


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.csv"
    rows = [{"doc_id": "a", "url": KNOWN, "date_downloaded": "2026-10-28", "last_verified": "2026-10-30"},
            {"doc_id": "b", "url": "https://x.org/b.pdf", "date_downloaded": "2026-10-31", "last_verified": ""}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


class FakeSearch:
    def __init__(self, results):
        self.results, self.args = results, None

    def invoke(self, args):
        self.args = args
        return {"results": self.results}


def test_last_checked_is_the_latest_check_or_download_day(manifest):
    assert last_checked(manifest) == date(2026, 10, 31)


def test_search_is_limited_to_official_sites_and_to_items_after_the_last_check(manifest):
    client = FakeSearch([{"url": "https://www.rbi.org.in/new", "title": "New", "content": "text"}])
    fetch_official_updates("nro account rules", date(2026, 10, 31), client=client, manifest_path=manifest)
    assert client.args["include_domains"] == config.OFFICIAL_DOMAINS
    assert client.args["start_date"] == "2026-10-31"


def test_documents_already_in_the_manifest_are_not_reported_as_new(manifest):
    client = FakeSearch([{"url": KNOWN, "title": "Known", "content": "x"},
                         {"url": "https://www.rbi.org.in/new", "title": "New", "content": "y"}])
    out = fetch_official_updates("q", date(2026, 10, 31), client=client, manifest_path=manifest)
    assert [u["url"] for u in out] == ["https://www.rbi.org.in/new"] and out[0]["site"] == "rbi.org.in"


def test_update_results_are_capped(manifest):
    client = FakeSearch([{"url": f"https://www.rbi.org.in/{i}", "title": "t", "content": "c"} for i in range(10)])
    out = fetch_official_updates("q", date(2026, 10, 31), client=client, manifest_path=manifest)
    assert len(out) == config.OFFICIAL_UPDATE_MAX_RESULTS


def test_without_a_search_key_the_check_reports_unavailable(manifest, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", None)
    with pytest.raises(SearchUnavailable):
        fetch_official_updates("q", date(2026, 10, 31), manifest_path=manifest)


def test_comparison_failure_keeps_the_item_as_unchecked(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(llm, "structured_call", boom)
    chunk = Chunk("t", {"title": "Doc", "page": 1}, 0.9)
    out = compare_updates([{"url": "u", "title": "t", "snippet": "s", "date": ""}], [chunk])
    assert out[0].verdict == "unchecked"


def test_comparison_reason_has_no_dashes(monkeypatch):
    monkeypatch.setattr(llm, "structured_call",
                        lambda *a, **k: UpdateVerdictResult(label="changes", reason="New limit — not the old one"))
    chunk = Chunk("t", {"title": "Doc", "page": 1}, 0.9)
    out = compare_updates([{"url": "u", "title": "t", "snippet": "s", "date": ""}], [chunk])
    assert out[0].reason == "New limit, not the old one"


# --- citation wording ---------------------------------------------------------
def cite(published=""):
    return Citation(document_title="Master Circular", issuer="RBI", page=3, doc_id="d",
                    page_url="https://x.org/d.pdf#page=3", last_verified="2026-11-01", published=published)


def test_citation_names_release_date_page_and_both_links():
    text = describe_citation(cite("2015-07-01"))
    assert "released 1 July 2015, page 3" in text
    assert "(https://x.org/d.pdf#page=3)" in text and "(https://x.org/d.pdf)" in text


def test_citation_without_a_known_date_does_not_invent_one():
    assert "released" not in describe_citation(cite(""))


def test_pretty_date_leaves_other_text_alone():
    assert pretty_date("2026-01-09") == "9 January 2026" and pretty_date("sometime") == "sometime"


# --- no em or en dashes anywhere the user reads ---------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("fast — but careful", "fast, but careful"),
    ("FY 2024–25", "FY 2024-25"),
    ("no dash here", "no dash here"),
    ("", ""),
])
def test_plain_removes_dashes(raw, expected):
    assert plain(raw) == expected


def test_fixed_user_facing_messages_have_no_dashes():
    from copilot.schemas import DISCLAIMER
    from copilot import prompts
    texts = [advice.REFUSAL_ADVICE, advice.REFUSAL_OUT_OF_SCOPE, advice.US_TAX_MESSAGE, DISCLAIMER,
             PIIResult(True, ["pan"]).message, prompts.ANSWER_ASSEMBLY, prompts.UPDATE_COMPARISON,
             prompts.COMMUNITY_COMPARISON]
    assert not any("—" in t or "–" in t for t in texts)
