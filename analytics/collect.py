"""Collect public community threads with Tavily. Stores ONLY url, title, snippet, date, site.

Run:  python analytics/collect.py        (needs TAVILY_API_KEY)
Usernames and identifiers are scrubbed; nothing else about a thread is kept.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot.tools import fetch_community  # noqa: E402

# Several search phrasings per scenario (search terms only, not facts).
QUERIES = {
    "residential_status": ["NRI returning to India residential status days rules experience",
                           "returned to India how many days resident NRI tax status"],
    "nre_nro_accounts": ["NRE NRO account after returning to India convert resident savings",
                         "returning NRI what to do with NRE account"],
    "us_stocks_rsus": ["returning to India US stocks RSUs what happens experience",
                       "moving back to India vested RSUs reporting"],
    "foreign_retirement": ["returning NRI 401k India what to do",
                           "move to India foreign retirement account reporting"],
    "foreign_asset_reporting": ["Schedule FA foreign assets reporting returning NRI",
                                "returning NRI foreign assets income tax return India"],
}


def collect(client=None, path=config.COMMUNITY_JSONL,
            per_query: int = config.COMMUNITY_COLLECT_PER_QUERY) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        seen = {json.loads(l)["url"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    added = 0
    with open(path, "a", encoding="utf-8") as f:
        for qs in QUERIES.values():
            for q in qs:
                for t in fetch_community(q, client=client, max_results=per_query):
                    if t["url"] and t["url"] not in seen:  # de-duplicate by URL
                        seen.add(t["url"])
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")
                        added += 1
    return added


if __name__ == "__main__":
    print(f"added {collect()} new threads -> {config.COMMUNITY_JSONL}")
