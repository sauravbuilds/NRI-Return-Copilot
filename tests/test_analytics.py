import json

import pandas as pd

from analytics import chart, collect, tag
from copilot import llm
from copilot.schemas import ThreadTag


class Client:
    def invoke(self, _):
        return {"results": [{"url": "https://a.com/1", "title": "t u/bob", "content": "c @amy"},
                            {"url": "https://a.com/1", "title": "dup", "content": "dup"}]}


def test_collect_stores_only_allowed_fields_dedupes_and_no_usernames(tmp_path):
    p = tmp_path / "t.jsonl"
    n = collect.collect(client=Client(), path=p)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert n == len(rows) == 1  # same URL across all queries is stored once
    assert set(rows[0]) == {"url", "title", "snippet", "date", "site"}
    assert "bob" not in str(rows) and "amy" not in str(rows)


def test_tagger_writes_tags_and_skips_failures(tmp_path, monkeypatch):
    src, dest = tmp_path / "s.jsonl", tmp_path / "d.jsonl"
    src.write_text("\n".join(json.dumps({"url": f"u{i}", "title": "t", "snippet": "s", "date": "", "site": "x"})
                             for i in range(3)), encoding="utf-8")
    calls = iter([ThreadTag(problem_type="other", stage="unclear", resolved="no"), RuntimeError("x"),
                  ThreadTag(problem_type="nre_nro_accounts", stage="first_year", resolved="yes")])

    def fake(schema, system, user):
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(llm, "structured_call", fake)
    assert tag.tag_all(src, dest) == 2


def test_aggregate_and_chart_show_sample_size(tmp_path):
    df = pd.DataFrame({"problem_type": ["other", "nre_nro_accounts", "nre_nro_accounts"],
                       "stage": ["unclear"] * 3, "resolved": ["no", "yes", "no"]})
    agg = chart.aggregate(df)
    assert agg["n"] == 3 and agg["problem_type"]["nre_nro_accounts"] == 2
    out = chart.make_chart(df, tmp_path / "c.png")
    assert out.exists() and out.stat().st_size > 1000


def test_empty_data_is_handled():
    assert chart.aggregate(pd.DataFrame())["n"] == 0
