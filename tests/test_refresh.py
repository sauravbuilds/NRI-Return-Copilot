import csv

import pytest

from copilot.sources import footer_lines
from scripts import refresh as R

COLS = ["doc_id", "title", "issuer", "url", "date_published", "date_downloaded",
        "scenario_tags", "sha256", "last_verified", "effective_date"]


@pytest.fixture
def env(tmp_path):
    off, arch = tmp_path / "official", tmp_path / "archive"
    off.mkdir()
    (off / "d1.pdf").write_bytes(b"%PDF-old")
    row = dict.fromkeys(COLS, "")
    row.update(doc_id="d1", title="T", issuer="RBI", url="http://x/d1.pdf",
               sha256=R.sha256_of(off / "d1.pdf"), last_verified="2026-01-01")
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerow(row)
    return dict(manifest_path=manifest, official_dir=off, archive_dir=arch,
                log_path=tmp_path / "log.md", today="2026-11-01")


def row_of(env):
    return list(csv.DictReader(open(env["manifest_path"], encoding="utf-8")))[0]


def test_unchanged_file_keeps_hash_and_updates_last_verified(env):
    old_hash = row_of(env)["sha256"]
    res = R.refresh(**env, fetch=lambda u, d: d.write_bytes(b"%PDF-old"))
    r = row_of(env)
    assert res["unchanged"] == ["d1"]
    assert r["sha256"] == old_hash and r["last_verified"] == "2026-11-01"
    assert r["date_downloaded"] == ""  # untouched
    assert not env["archive_dir"].exists()


def test_failed_download_keeps_old_file_and_logs(env):
    def boom(u, d):
        raise ConnectionError("down")
    res = R.refresh(**env, fetch=boom)
    assert res["failed"][0][0] == "d1"
    assert (env["official_dir"] / "d1.pdf").read_bytes() == b"%PDF-old"
    assert row_of(env)["last_verified"] == "2026-01-01"
    assert "d1" in env["log_path"].read_text(encoding="utf-8")


def test_changed_file_is_archived_and_replaced(env):
    res = R.refresh(**env, fetch=lambda u, d: d.write_bytes(b"%PDF-new"))
    assert res["changed"] == ["d1"]
    assert (env["archive_dir"] / "d1_2026-11-01.pdf").read_bytes() == b"%PDF-old"
    assert (env["official_dir"] / "d1.pdf").read_bytes() == b"%PDF-new"
    r = row_of(env)
    assert r["sha256"] == R.sha256_of(env["official_dir"] / "d1.pdf")
    assert r["date_downloaded"] == "2026-11-01"


def test_crawl_fetcher_writes_pdfs_and_raises_for_failed_documents(tmp_path, monkeypatch):
    import scripts.crawler as crawler
    monkeypatch.setattr(crawler, "fetch_pdfs", lambda targets: {"ok": b"%PDF-1", "bad": RuntimeError("HTTP 403")})
    rows = [{"doc_id": "ok", "url": "http://x/ok.pdf"}, {"doc_id": "bad", "url": "http://x/bad.pdf"}]
    fetch = R.crawl_fetcher(rows)
    fetch("http://x/ok.pdf", tmp_path / "ok.pdf")
    assert (tmp_path / "ok.pdf").read_bytes() == b"%PDF-1"
    with pytest.raises(RuntimeError, match="HTTP 403"):
        fetch("http://x/bad.pdf", tmp_path / "bad.pdf")
    assert not (tmp_path / "bad.pdf").exists()


def test_footer_shows_effective_date_only_when_present(env):
    assert "effective" not in footer_lines(env["manifest_path"])[0]
    rows = list(csv.DictReader(open(env["manifest_path"], encoding="utf-8")))
    rows[0]["effective_date"] = "2025-04-01"
    R._write(env["manifest_path"], rows, COLS)
    assert "effective 2025-04-01" in footer_lines(env["manifest_path"])[0]
