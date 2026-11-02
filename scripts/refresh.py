"""Re-download every manifest document and detect changes.

Run:  python scripts/refresh.py

Documents are fetched with the Scrapy crawler in scripts/crawler.py.
For each row: save the download to a temp file, compare SHA-256 with the stored one.
 - failed download  -> keep the old file, log it (nothing is ever deleted first)
 - unchanged        -> update last_verified only
 - changed          -> archive the old file, save the new one, update hashes/dates
If anything changed, the ChromaDB index is rebuilt and we warn when a changed
document is the expected source of a golden-set question.
Only documents listed in the manifest are covered.
"""
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

GOLDEN_PATH = config.EVALS_DIR / "golden_set.json"


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crawl_fetcher(rows: list[dict]):
    """A `fetch(url, dest)` backed by one Scrapy crawl of every manifest url. A document
    that could not be fetched raises when asked for, so it is logged and its old file kept."""
    from scripts.crawler import fetch_pdfs
    fetched = fetch_pdfs({row["doc_id"]: row["url"] for row in rows})
    doc_for_url = {row["url"]: row["doc_id"] for row in rows}

    def fetch(url: str, dest: Path) -> None:
        outcome = fetched[doc_for_url[url]]
        if isinstance(outcome, Exception):
            raise outcome
        dest.write_bytes(outcome)
    return fetch


def golden_docs() -> set[str]:
    """doc_ids named as the expected source of any golden-set question."""
    if not GOLDEN_PATH.exists():
        return set()
    text = json.dumps(json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
    return {r["doc_id"] for r in _read(config.MANIFEST_PATH) if r["doc_id"] in text}


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def refresh(manifest_path=None, official_dir=None, archive_dir=None, log_path=None,
            fetch=None, today=None) -> dict:
    manifest_path = Path(manifest_path or config.MANIFEST_PATH)
    official_dir = Path(official_dir or config.OFFICIAL_DIR)
    archive_dir = Path(archive_dir or config.ARCHIVE_DIR)
    log_path = Path(log_path or config.REFRESH_LOG_PATH)
    today = today or date.today().isoformat()

    rows = _read(manifest_path)
    fetch = fetch or crawl_fetcher(rows)
    columns = list(rows[0].keys()) if rows else []
    result = {"changed": [], "failed": [], "unchanged": []}

    for row in rows:
        doc_id = row["doc_id"]
        target = official_dir / f"{doc_id}.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "new.pdf"
            try:
                fetch(row["url"], tmp_file)
                new_hash = sha256_of(tmp_file)
            except Exception as e:  # keep the old file, log, move on
                result["failed"].append((doc_id, f"{type(e).__name__}: {e}"))
                continue

            if target.exists() and new_hash == row.get("sha256"):
                row["last_verified"] = today
                result["unchanged"].append(doc_id)
                continue

            if target.exists():  # archive only after a successful download
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(archive_dir / f"{doc_id}_{today}.pdf"))
            shutil.copy(tmp_file, target)
            row.update(sha256=new_hash, date_downloaded=today, last_verified=today)
            result["changed"].append(doc_id)

    if rows:
        _write(manifest_path, rows, columns)
    _write_log(log_path, result, today)
    return result


def _write_log(log_path: Path, result: dict, today: str) -> None:
    lines = [f"# Refresh log\n\nLast run: {today}\n"]
    for key in ("changed", "failed", "unchanged"):
        lines.append(f"\n## {key.capitalize()} ({len(result[key])})\n")
        for item in result[key]:
            lines.append(f"- {item[0]}: {item[1]}" if isinstance(item, tuple) else f"- {item}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = refresh()
    print({k: len(v) for k, v in result.items()})
    for doc_id, why in result["failed"]:
        print(f"FAILED {doc_id}: {why}")
    if result["changed"]:
        from scripts.ingest import build_index
        build_index()
        hit = sorted(set(result["changed"]) & golden_docs())
        if hit:
            print(f"WARNING: changed documents are expected sources in the golden set: {hit}. "
                  "Re-run the evals: python evals/run_evals.py")


if __name__ == "__main__":
    main()
