"""Build the prepared answers shown under "Common questions".

Run:  python scripts/build_prepared_answers.py

Reads data/common_questions.json and data/curated_answers.json, checks that every quote is on
its page of the downloaded PDF, and writes data/prepared_answers.json. It stops with the
failing quote if anything cannot be verified. Run it again after the documents are refreshed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import curation, prepared  # noqa: E402


def main() -> None:
    entries = curation.build_entries()
    quotes = sum(len(f.get("citations", [])) for e in entries for f in e["answer"]["official_findings"])
    prepared.save(entries)
    print(f"Verified and saved {len(entries)} answers ({quotes} cited statements checked against their pages)")


if __name__ == "__main__":
    main()
