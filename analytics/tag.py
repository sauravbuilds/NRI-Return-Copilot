"""Tag each collected thread (problem type, stage, resolved) with structured LLM output.

Run:  python analytics/tag.py        (needs an LLM key)
Writes analytics/threads_tagged.jsonl.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot import llm  # noqa: E402
from copilot.prompts import THREAD_TAGGER  # noqa: E402
from copilot.schemas import ThreadTag  # noqa: E402

TAGGED_PATH = config.ROOT / "analytics" / "threads_tagged.jsonl"


def tag_thread(thread: dict) -> ThreadTag:
    return llm.structured_call(ThreadTag, THREAD_TAGGER, f"Title: {thread['title']}\nSnippet: {thread['snippet']}",
                               fast=True)


def tag_all(src=config.COMMUNITY_JSONL, dest=TAGGED_PATH) -> int:
    rows = [json.loads(l) for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
    n = 0
    with open(dest, "w", encoding="utf-8") as f:
        for t in rows:
            try:
                tag = tag_thread(t)
            except Exception:
                continue  # an untaggable thread is left out, not guessed
            f.write(json.dumps({**t, **tag.model_dump()}, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    print(f"tagged {tag_all()} threads -> {TAGGED_PATH}")
