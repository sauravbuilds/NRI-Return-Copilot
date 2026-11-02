"""Compare community threads with the official passages: agrees / contradicts / unrelated.

All threads go to the model in one call, so the time does not grow with the number of threads.
"""
from copilot import llm
from copilot.prompts import COMMUNITY_COMPARISON
from copilot.schemas import CommunityFinding, ThreadVerdicts, VerdictResult
from copilot.text import plain

NOT_COMPARED = "I couldn't compare this one with the official text."


def format_passages(chunks) -> str:
    return "\n\n".join(f"[{c.metadata['title']}, page {c.metadata['page']}]\n{c.text}" for c in chunks)


def compare_many(snippets: list[str], chunks) -> list[VerdictResult]:
    """One verdict per snippet, in order. Raises if no model could answer. A snippet the model
    skipped is reported as unrelated rather than given an invented verdict."""
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(snippets, start=1))
    user = f"OFFICIAL PASSAGES:\n{format_passages(chunks)}\n\nCOMMUNITY SNIPPETS:\n{numbered}"
    by_number = {v.number: v for v in llm.structured_call(ThreadVerdicts, COMMUNITY_COMPARISON, user, fast=True).verdicts}
    return [VerdictResult(label=by_number[i].label, reason=by_number[i].reason) if i in by_number
            else VerdictResult(label="unrelated", reason=NOT_COMPARED)
            for i in range(1, len(snippets) + 1)]


def compare_snippet(snippet: str, chunks) -> VerdictResult:
    return compare_many([snippet], chunks)[0]


def compare_threads(threads: list[dict], chunks) -> list[CommunityFinding]:
    """Community text never enters official findings. A failed comparison must not hide the
    threads or invent a verdict."""
    if not threads:
        return []
    try:
        verdicts = compare_many([t["snippet"] for t in threads], chunks)
    except Exception:
        verdicts = [VerdictResult(label="unrelated", reason=NOT_COMPARED)] * len(threads)
    return [CommunityFinding(url=t["url"], title=t["title"], snippet=t["snippet"],
                             verdict=v.label, reason=plain(v.reason))
            for t, v in zip(threads, verdicts)]
