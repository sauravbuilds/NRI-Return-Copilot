"""Compare each community thread with the official passages: agrees / contradicts / unrelated."""
from copilot import llm
from copilot.prompts import COMMUNITY_COMPARISON
from copilot.schemas import CommunityFinding, VerdictResult
from copilot.text import plain


def format_passages(chunks) -> str:
    return "\n\n".join(f"[{c.metadata['title']}, page {c.metadata['page']}]\n{c.text}" for c in chunks)


def compare_snippet(snippet: str, chunks) -> VerdictResult:
    user = f"OFFICIAL PASSAGES:\n{format_passages(chunks)}\n\nCOMMUNITY SNIPPET:\n{snippet}"
    return llm.structured_call(VerdictResult, COMMUNITY_COMPARISON, user)


def compare_threads(threads: list[dict], chunks) -> list[CommunityFinding]:
    """Threads are judged independently. Community text never enters official findings."""
    findings = []
    for t in threads:
        try:
            v = compare_snippet(t["snippet"], chunks)
        except Exception:  # a failed comparison must not hide the thread or invent a verdict
            v = VerdictResult(label="unrelated", reason="I couldn't compare this one with the official text.")
        findings.append(CommunityFinding(url=t["url"], title=t["title"], snippet=t["snippet"],
                                         verdict=v.label, reason=plain(v.reason)))
    return findings
