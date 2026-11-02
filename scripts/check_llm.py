"""Check each configured LLM key, and whether its model can do what the pipeline needs.

Run:  python scripts/check_llm.py
For every provider that has a key it tests: a plain reply, structured output, and tool calling.
Providers that fail structured output or tool calling should be moved down or out of
LLM_PROVIDER_ORDER. Prints no keys.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot import llm  # noqa: E402
from copilot.schemas import ScopeResult  # noqa: E402
from copilot.tools import days_in_india_calculator  # noqa: E402


def check(provider: str) -> dict:
    model = llm.get_chat_model(provider)
    out = {}
    for name, fn in {
        "reply": lambda: model.invoke("Reply with the single word: ok").content,
        "structured_output": lambda: model.with_structured_output(ScopeResult, method="function_calling").invoke(
            "Classify: 'Which shares should I buy?'").label,
        "tool_calling": lambda: [c["name"] for c in model.bind_tools([days_in_india_calculator]).invoke(
            "I was in India from 2024-01-01 to 2024-01-10. How many days is that?").tool_calls],
    }.items():
        try:
            out[name] = "OK: " + str(fn())[:60]
        except Exception as e:
            out[name] = f"FAIL: {type(e).__name__}: {str(e)[:100]}"
    return out


if __name__ == "__main__":
    found = llm.available_providers()
    if not found:
        sys.exit("No provider keys found. Add keys to .env (see .env.example).")
    for p in found:
        print(f"\n{p} ({config.PROVIDERS[p]['model']})")
        for k, v in check(p).items():
            print(f"  {k:18} {v}")
