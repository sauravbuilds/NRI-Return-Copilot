"""All model calls go through here, so providers are swappable and testable.

Providers (config.PROVIDER_ORDER) are tried in order. A provider is skipped when
it has no key. If a call fails (rate limit, outage, bad output), the next one is
tried. `last_model_used` records who answered, so eval runs can log it.
"""
import os
import time

import config

last_model_used: str = ""
calls_by_model: dict[str, int] = {}
_busy_until: float = 0.0  # monotonic time until which every provider counts as rate limited


def available_providers() -> list[str]:
    return [p for p in config.PROVIDER_ORDER
            if p in config.PROVIDERS and os.getenv(config.PROVIDERS[p]["key_env"])]


def llm_available() -> bool:
    return bool(available_providers())


def model_name(provider: str, fast: bool = False) -> str:
    cfg = config.PROVIDERS[provider]
    return cfg["fast_model"] if fast else cfg["model"]


def get_chat_model(provider: str, fast: bool = False):
    cfg = config.PROVIDERS[provider]
    key = os.getenv(cfg["key_env"])
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model_name(provider, fast), temperature=config.TEMPERATURE, api_key=key)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name(provider, fast), temperature=config.TEMPERATURE, api_key=key,
                      base_url=cfg["base_url"], max_retries=config.LLM_MAX_RETRIES,
                      timeout=config.LLM_TIMEOUT_SECONDS)


def _candidates(fast: bool) -> list[tuple[str, bool]]:
    """(provider, use_fast_model) in the order to try. Every provider's preferred model comes
    first. The other model of each provider follows as a fallback, so a call is not lost when
    one model's daily quota is used up."""
    providers = available_providers()
    other = [p for p in providers if config.PROVIDERS[p]["fast_model"] != config.PROVIDERS[p]["model"]]
    return [(p, fast) for p in providers] + [(p, not fast) for p in other]


def _try_each(action, fast: bool = False):
    """Run action(provider, model) on each candidate until one works. If all of them fail and
    at least one was rate limited, wait and make one more round. When every candidate is rate
    limited, further calls fail at once for a short cooldown, so a busy service answers
    quickly with an error instead of making each step wait."""
    global last_model_used, _busy_until
    if time.monotonic() < _busy_until:
        raise RuntimeError("all providers are rate limited, retrying shortly")
    for round_number in range(1 + config.RATE_LIMIT_EXTRA_ROUNDS):
        errors, limited = [], 0
        for p, use_fast in _candidates(fast):
            try:
                out = action(p, get_chat_model(p, use_fast))
                last_model_used = f"{p}:{model_name(p, use_fast)}"
                calls_by_model[last_model_used] = calls_by_model.get(last_model_used, 0) + 1
                return out
            except Exception as e:  # rate limit, outage, malformed output...
                errors.append(f"{p}{':fast' if use_fast else ''}: {type(e).__name__}")
                limited += "RateLimit" in type(e).__name__
        if not limited or round_number == config.RATE_LIMIT_EXTRA_ROUNDS:
            break
        time.sleep(config.RATE_LIMIT_WAIT_SECONDS)
    if errors and limited == len(errors):
        _busy_until = time.monotonic() + config.RATE_LIMIT_COOLDOWN_SECONDS
    raise RuntimeError("no LLM provider worked (" + "; ".join(errors or ["none have a key"]) + ")")


def structured_call(schema, system: str, user: str, fast: bool = False):
    """Ask for output that fits a Pydantic schema (temperature 0).
    method='function_calling' is the most widely supported across free models.
    `fast` selects each provider's smaller model, for steps that do not write the answer."""
    def action(_p, model):
        return model.with_structured_output(schema, method="function_calling").invoke(
            [("system", system), ("human", user)])
    return _try_each(action, fast)


def tool_call_step(system: str, user: str, tools: list, fast: bool = False):
    """Let the model choose tools. Returns the model's message; tool_calls lists its choices."""
    def action(_p, model):
        return model.bind_tools(tools).invoke([("system", system), ("human", user)])
    return _try_each(action, fast)
