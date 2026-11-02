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


def available_providers() -> list[str]:
    return [p for p in config.PROVIDER_ORDER
            if p in config.PROVIDERS and os.getenv(config.PROVIDERS[p]["key_env"])]


def llm_available() -> bool:
    return bool(available_providers())


def get_chat_model(provider: str):
    cfg = config.PROVIDERS[provider]
    key = os.getenv(cfg["key_env"])
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=cfg["model"], temperature=config.TEMPERATURE, api_key=key)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=cfg["model"], temperature=config.TEMPERATURE, api_key=key,
                      base_url=cfg["base_url"], max_retries=config.LLM_MAX_RETRIES,
                      timeout=config.LLM_TIMEOUT_SECONDS)


def _try_each(action):
    """Run action(provider, model) on each available provider until one works. If all of
    them fail and at least one was rate limited, wait and make one more round."""
    global last_model_used
    for round_number in range(1 + config.RATE_LIMIT_EXTRA_ROUNDS):
        errors, rate_limited = [], False
        for p in available_providers():
            try:
                out = action(p, get_chat_model(p))
                last_model_used = f"{p}:{config.PROVIDERS[p]['model']}"
                calls_by_model[last_model_used] = calls_by_model.get(last_model_used, 0) + 1
                return out
            except Exception as e:  # rate limit, outage, malformed output...
                errors.append(f"{p}: {type(e).__name__}")
                rate_limited = rate_limited or "RateLimit" in type(e).__name__
        if not rate_limited or round_number == config.RATE_LIMIT_EXTRA_ROUNDS:
            break
        time.sleep(config.RATE_LIMIT_WAIT_SECONDS)
    raise RuntimeError("no LLM provider worked (" + "; ".join(errors or ["none have a key"]) + ")")


def structured_call(schema, system: str, user: str):
    """Ask for output that fits a Pydantic schema (temperature 0).
    method='function_calling' is the most widely supported across free models."""
    def action(_p, model):
        return model.with_structured_output(schema, method="function_calling").invoke(
            [("system", system), ("human", user)])
    return _try_each(action)


def tool_call_step(system: str, user: str, tools: list):
    """Let the model choose tools. Returns the model's message; tool_calls lists its choices."""
    def action(_p, model):
        return model.bind_tools(tools).invoke([("system", system), ("human", user)])
    return _try_each(action)
