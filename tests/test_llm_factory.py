import pytest

import config
from copilot import llm


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    for cfg in config.PROVIDERS.values():
        monkeypatch.delenv(cfg["key_env"], raising=False)


def test_no_keys_means_no_llm():
    assert not llm.llm_available()
    with pytest.raises(RuntimeError, match="none have a key"):
        llm.structured_call(dict, "s", "u")


def test_providers_without_key_are_skipped_and_order_respected(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert llm.available_providers() == ["groq", "mistral"]


def test_models_are_built_at_temperature_zero(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    m = llm.get_chat_model("groq")
    assert m.temperature == 0 and m.model_name == config.PROVIDERS["groq"]["model"]


def test_falls_back_to_next_provider_and_records_who_answered(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    def action(p, _model):
        calls.append(p)
        if p == "groq":
            raise RuntimeError("429 rate limit")
        return "ok"

    assert llm._try_each(action) == "ok"
    assert calls == ["groq", "gemini"]
    assert llm.last_model_used.startswith("gemini:")


def test_all_failing_raises_clear_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(RuntimeError, match="no LLM provider worked"):
        llm._try_each(lambda p, m: (_ for _ in ()).throw(ValueError("x")))


class RateLimitError(Exception):
    """Named like the providers' own rate-limit errors."""


def test_rate_limited_providers_get_one_more_round_after_a_wait(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    waits, attempts = [], []
    monkeypatch.setattr(llm.time, "sleep", waits.append)

    def action(p, _model):
        attempts.append(p)
        if len(attempts) == 1:
            raise RateLimitError("tokens per minute")
        return "ok"

    assert llm._try_each(action) == "ok"
    assert waits == [config.RATE_LIMIT_WAIT_SECONDS] and attempts == ["groq", "groq"]


def test_other_failures_do_not_wait(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    with pytest.raises(RuntimeError):
        llm._try_each(lambda p, m: (_ for _ in ()).throw(ValueError("bad output")))
    assert waits == []
