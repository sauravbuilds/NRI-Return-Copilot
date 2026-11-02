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


def test_small_steps_use_the_fast_model_and_the_answer_uses_the_main_one(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    cfg = config.PROVIDERS["groq"]
    assert llm.get_chat_model("groq", fast=True).model_name == cfg["fast_model"]
    assert llm.get_chat_model("groq").model_name == cfg["model"]
    llm._try_each(lambda p, m: "ok", fast=True)
    assert llm.last_model_used == f"groq:{cfg['fast_model']}"
    llm._try_each(lambda p, m: "ok")
    assert llm.last_model_used == f"groq:{cfg['model']}"


def test_a_provider_without_a_separate_fast_model_uses_its_main_one():
    cfg = config.PROVIDERS["openrouter"]
    assert cfg["fast_model"] == cfg["model"]


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


def test_the_other_model_of_a_provider_is_a_fallback_when_one_is_rate_limited(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    cfg, tried = config.PROVIDERS["groq"], []

    def action(p, model):
        tried.append(model.model_name)
        if model.model_name == cfg["model"]:
            raise RateLimitError("daily tokens")
        return "ok"
    assert llm._try_each(action) == "ok"
    assert tried == [cfg["model"], cfg["fast_model"]] and llm.last_model_used == f"groq:{cfg['fast_model']}"


def test_rate_limited_providers_get_one_more_round_after_a_wait(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setitem(config.PROVIDERS["groq"], "fast_model", config.PROVIDERS["groq"]["model"])
    waits, attempts = [], []
    monkeypatch.setattr(llm.time, "sleep", waits.append)

    def action(p, _model):
        attempts.append(p)
        if len(attempts) == 1:
            raise RateLimitError("tokens per minute")
        return "ok"

    assert llm._try_each(action) == "ok"
    assert waits == [config.RATE_LIMIT_WAIT_SECONDS] and attempts == ["groq", "groq"]


def test_after_every_provider_is_rate_limited_calls_fail_at_once_for_a_while(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def limited(p, _model):
        calls.append(p)
        raise RateLimitError("daily quota")
    with pytest.raises(RuntimeError, match="no LLM provider worked"):
        llm._try_each(limited)
    attempts = len(calls)
    with pytest.raises(RuntimeError, match="retrying shortly"):
        llm._try_each(limited)
    assert len(calls) == attempts  # the second call never reached a provider


def test_the_cooldown_ends(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm._busy_until = llm.time.monotonic() - 1
    assert llm._try_each(lambda p, m: "ok") == "ok"


def test_other_failures_do_not_wait(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    with pytest.raises(RuntimeError):
        llm._try_each(lambda p, m: (_ for _ in ()).throw(ValueError("bad output")))
    assert waits == []
