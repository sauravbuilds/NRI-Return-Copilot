import pytest

from copilot import llm


@pytest.fixture(autouse=True)
def providers_not_busy():
    """The model client remembers a rate-limit cooldown; no test should inherit another's."""
    llm._busy_until = 0.0
    yield
    llm._busy_until = 0.0
