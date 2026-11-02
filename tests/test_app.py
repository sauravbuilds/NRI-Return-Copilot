"""UI tests with Streamlit's AppTest. The pipeline is replaced by a canned result."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from copilot import llm, orchestrator
from copilot.orchestrator import PipelineResult
from copilot.schemas import (Answer, Citation, CommunityFinding, Finding, MissingItem, OfficialUpdate,
                             Persona, RoadmapStep, DISCLAIMER)

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CITE = Citation(document_title="Doc One", issuer="RBI", page=2, doc_id="d1",
                page_url="https://x.org/d1.pdf#page=2", last_verified="2026-11-01", published="2015-07-01")


def canned(kind="answered", review=False, updates=()):
    ans = Answer(
        summary="Plain answer.", official_findings=[Finding(statement="Official statement", citations=[CITE])],
        roadmap=[RoadmapStep(stage="first_30_days", step="Do the thing", citations=[CITE])],
        missing_information=[MissingItem(field="move_date", why_it_matters="matters", if_this_then_that="then changes")],
        community_findings=[CommunityFinding(url="https://f.com/1", title="Forum post", snippet="people say",
                                             verdict="contradicts", reason="Conflicts with page 2")],
        official_updates=list(updates), confidence="high", needs_human_review=review,
        review_reasons=["A community thread contradicts the official source"] if review else [],
        ca_brief="BRIEF TEXT" if review else None)
    return PipelineResult(kind, ans.summary, ans, Persona())


def run_direct(monkeypatch, result):
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda q, p=None, community=True: result)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Ask directly").run()
    at.text_area(key="direct_question").set_value("What happens to my NRO account?").run()
    at.button[0].click().run()
    return at


def headers(at):
    return [s.value for s in at.subheader]


def test_four_sections_in_order_with_separate_community_tier(monkeypatch):
    at = run_direct(monkeypatch, canned(review=True))
    assert not at.exception
    numbered = [h for h in headers(at) if h[0].isdigit()]
    assert numbered == ["1. Your roadmap", "2. What I could not determine",
                        "3. Community experience (not official)", "4. Take this to a CA"]
    text = " ".join(m.value for m in at.markdown)
    # citation names the release date and links to the exact page and to the whole document
    assert "Doc One (RBI), released 1 July 2015, page 2" in text
    assert "[open that page](https://x.org/d1.pdf#page=2)" in text
    assert "[get the full document](https://x.org/d1.pdf)" in text
    assert "Contradicts official text" in text and "Forum post" in text
    assert "[Forum post](https://f.com/1) on f.com" in text  # link opens the conversation
    assert any("chartered accountant" in w.value for w in at.warning)  # banner


def test_newer_official_publication_is_shown_above_the_roadmap(monkeypatch):
    update = OfficialUpdate(url="https://www.rbi.org.in/new.pdf", title="New circular", snippet="rule changed",
                            date="2026-11-02", verdict="changes", reason="It replaces the old limit.")
    at = run_direct(monkeypatch, canned(updates=[update]))
    assert any("Heads up" in w.value for w in at.warning)
    text = " ".join(m.value for m in at.markdown)
    assert "[New circular](https://www.rbi.org.in/new.pdf)" in text and "May change this answer" in text
    assert "Published 2 November 2026" in text


def test_no_updates_means_no_heads_up(monkeypatch):
    at = run_direct(monkeypatch, canned())
    assert not any("Heads up" in w.value for w in at.warning)


def test_privacy_line_and_disclaimer_always_shown(monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("don't type phone numbers" in i.value for i in at.info)
    assert any(DISCLAIMER in c.value for c in at.caption)


def test_pii_block_and_refusal_show_friendly_messages(monkeypatch):
    at = run_direct(monkeypatch, PipelineResult("pii_blocked", "Please resend without a phone number."))
    assert any("without a phone number" in e.value for e in at.error)
    at = run_direct(monkeypatch, PipelineResult("refused", "I can't recommend investments."))
    assert any("can't recommend" in i.value for i in at.info)


def test_abstention_message_and_brief_download(monkeypatch):
    ans = Answer(summary="Could not find it.", abstained=True, needs_human_review=True,
                 review_reasons=["No official text supports an answer"], ca_brief="BRIEF")
    at = run_direct(monkeypatch, PipelineResult("abstained", ans.summary, ans, Persona()))
    assert any("Could not find it" in w.value for w in at.warning)
    assert "4. Take this to a CA" in headers(at)


def test_a_session_cannot_exceed_the_question_limit(monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_QUESTIONS_PER_SESSION", 1)
    monkeypatch.setattr(config, "MIN_SECONDS_BETWEEN_QUESTIONS", 0)
    calls = []
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(orchestrator, "run_pipeline",
                        lambda q, p=None, community=True: calls.append(q) or canned())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Ask directly").run()
    at.text_area(key="direct_question").set_value("first question").run()
    at.button[0].click().run()
    at.button[0].click().run()
    assert len(calls) == 1 and any("question limit" in w.value for w in at.warning)


def test_questions_asked_in_quick_succession_are_held(monkeypatch):
    import config
    monkeypatch.setattr(config, "MIN_SECONDS_BETWEEN_QUESTIONS", 60)
    calls = []
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(orchestrator, "run_pipeline",
                        lambda q, p=None, community=True: calls.append(q) or canned())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Ask directly").run()
    at.text_area(key="direct_question").set_value("a question").run()
    at.button[0].click().run()
    at.button[0].click().run()
    assert len(calls) == 1 and any("few seconds" in w.value for w in at.warning)


def test_missing_model_gives_clear_message(monkeypatch):
    monkeypatch.setattr(llm, "llm_available", lambda: False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Ask directly").run()
    at.text_area(key="direct_question").set_value("hi").run()
    at.button[0].click().run()
    assert any("No language model" in e.value for e in at.error)


def test_guided_mode_walks_through_eight_questions():
    at = AppTest.from_file(APP, default_timeout=30).run()
    def click(label):
        next(b for b in at.button if b.label == label).click().run()

    for expected in range(1, 9):
        assert any(f"Question {expected} of 8" in str(p.text) for p in at.get("progress"))
        click("Next")
    assert any(b.label == "Get my roadmap" for b in at.button)


def test_guided_back_button_returns_to_previous_question():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not any(b.label == "Back" for b in at.button)  # nothing before the first question
    next(b for b in at.button if b.label == "Next").click().run()
    next(b for b in at.button if b.label == "Back").click().run()
    assert any("Question 1 of 8" in str(p.text) for p in at.get("progress"))


def test_switching_mode_clears_the_previous_result(monkeypatch):
    at = run_direct(monkeypatch, canned())
    assert "1. Your roadmap" in headers(at)
    at.radio(key="mode").set_value("Guided").run()
    assert "1. Your roadmap" not in headers(at)
