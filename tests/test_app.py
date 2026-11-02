"""UI tests with Streamlit's AppTest. The pipeline is replaced by a canned result."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from copilot import llm, orchestrator
from copilot.orchestrator import PipelineResult
from copilot.schemas import (Answer, Citation, CommunityFinding, CommunityLink, Finding, MissingItem,
                             OfficialUpdate, Persona, RoadmapStep, DISCLAIMER)

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CITE = Citation(document_title="Doc One", issuer="RBI", page=2, doc_id="d1",
                page_url="https://x.org/d1.pdf#page=2", last_verified="2026-11-01", published="2015-07-01")


@pytest.fixture(autouse=True)
def no_index_work(monkeypatch):
    """UI tests need neither the search index nor the embedding model."""
    import streamlit
    from copilot import retrieval, tools
    from scripts import ingest
    monkeypatch.setattr(retrieval, "warm_up", lambda: None)
    monkeypatch.setattr(ingest, "build_index", lambda *a, **k: 0)
    monkeypatch.setattr(tools, "fetch_community", lambda q: [])  # no real searches from UI tests
    streamlit.cache_data.clear()


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
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda q, p=None, community=True, on_step=None: result)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio(key="mode").set_value("Ask directly").run()
    at.text_area(key="direct_question").set_value("What happens to my NRO account?").run()
    at.button[0].click().run()
    return at


def headers(at):
    return [s.value for s in at.subheader]


def test_an_abstained_answer_still_shows_community_discussions_and_us_sources(monkeypatch):
    ans = Answer(summary="Could not find it.", abstained=True, needs_human_review=True, ca_brief="BRIEF",
                 review_reasons=["The official documents I have don't cover this question."],
                 community_links=[CommunityLink(url="https://reddit.com/r/nri/1", title="Moving my 401k",
                                                snippet="people say", site="reddit.com")])
    links = [{"title": "IRS page", "issuer": "IRS", "url": "https://irs.gov/x"}]
    at = run_direct(monkeypatch, PipelineResult("abstained", ans.summary, ans, Persona(), links=links))
    text = " ".join(m.value for m in at.markdown)
    assert "[Moving my 401k](https://reddit.com/r/nri/1) on reddit.com" in text
    assert "[IRS page (IRS)](https://irs.gov/x)" in text and "Official sources for the US side" in text


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


def test_visitors_do_not_see_the_run_trace_or_model_name(monkeypatch):
    result = canned()
    result.trace, result.model_used = [("pii_guard", "passed"), ("scope_guard", "in_scope")], "groq:some-model"
    at = run_direct(monkeypatch, result)
    assert not any("How this answer was produced" in e.label for e in at.expander)
    assert not any("Answered by" in c.value for c in at.caption)


def test_run_trace_can_be_switched_on_for_developers(monkeypatch):
    import config
    monkeypatch.setattr(config, "SHOW_RUN_TRACE", True)
    result = canned()
    result.trace, result.model_used = [("pii_guard", "passed")], "groq:some-model"
    at = run_direct(monkeypatch, result)
    assert any("How this answer was produced" in e.label for e in at.expander)


def test_no_insights_tab_or_setup_instructions_without_collected_data(monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.tabs
    assert not any("collect.py" in i.value for i in at.info)


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


def prepared_item():
    from copilot.prepared import PreparedAnswer
    return PreparedAnswer("q1", "What happens to my NRO account?", canned().answer, "2026-11-02")


def test_common_questions_open_a_prepared_answer_without_running_the_pipeline(monkeypatch):
    from copilot import prepared
    monkeypatch.setattr(prepared, "available", lambda: [prepared_item()])
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert [b.label for b in at.sidebar.button] == ["What happens to my NRO account?"]
    at.sidebar.button[0].click().run()
    assert not at.exception and "1. Your roadmap" in headers(at)
    assert any("Prepared answer, based on the official documents as of 2 November 2026" in c.value for c in at.caption)


THREAD = {"url": "https://reddit.com/r/nri/9", "title": "Fresh thread", "snippet": "new", "date": "",
          "site": "reddit.com"}


def open_prepared(monkeypatch, fetch):
    from copilot import prepared, tools
    monkeypatch.setattr(prepared, "available", lambda: [prepared_item()])
    monkeypatch.setattr(tools, "fetch_community", fetch)
    monkeypatch.setattr(llm, "structured_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model")))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    return at, " ".join(m.value for m in at.markdown)


def test_prepared_answers_show_live_discussions_in_place_of_the_saved_ones(monkeypatch):
    at, text = open_prepared(monkeypatch, lambda q: [THREAD])
    assert not at.exception and "[Fresh thread](https://reddit.com/r/nri/9) on reddit.com" in text
    assert "Forum post" not in text  # the discussions saved with the answer are replaced
    assert "3. Community experience (not official)" in headers(at)


def test_when_the_live_search_fails_the_saved_discussions_are_shown(monkeypatch):
    def down(q):
        raise RuntimeError("search unavailable")
    at, text = open_prepared(monkeypatch, down)
    assert not at.exception and "Forum post" in text and "Fresh thread" not in text


def test_when_the_live_search_finds_nothing_the_saved_discussions_are_shown(monkeypatch):
    _, text = open_prepared(monkeypatch, lambda q: [])
    assert "Forum post" in text


def test_a_prepared_question_is_searched_once_for_all_visitors(monkeypatch):
    searches = []
    from copilot import prepared, tools
    monkeypatch.setattr(prepared, "available", lambda: [prepared_item()])
    monkeypatch.setattr(tools, "fetch_community", lambda q: searches.append(q) or [THREAD])
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()
    at.sidebar.button[0].click().run()
    assert len(searches) == 1


def test_no_common_questions_means_no_sidebar_list(monkeypatch):
    from copilot import prepared
    monkeypatch.setattr(prepared, "available", lambda: [])
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.sidebar.button


def test_a_live_answer_is_not_labelled_as_prepared(monkeypatch):
    from copilot import prepared
    monkeypatch.setattr(prepared, "available", lambda: [])
    at = run_direct(monkeypatch, canned())
    assert not any("Prepared answer" in c.value for c in at.caption)


def test_the_page_shows_no_stray_values_or_internal_wording():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.json and not any(c.value == "None" for c in at.code)
    page = " ".join([*(m.value for m in at.markdown), *(c.value for c in at.caption), *(i.value for i in at.info)])
    for internal in (".py", ".env", "guard", "groq", "gemini", "openrouter", "threshold", "pytest"):
        assert internal not in page.lower()


def test_choice_questions_offer_a_write_your_own_box():
    at = AppTest.from_file(APP, default_timeout=30).run()
    def next_question():
        next(b for b in at.button if b.label == "Next").click().run()
    assert any("Something else" in t.label for t in at.text_input)          # question 1
    for _ in range(5):
        next_question()
    assert any("Something else" in t.label for t in at.text_input)          # question 6: accounts


def test_unknown_move_date_is_not_filled_in_with_a_default():
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.button if b.label == "Next").click().run()
    assert [s.value for s in at.selectbox] == ["Not sure", "Not sure"]


def test_a_session_cannot_exceed_the_question_limit(monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_QUESTIONS_PER_SESSION", 1)
    monkeypatch.setattr(config, "MIN_SECONDS_BETWEEN_QUESTIONS", 0)
    calls = []
    monkeypatch.setattr(llm, "llm_available", lambda: True)
    monkeypatch.setattr(orchestrator, "run_pipeline",
                        lambda q, p=None, community=True, on_step=None: calls.append(q) or canned())
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
                        lambda q, p=None, community=True, on_step=None: calls.append(q) or canned())
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
    assert any("temporarily unavailable" in e.value for e in at.error)


def test_guided_mode_walks_through_eight_questions():
    at = AppTest.from_file(APP, default_timeout=30).run()
    def click(label):
        next(b for b in at.button if b.label == label).click().run()

    for expected in range(1, 10):
        assert any(f"Question {expected} of 9" in str(p.text) for p in at.get("progress"))
        click("Next")
    assert any(b.label == "Get my roadmap" for b in at.button)


def test_guided_back_button_returns_to_previous_question():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not any(b.label == "Back" for b in at.button)  # nothing before the first question
    next(b for b in at.button if b.label == "Next").click().run()
    next(b for b in at.button if b.label == "Back").click().run()
    assert any("Question 1 of 9" in str(p.text) for p in at.get("progress"))


def test_switching_mode_clears_the_previous_result(monkeypatch):
    at = run_direct(monkeypatch, canned())
    assert "1. Your roadmap" in headers(at)
    at.radio(key="mode").set_value("Guided").run()
    assert "1. Your roadmap" not in headers(at)
