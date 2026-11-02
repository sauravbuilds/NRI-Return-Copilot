"""NRI Return Copilot: Streamlit app.  Run:  streamlit run app.py

Two modes (Guided intake, Ask directly) feed the same pipeline. Results show four
sections in a fixed order: roadmap, what could not be determined, community
(not official), and the CA brief.
"""
import time
from datetime import date

import streamlit as st

import config
from analytics.chart import NOTE, aggregate, load
from copilot import llm, orchestrator, retrieval
from copilot import prepared
from copilot.orchestrator import PipelineResult
from copilot.persona import (ACCOUNT_OPTIONS, ASSET_OPTIONS, CITIZENSHIP_OPTIONS, GOAL_OPTIONS,
                             RELOCATION_OPTIONS, persona_from_intake)
from copilot.schemas import DISCLAIMER, Persona
from copilot.tools import fetch_community, site_of
from scripts.ingest import build_index
from copilot.sources import describe_citation, footer_lines, pretty_date

PRIVACY = ("Just share the context. Please don't type phone numbers, ID numbers or account numbers, "
           "and don't upload documents.")
STAGE_TITLES = {"before_you_move": "Before you move", "first_30_days": "First 30 days", "first_year": "First year"}
BADGES = {"agrees": "✅ Agrees with official text", "contradicts": "⚠️ Contradicts official text",
          "unrelated": "➖ Unrelated to the official text"}
UPDATE_BADGES = {"agrees": "✅ Matches what the documents say", "changes": "⚠️ May change this answer",
                 "unchecked": "❔ Not compared yet", "unrelated": "➖ Unrelated"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
DEFAULT_QUESTION = "Which Indian rules and paperwork apply to my situation?"
NOT_SURE = "Not sure"
OTHER_PROMPT = "Something else? Write it here (optional)"

# One question per screen, each with a line on why it is asked.
STEPS = ["relocation", "move_date", "citizenship", "countries", "days", "accounts", "assets", "goals", "concern"]
WHY_WE_ASK = {
    "relocation": "The rules differ depending on whether you have already moved or are still planning to.",
    "move_date": "Most of the rules start counting from the date you become a resident.",
    "citizenship": "Some rules differ for Indian citizens, OCI cardholders and PIO cardholders.",
    "countries": "The countries you have lived in or hold assets in. For example: USA, Canada.",
    "days": "Residential status depends on the days spent in India. Rough numbers are fine. "
            "Leave a year at 0 if you don't know.",
    "accounts": "NRE, NRO and resident accounts are treated differently once you return.",
    "assets": "Different assets come with different reporting and transfer rules.",
    "goals": "So the roadmap can focus on what you actually want to do.",
    "concern": "In your own words. Please leave out names and numbers.",
}


def recent_years(count: int = 3) -> list[int]:
    """The last few full calendar years, most recent first."""
    return [date.today().year - i for i in range(1, count + 1)]


def render_updates(updates) -> None:
    """Newer official publications, shown above the answer so they are seen first."""
    st.warning("**Heads up:** the official sites have published something newer than the documents "
               "this answer is based on. I haven't read these in full, so please open them and check.")
    for u in updates:
        with st.container(border=True):
            st.markdown(f"**{UPDATE_BADGES[u.verdict]}**")
            st.markdown(f"[{u.title or u.url}]({u.url})" + (f"  \nPublished {pretty_date(u.date)}" if u.date else ""))
            st.write(u.snippet)
            st.caption(u.reason)


def render_extras(links: list, community_links: list) -> None:
    """Official sources for a question's US side, and community discussions to read when there
    is no official answer to compare them with."""
    if links:
        st.markdown("**Official sources for the US side**")
        for link in links:
            st.markdown(f"- [{link['title']} ({link['issuer']})]({link['url']})")
    if community_links:
        st.markdown("**Community discussions you can read (not official)**")
        st.caption("Personal experiences on Reddit, X and Quora. They are not authoritative, but people "
                   "in the same situation may be able to help.")
        render_community_links(community_links)


def render_community_links(community_links: list) -> None:
    for c in community_links:
        with st.container(border=True):
            st.markdown(f"[{c.title or c.url}]({c.url}) on {c.site or site_of(c.url)}")
            if c.snippet:
                st.write(c.snippet)


def render_result(result) -> None:
    if result.kind == "pii_blocked":
        st.error(result.message)
        return
    if result.kind == "refused":
        st.info(result.message)
        return
    if result.kind in ("us_side_tax", "abstained"):
        st.warning(result.message)
    ans = result.answer
    if ans is None:
        return
    if ans.needs_human_review:
        st.warning("**Please review this with a chartered accountant.** Why: " + "; ".join(ans.review_reasons))

    if result.kind == "answered":
        if ans.official_updates:
            render_updates(ans.official_updates)
        st.subheader("1. Your roadmap")
        st.write(ans.summary)
        st.caption(f"Confidence: {ans.confidence}. {ans.confidence_reason}")
        for stage, title in STAGE_TITLES.items():
            steps = [s for s in ans.roadmap if s.stage == stage]
            if steps:
                st.markdown(f"**{title}**")
                for s in steps:
                    st.markdown(f"- {s.step}  \n  Source: " + "; ".join(describe_citation(c) for c in s.citations))
        if ans.official_findings:
            st.markdown("**What the official documents say**")
            for f in ans.official_findings:
                st.markdown(f"- {f.statement}  \n  Source: " + "; ".join(describe_citation(c) for c in f.citations))

        st.subheader("2. What I could not determine")
        if ans.missing_information:
            for m in ans.missing_information:
                st.markdown(f"- **{m.field.replace('_', ' ').capitalize()}**: {m.why_it_matters}  \n"
                            f"  *If this changes:* {m.if_this_then_that}")
        else:
            st.write("Nothing important was missing from your context.")

        st.subheader("3. Community experience (not official)")
        st.caption("What people are saying on Reddit, X and Quora. Handy for real experiences, but not "
                   "official. Each link opens the conversation.")
        if ans.community_findings:
            for c in ans.community_findings:
                with st.container(border=True):
                    st.markdown(f"**{BADGES[c.verdict]}**")
                    st.markdown(f"[{c.title or c.url}]({c.url}) on {site_of(c.url)}")
                    st.write(c.snippet)
                    st.caption(c.reason)
        if ans.community_links:
            render_community_links(ans.community_links)
        if not ans.community_findings and not ans.community_links:
            st.write("No community threads were found or community search is not configured.")

    # An answered result shows its community links inside section 3.
    render_extras(result.links, [] if result.kind == "answered" else ans.community_links)

    if ans.ca_brief:
        st.subheader("4. Take this to a CA")
        st.download_button("Download CA brief", ans.ca_brief, file_name="ca_brief.txt")
        with st.expander("Preview the brief"):
            st.text(ans.ca_brief)


def render_trace(result) -> None:
    """Show which steps ran. Contains guard outcomes and tool names only, never raw input."""
    if not result.trace:
        return
    with st.expander("How this answer was produced"):
        for step, detail in result.trace:
            st.write(f"- **{step.replace('_', ' ')}**: {detail}")
        if result.model_used:
            st.caption(f"Answered by {result.model_used}")


def within_usage_limits() -> bool:
    """Per-session cap and pause between questions, so one visitor cannot use up the shared quota."""
    ss = st.session_state
    if ss.get("asked", 0) >= config.MAX_QUESTIONS_PER_SESSION:
        st.warning("You've reached the question limit for this session. Please come back a little later.")
        return False
    if time.monotonic() - ss.get("last_asked", float("-inf")) < config.MIN_SECONDS_BETWEEN_QUESTIONS:
        st.warning("Give it a few seconds before asking the next question.")
        return False
    ss["asked"], ss["last_asked"] = ss.get("asked", 0) + 1, time.monotonic()
    return True


def run(question: str, persona=None) -> None:
    if not llm.llm_available():
        st.error("The assistant is temporarily unavailable. Please try again in a few minutes.")
        return
    if not within_usage_limits():
        return
    st.session_state["prepared_on"] = None
    with st.status("Working on it", expanded=True) as status:
        try:
            st.session_state["result"] = orchestrator.run_pipeline(
                question, persona, community=st.session_state.get("community", True),
                on_step=lambda message: status.update(label=message))
            status.update(label="Done", state="complete", expanded=False)
        except Exception:
            st.session_state["result"] = None
            status.update(label="Something went wrong", state="error")
            st.error("Something went wrong while answering. Please try again in a moment.")


def with_other(key: str, a: dict) -> None:
    """A free-text box under a multiple-choice question, for anything the options do not cover."""
    a[key] = st.text_input(OTHER_PROMPT, value=a.get(key, ""), key=f"w_{key}")


def pick_one(label: str, options: dict, name: str, a: dict) -> None:
    current = a.get(name)
    a[name] = st.radio(label, list(options), index=list(options).index(current) if current in options else None,
                       key=f"w_{name}")


def guided_mode() -> None:
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("answers", {})
    a = ss["answers"]
    if ss["step"] >= len(STEPS):
        question = st.text_input("Anything specific you'd like to ask? (optional)", value=DEFAULT_QUESTION)
        if st.button("Get my roadmap"):
            run(question, persona_from_intake(a))
        if st.button("Start over"):
            ss["step"], ss["answers"], ss["result"] = 0, {}, None
            st.rerun()
        return

    name = STEPS[ss["step"]]
    st.progress((ss["step"] + 1) / len(STEPS), text=f"Question {ss['step'] + 1} of {len(STEPS)}")
    st.caption(WHY_WE_ASK[name])
    if name == "relocation":
        pick_one("Are you planning to move to India, or have you already?", RELOCATION_OPTIONS, "relocation", a)
        with_other("relocation_other", a)
    elif name == "move_date":
        st.write("When did you or will you move?")
        years = [str(y) for y in range(date.today().year - 10, date.today().year + 4)]
        c1, c2 = st.columns(2)
        month = c1.selectbox("Month", [NOT_SURE] + MONTHS, key="w_month")
        year = c2.selectbox("Year", [NOT_SURE] + years, key="w_year")
        known = month in MONTHS and year != NOT_SURE
        a["move_month"], a["move_year"] = (MONTHS.index(month) + 1, int(year)) if known else (None, None)
    elif name == "citizenship":
        pick_one("Which best describes you?", CITIZENSHIP_OPTIONS, "citizenship", a)
        with_other("citizenship_other", a)
    elif name == "countries":
        a["countries"] = st.text_input("Which countries have you lived in, or hold assets in? (comma separated)",
                                       key="w_countries", placeholder="USA, Canada").split(",")
    elif name == "days":
        st.write("Roughly how many days did you spend in India in each recent year? (optional)")
        a["days"] = {}
        for year in recent_years():
            d = st.number_input(str(year), 0, 366, 0, key=f"w_days_{year}")
            if d:
                a["days"][year] = d
    elif name == "accounts":
        a["accounts"] = st.multiselect("Which accounts do you hold in India?", list(ACCOUNT_OPTIONS),
                                       default=a.get("accounts", []), key="w_accounts")
        with_other("accounts_other", a)
    elif name == "assets":
        a["assets"] = st.multiselect("Which assets do you hold outside India?", list(ASSET_OPTIONS),
                                     default=a.get("assets", []), key="w_assets")
        with_other("assets_other", a)
    elif name == "goals":
        a["goals"] = st.multiselect("What do you want to do with them?", list(GOAL_OPTIONS),
                                    default=a.get("goals", []), key="w_goals")
        with_other("goals_other", a)
    elif name == "concern":
        a["concern"] = st.text_input("What is your biggest concern right now?", value=a.get("concern", ""),
                                     key="w_concern")
    back, forward = st.columns(2)
    if ss["step"] > 0 and back.button("Back"):
        ss["step"] -= 1
        st.rerun()
    if forward.button("Next"):
        ss["step"] += 1
        st.rerun()


def direct_mode() -> None:
    q = st.text_area("Ask your question", placeholder="For example: What happens to my NRO account when I return?",
                     key="direct_question")
    if st.button("Ask") and q.strip():
        run(q.strip())


def common_questions_sidebar() -> None:
    """Questions many returning NRIs ask. Each has an answer prepared from the official
    documents, so it opens at once. Hidden when none are available or the documents have
    changed since they were prepared."""
    items = prepared.available()
    if not items:
        return
    with st.sidebar:
        st.subheader("Common questions")
        st.caption("Select one to see the answer straight away.")
        for item in items:
            if st.button(item.question, key=f"prepared_{item.id}", use_container_width=True):
                answer = item.answer
                threads = live_discussions(item.question)
                if threads:  # fresh discussions replace the ones saved with the answer
                    answer = answer.model_copy(update={"community_findings": [],
                                                       "community_links": orchestrator.as_links(threads)})
                st.session_state["result"] = PipelineResult("answered", answer.summary, answer, Persona())
                st.session_state["prepared_on"] = item.prepared_on


@st.cache_data(ttl=config.PREPARED_THREADS_TTL_SECONDS, show_spinner=False)
def live_discussions(question: str) -> list[dict]:
    """Current community discussions for a prepared question: one search, no model call. The
    result is shared by every visitor for a day, so the search allowance is not used up by
    repeat clicks. Any failure falls back to the discussions saved with the answer."""
    try:
        return fetch_community(question)
    except Exception:
        return []


def insights_tab(agg: dict) -> None:
    st.subheader("Pain points of returning NRIs")
    st.caption(f"Based on {agg['n']} public threads. {NOTE}")
    st.bar_chart(agg["problem_type"])
    c1, c2 = st.columns(2)
    c1.markdown("**By stage**")
    c1.bar_chart(agg["stage"])
    c2.markdown("**Resolved?**")
    c2.bar_chart(agg["resolved"])


def footer() -> None:
    st.divider()
    with st.expander("Sources and when they were last verified"):
        for line in footer_lines():
            st.write("- " + line)
    st.caption(DISCLAIMER)


@st.cache_resource(show_spinner="Getting ready. The first visit loads the search tools and can take a minute or two.")
def prepare_search() -> None:
    """A fresh checkout has no search index (it is built from the PDFs, not stored in git).
    The embedding model is loaded here too, so the first question is not slow."""
    if not config.CHROMA_DIR.exists():
        build_index()
    retrieval.warm_up()


def main() -> None:
    st.set_page_config(page_title="NRI Return Copilot", page_icon="🧭")
    prepare_search()
    st.title("NRI Return Copilot")
    st.caption("Which Indian rules and paperwork apply when you return? Answers come only from official "
               "documents, with the exact page.")
    st.info(PRIVACY)
    common_questions_sidebar()
    insights = aggregate(load())
    # The Insights tab only appears once tagged community threads exist.
    tabs = st.tabs(["Copilot", "Insights"]) if insights["n"] else [st.container()]
    with tabs[0]:
        mode = st.radio("How would you like to start?", ["Guided", "Ask directly"], horizontal=True, key="mode",
                        on_change=lambda: st.session_state.update(result=None, prepared_on=None))
        st.checkbox("Include community experiences (shown separately, not official)", value=True,
                    key="community")
        if mode == "Guided":
            guided_mode()
        else:
            direct_mode()
        if st.session_state.get("result") is not None:
            st.divider()
            if st.session_state.get("prepared_on"):
                st.caption(f"Prepared answer, based on the official documents as of "
                           f"{pretty_date(st.session_state['prepared_on'])}.")
            render_result(st.session_state["result"])
            if config.SHOW_RUN_TRACE:
                render_trace(st.session_state["result"])
    if insights["n"]:
        with tabs[1]:
            insights_tab(insights)
    footer()


main()
