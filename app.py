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
from copilot import llm, orchestrator
from copilot.persona import (ACCOUNT_OPTIONS, ASSET_OPTIONS, GOAL_OPTIONS, RELOCATION_OPTIONS,
                             persona_from_intake)
from copilot.schemas import DISCLAIMER
from copilot.tools import site_of
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

# One question per screen.
STEPS = ["relocation", "move_date", "countries", "days", "accounts", "assets", "goals", "concern"]


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


def render_result(result) -> None:
    if result.kind == "pii_blocked":
        st.error(result.message)
        return
    if result.kind == "refused":
        st.info(result.message)
        return
    if result.kind == "us_side_tax":
        st.warning(result.message)
        for link in result.links:
            st.markdown(f"- [{link['title']} ({link['issuer']})]({link['url']})")
    ans = result.answer
    if result.kind == "abstained":
        st.warning(result.message)
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
        else:
            st.write("No community threads were found or community search is not configured.")

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
        st.error("No language model is configured. Set at least one provider key in the environment "
                 "(see `.env.example`) and restart the app.")
        return
    if not within_usage_limits():
        return
    with st.spinner("Searching the official documents..."):
        try:
            st.session_state["result"] = orchestrator.run_pipeline(
                question, persona, community=st.session_state.get("community", True))
        except Exception:
            st.session_state["result"] = None
            st.error("Something went wrong while answering. Please try again in a moment.")


def guided_mode() -> None:
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("answers", {})
    a = ss["answers"]
    if ss["step"] >= len(STEPS):
        question = st.text_input("Your question (optional)", value=DEFAULT_QUESTION)
        if st.button("Get my roadmap"):
            run(question, persona_from_intake(a))
        if st.button("Start over"):
            ss["step"], ss["answers"], ss["result"] = 0, {}, None
            st.rerun()
        return

    name = STEPS[ss["step"]]
    st.progress((ss["step"] + 1) / len(STEPS), text=f"Question {ss['step'] + 1} of {len(STEPS)}")
    if name == "relocation":
        a["relocation"] = st.radio("Are you planning to relocate to India, or have you already?",
                                   list(RELOCATION_OPTIONS), key="w_relocation")
    elif name == "move_date":
        st.write("When did you or will you move?")
        c1, c2 = st.columns(2)
        a["move_month"] = MONTHS.index(c1.selectbox("Month", MONTHS, key="w_month")) + 1
        a["move_year"] = c2.number_input("Year", 2000, 2100, 2026, key="w_year")
    elif name == "countries":
        a["countries"] = st.text_input("Which countries have you lived in recently? (comma separated)",
                                       key="w_countries").split(",")
    elif name == "days":
        st.write("Roughly how many days did you spend in India in each of the last few years? (optional)")
        a["days"] = {}
        for year in recent_years():
            d = st.number_input(str(year), 0, 366, 0, key=f"w_days_{year}")
            if d:
                a["days"][year] = d
    elif name == "accounts":
        a["accounts"] = st.multiselect("Which Indian accounts do you hold?", list(ACCOUNT_OPTIONS), key="w_accounts")
    elif name == "assets":
        a["assets"] = st.multiselect("Which foreign assets do you hold?", list(ASSET_OPTIONS), key="w_assets")
    elif name == "goals":
        a["goals"] = st.multiselect("What do you want to do with them?", list(GOAL_OPTIONS), key="w_goals")
    elif name == "concern":
        a["concern"] = st.text_input("What is your biggest concern right now? (short, no personal details)",
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


def insights_tab() -> None:
    st.subheader("Pain points of returning NRIs")
    df = load()
    agg = aggregate(df)
    if agg["n"] == 0:
        st.info("No community threads have been collected and tagged yet. "
                "Run `analytics/collect.py` then `analytics/tag.py`.")
        return
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


@st.cache_resource(show_spinner="Preparing the document index. This only happens on the first start.")
def ensure_index() -> None:
    """A fresh checkout has no search index (it is built from the PDFs, not stored in git)."""
    if not config.CHROMA_DIR.exists():
        build_index()


def main() -> None:
    st.set_page_config(page_title="NRI Return Copilot", page_icon="🧭")
    ensure_index()
    st.title("NRI Return Copilot")
    st.caption("Which Indian rules and paperwork apply when you return? Answers come only from official "
               "documents, with the exact page.")
    st.info(PRIVACY)
    tab_main, tab_insights = st.tabs(["Copilot", "Insights"])
    with tab_main:
        mode = st.radio("How would you like to start?", ["Guided", "Ask directly"], horizontal=True, key="mode",
                        on_change=lambda: st.session_state.update(result=None))
        st.checkbox("Include community experiences (shown separately, not official)", value=True,
                    key="community")
        guided_mode() if mode == "Guided" else direct_mode()
        if st.session_state.get("result") is not None:
            st.divider()
            render_result(st.session_state["result"])
            render_trace(st.session_state["result"])
    with tab_insights:
        insights_tab()
    footer()


main()
