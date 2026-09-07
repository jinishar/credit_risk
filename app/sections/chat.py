"""Ask-the-data section: NL question → validated SQL → readable answer."""
from __future__ import annotations

import streamlit as st

from app import data, ui
from src.talk_to_data.prompt_templates import CANONICAL_PATTERNS
from src.talk_to_data.query_runner import ChatSession
from src.utils.config import GEMINI_API_KEY


def _md_safe(text: str) -> str:
    # `$` pairs are read as LaTeX math by Streamlit markdown; dollar amounts in
    # the narration ("$169,078 ... $165,612") would render as italic math.
    return (text or "").replace("$", r"\$")


def render() -> None:
    ui.section_header(
        "Ask the Data",
        "Plain-English questions answered with validated, read-only SQL over the applications table.",
        kicker="Talk to the data",
    )

    if not GEMINI_API_KEY:
        ui.callout(
            "No <code>GEMINI_API_KEY</code> set — the five sample questions below still run "
            "end to end; free-form questions need a key."
        )

    # Guard, not setdefault: ChatSession holds conversation history, so it must
    # be built once per session. It reuses the app's cached DuckDB connection
    # rather than re-loading the CSV into a new one.
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(connection=data.duckdb_connection())
    st.session_state.setdefault("chat_log", [])

    with ui.panel("Sample questions", "One click — pre-verified SQL, no LLM call."):
        cols = st.columns(len(CANONICAL_PATTERNS))
        for col, pattern in zip(cols, CANONICAL_PATTERNS):
            if col.button(pattern["question"], key=f"sample_{pattern['question']}",
                          use_container_width=True):
                st.session_state["pending_question"] = pattern["question"]

    question = st.chat_input("Ask about the applicant data…") or st.session_state.pop("pending_question", None)
    if question:
        with st.spinner("Thinking…"):
            st.session_state.chat_log.append(st.session_state.chat_session.ask(question))

    for turn in reversed(st.session_state.chat_log):
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            if turn.error:
                st.error(_md_safe(turn.narrative))
            else:
                st.write(_md_safe(turn.narrative))
                with st.expander("SQL used"):
                    st.code(turn.sql, language="sql")
                if not turn.result.empty:
                    st.dataframe(turn.result, hide_index=True, use_container_width=True)
