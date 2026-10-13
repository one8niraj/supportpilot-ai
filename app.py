"""SupportPilot: Streamlit UI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.config import MAX_MESSAGE_CHARS
from src.errors import SupportPilotError
from src.schemas import TriageResult
from src.service import analyze_message, find_unsupported_claims, validate_message

EXAMPLE_MESSAGE = (
    "I was charged twice for my order, and nobody has responded to my refund request."
)

st.set_page_config(page_title="SupportPilot", page_icon="🛟", layout="centered")


def _load_example() -> None:
    st.session_state["message"] = EXAMPLE_MESSAGE


def _run_analysis(message: str) -> None:
    """Validate, call the service, and store the result (or a safe error)."""
    st.session_state["error"] = None
    st.session_state["result"] = None
    try:
        validate_message(message)
        with st.spinner("Analyzing message..."):
            result = analyze_message(message)
    except SupportPilotError as exc:
        st.session_state["error"] = exc.user_message
        return
    except Exception:  # noqa: BLE001 - last-resort guard; details are not shown
        st.session_state["error"] = (
            "Something unexpected went wrong. Please try again."
        )
        return
    st.session_state["result"] = result
    # Seed the editable draft before the text_area widget is created below.
    st.session_state["draft_reply"] = result.draft_reply


def _render_result(result: TriageResult) -> None:
    st.divider()
    st.subheader("Analysis")

    col_category, col_priority = st.columns(2)
    col_category.metric("Category", result.category.value.title())
    col_priority.metric("Priority", result.priority.value.title())

    if result.needs_escalation:
        st.warning(f"**Escalation suggested:** {result.escalation_reason}")

    st.markdown("**Issue summary**")
    st.write(result.summary)

    st.markdown("**Recommended next action**")
    st.write(result.recommended_action)

    st.markdown("**Draft customer reply** (editable)")
    claim_warnings = find_unsupported_claims(st.session_state.get("draft_reply", ""))
    for label in claim_warnings:
        st.error(
            f"Reviewer check: the draft may contain text that {label}. "
            "SupportPilot cannot verify or perform such actions. Edit before use."
        )
    st.text_area(
        "Draft customer reply",
        key="draft_reply",
        height=260,
        label_visibility="collapsed",
    )

    st.info(
        "**Human review required.** This is an AI-generated draft. Check it for "
        "accuracy and tone before using it. **No action has been taken:** no reply "
        "was sent, and no refund, account change or payment was made."
    )


def main() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("message", "")

    st.title("SupportPilot")
    st.caption(
        "AI-powered support triage copilot. Paste a customer message to get a "
        "category, priority, summary, recommended next step and a draft reply "
        "for a human agent to review."
    )
    st.caption(
        "Privacy: the message text is sent to the Google Gemini API for analysis. "
        "Use fictional or properly redacted data only. Nothing is saved by this app."
    )

    message = st.text_area(
        "Customer message",
        key="message",
        height=180,
        placeholder="Paste the customer's message here...",
    )
    length = len(message.strip())
    over_limit = length > MAX_MESSAGE_CHARS
    st.caption(f"{length} / {MAX_MESSAGE_CHARS} characters")
    if over_limit:
        st.error(f"The message is over the {MAX_MESSAGE_CHARS}-character limit.")

    col_analyze, col_example = st.columns([1, 1])
    analyze_clicked = col_analyze.button(
        "Analyze message", type="primary", use_container_width=True
    )
    col_example.button(
        "Load fictional example", on_click=_load_example, use_container_width=True
    )

    if analyze_clicked:
        _run_analysis(message)

    if st.session_state["error"]:
        st.error(st.session_state["error"])

    result = st.session_state["result"]
    if result is not None:
        _render_result(result)


main()
