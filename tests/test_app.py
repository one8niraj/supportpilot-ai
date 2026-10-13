"""UI smoke tests using Streamlit's AppTest with the service mocked out."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.errors import ProviderError
from src.schemas import TriageResult
from tests.conftest import make_output

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _result(**overrides) -> TriageResult:
    return TriageResult.model_validate(make_output(**overrides).model_dump())


def test_initial_render_has_title_and_no_result():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert at.title[0].value == "SupportPilot"
    assert len(at.error) == 0


def test_empty_message_shows_error(monkeypatch):
    at = AppTest.from_file(APP).run()
    at.button[0].click().run()
    assert any("enter a customer message" in e.value.lower() for e in at.error)


def test_successful_analysis_renders_result_and_notice(monkeypatch):
    monkeypatch.setattr("src.service.analyze_message", lambda message: _result())
    at = AppTest.from_file(APP).run()
    at.text_area(key="message").input("I was charged twice.").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("No action has been taken" in i.value for i in at.info)
    assert at.text_area(key="draft_reply").value.startswith("Hello")


def test_escalation_warning_shown(monkeypatch):
    result = _result(escalation_reason="Possible fraud; send to security team.")
    monkeypatch.setattr("src.service.analyze_message", lambda message: result)
    at = AppTest.from_file(APP).run()
    at.text_area(key="message").input("Someone used my card.").run()
    at.button[0].click().run()
    assert any("Escalation suggested" in w.value for w in at.warning)


def test_provider_error_is_shown_safely(monkeypatch):
    def boom(message):
        raise ProviderError("The AI service timed out. Please try again.")

    monkeypatch.setattr("src.service.analyze_message", boom)
    at = AppTest.from_file(APP).run()
    at.text_area(key="message").input("Hello").run()
    at.button[0].click().run()
    assert any("timed out" in e.value for e in at.error)
    assert not at.exception


def test_unexpected_exception_hides_details(monkeypatch):
    def boom(message):
        raise RuntimeError("internal detail sk-LEAK")

    monkeypatch.setattr("src.service.analyze_message", boom)
    at = AppTest.from_file(APP).run()
    at.text_area(key="message").input("Hello").run()
    at.button[0].click().run()
    assert any("unexpected" in e.value.lower() for e in at.error)
    assert all("sk-LEAK" not in e.value for e in at.error)


def test_risky_draft_gets_reviewer_warning(monkeypatch):
    risky = _result(
        draft_reply="Good news, we have refunded your order in full. Thanks for your patience."
    )
    monkeypatch.setattr("src.service.analyze_message", lambda message: risky)
    at = AppTest.from_file(APP).run()
    at.text_area(key="message").input("Where is my refund?").run()
    at.button[0].click().run()
    assert any("Reviewer check" in e.value for e in at.error)
