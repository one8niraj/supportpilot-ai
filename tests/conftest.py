"""Shared fixtures. No test in this suite makes a real network/API call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

from src.config import Settings
from src.schemas import ModelOutput


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Make sure a developer's real key/.env can never leak into tests."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SUPPORTPILOT_MODEL", "SUPPORTPILOT_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    # Stop python-dotenv from loading a local .env during tests.
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: False)


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="test-key-not-real", model="test-model")


def make_output(**overrides) -> ModelOutput:
    data = dict(
        category="billing",
        priority="high",
        summary="Customer reports being charged twice and an unanswered refund request.",
        recommended_action="Ask for the order number and the dates of both charges, then check billing.",
        draft_reply=(
            "Hello, I'm sorry about the double charge and the lack of response. "
            "Could you share your order number and the dates of both charges so we can look into it?"
        ),
        escalation_reason=None,
    )
    data.update(overrides)
    return ModelOutput(**data)


def make_response(
    text: str | None = None,
    output: ModelOutput | None = None,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
    block_reason: types.BlockedReason | None = None,
    no_candidates: bool = False,
) -> types.GenerateContentResponse:
    """Build a real SDK response object (not a loose mock)."""
    if text is None and output is not None:
        text = output.model_dump_json()
    feedback = types.GenerateContentResponsePromptFeedback(block_reason=block_reason) if block_reason else None
    if no_candidates:
        return types.GenerateContentResponse(candidates=[], prompt_feedback=feedback)
    parts = [types.Part(text=text)] if text is not None else []
    candidate = types.Candidate(
        content=types.Content(role="model", parts=parts),
        finish_reason=finish_reason,
    )
    return types.GenerateContentResponse(candidates=[candidate], prompt_feedback=feedback)


class FakeClient:
    """Minimal stand-in for ``genai.Client`` exposing models.generate_content."""

    def __init__(self, response=None, error=None):
        self.calls: list[dict] = []
        self._response = response
        self._error = error

        def generate_content(**kwargs):
            self.calls.append(kwargs)
            if self._error is not None:
                raise self._error
            return self._response

        self.models = SimpleNamespace(generate_content=generate_content)
