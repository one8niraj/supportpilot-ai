"""Core service: validate input, call Gemini, validate output.

Nothing here has side effects beyond one read-only API call. Customer message
text is never logged or stored.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from src.config import MAX_MESSAGE_CHARS, Settings, get_settings
from src.errors import (
    ConfigurationError,
    InvalidInputError,
    MalformedResponseError,
    ProviderError,
)
from src.prompts import SYSTEM_PROMPT, build_user_content
from src.schemas import ModelOutput, TriageResult

logger = logging.getLogger(__name__)

# Gemini "thinking" models count internal reasoning tokens against this limit,
# so it is deliberately generous. Visible output is limited by the schema/prompt.
MAX_OUTPUT_TOKENS = 4096

# Phrases in a draft reply that suggest the model claimed an action it cannot
# perform or promised an outcome/deadline. This is a heuristic safety net for
# the human reviewer, not a guarantee.
_UNSUPPORTED_CLAIM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "claims a refund or credit was issued or approved",
        re.compile(
            r"\b(?:we(?:'ve| have)|i(?:'ve| have))\s+(?:already\s+)?"
            r"(?:issued|processed|approved|sent|credited|refunded)\b"
            r"|\b(?:refund|credit|reimbursement)\s+(?:has|have|was|were)\s+been\s+"
            r"(?:issued|processed|approved|sent)\b"
            r"|\byour\s+refund\s+(?:is|has been)\s+(?:approved|on its way)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "claims an account change was made",
        re.compile(
            r"\byour\s+account\s+(?:has|have)\s+been\s+"
            r"(?:reset|restored|unlocked|closed|deleted|updated|reactivated|secured)\b"
            r"|\b(?:we(?:'ve| have)|i(?:'ve| have))\s+(?:reset|restored|unlocked|"
            r"closed|deleted|reactivated|secured)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "promises a specific timeframe or guaranteed outcome",
        re.compile(
            r"\bwithin\s+(?:the\s+next\s+)?\d+\s*(?:business\s+)?"
            r"(?:minutes?|hours?|days?|weeks?)\b"
            r"|\b(?:guarantee|guaranteed|definitely|for sure)\b"
            r"|\b\d+\s*(?:-|to)\s*\d+\s*(?:business\s+)?(?:hours?|days?|weeks?)\b",
            re.IGNORECASE,
        ),
    ),
]


def validate_message(message: Optional[str], max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Return the stripped message or raise ``InvalidInputError``."""
    if message is None or not message.strip():
        raise InvalidInputError("Please enter a customer message to analyze.")
    cleaned = message.strip()
    if len(cleaned) > max_chars:
        raise InvalidInputError(
            f"The message is {len(cleaned)} characters long. "
            f"The limit is {max_chars} characters."
        )
    return cleaned


def find_unsupported_claims(draft_reply: str) -> list[str]:
    """Return human-readable warnings for risky phrases in a draft reply."""
    return [label for label, pattern in _UNSUPPORTED_CLAIM_PATTERNS if pattern.search(draft_reply)]


def _build_client(settings: Settings) -> genai.Client:
    return genai.Client(
        api_key=settings.api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.timeout_seconds * 1000),  # SDK expects milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=settings.max_retries + 1, initial_delay=0.5, max_delay=2.0
            ),
        ),
    )


def _translate_api_error(exc: genai_errors.APIError) -> ProviderError | ConfigurationError:
    """Map Gemini SDK errors to safe messages. Raw error text is never exposed."""
    code = getattr(exc, "code", None)
    # Log only the exception type and status code, never the message body.
    logger.warning("Gemini call failed: %s (code=%s)", type(exc).__name__, code)

    if code == 400:
        # Gemini reports an invalid API key as HTTP 400. The message text is
        # inspected only to classify the error and is never shown or logged.
        if "api key" in str(getattr(exc, "message", "") or "").lower():
            return ConfigurationError(
                "The AI service rejected the API key. Check GEMINI_API_KEY."
            )
        return ConfigurationError(
            "The AI service rejected the request. The configured model may not support "
            "structured output. Check SUPPORTPILOT_MODEL."
        )
    if code in (401, 403):
        return ConfigurationError(
            "The AI service rejected the API key or this account lacks access. "
            "Check GEMINI_API_KEY."
        )
    if code == 404:
        return ConfigurationError(
            "The configured model was not found or is not available to this key. "
            "Check SUPPORTPILOT_MODEL."
        )
    if code == 429:
        return ProviderError(
            "The AI service is rate-limiting requests or your quota is exhausted. "
            "Wait a moment and try again, or check your plan and quota."
        )
    if code is not None and code >= 500:
        return ProviderError(
            "The AI service is temporarily unavailable or overloaded. Please try again."
        )
    return ProviderError("The AI service returned an error. Please try again later.")


def _extract_json_text(response: Any) -> str:
    """Return the model's JSON text, or raise MalformedResponseError."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        logger.warning("Prompt was blocked by the provider")
        raise MalformedResponseError(
            "The AI declined to analyze this message. A human should review it directly."
        )

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise MalformedResponseError("The AI service returned an empty response.")

    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason == types.FinishReason.MAX_TOKENS:
        logger.warning("Model response was truncated")
        raise MalformedResponseError("The AI response was cut off. Please try again.")
    if finish_reason not in (None, types.FinishReason.STOP, types.FinishReason.FINISH_REASON_UNSPECIFIED):
        logger.warning("Model response ended with finish_reason=%s", finish_reason)
        raise MalformedResponseError(
            "The AI response was blocked or incomplete. A human should review the message directly."
        )

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise MalformedResponseError("The AI response was empty. Please try again.")
    return text


def analyze_message(
    message: str,
    settings: Optional[Settings] = None,
    client: Optional[genai.Client] = None,
) -> TriageResult:
    """Analyze one customer message and return a validated ``TriageResult``.

    Args:
        message: raw customer text (untrusted).
        settings: optional override; otherwise read from the environment.
        client: optional pre-built client (used by tests to inject a fake).

    Raises:
        InvalidInputError, ConfigurationError, ProviderError, MalformedResponseError
    """
    cleaned = validate_message(message)

    if settings is None:
        settings = get_settings()
    if client is None:
        client = _build_client(settings)

    try:
        response = client.models.generate_content(
            model=settings.model,
            contents=build_user_content(cleaned),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=ModelOutput.model_json_schema(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
    except genai_errors.APIError as exc:
        raise _translate_api_error(exc) from None
    except httpx.TimeoutException:
        logger.warning("Gemini call timed out")
        raise ProviderError("The AI service timed out. Please try again.") from None
    except httpx.HTTPError as exc:
        logger.warning("Gemini call failed: %s", type(exc).__name__)
        raise ProviderError(
            "Could not reach the AI service. Check your network connection and try again."
        ) from None

    text = _extract_json_text(response)

    try:
        parsed = ModelOutput.model_validate_json(text)
        # Re-validate through the strict model; also catches over-long fields.
        return TriageResult.model_validate(parsed.model_dump())
    except ValidationError:
        logger.warning("Model output failed validation")
        raise MalformedResponseError(
            "The AI response was not in the expected format or did not meet quality "
            "checks (for example, a field was empty or too long). Please try again."
        ) from None
