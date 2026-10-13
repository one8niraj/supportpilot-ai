"""Exercise the real google-genai SDK against an in-process fake HTTP transport.

No network and no API key are needed. This checks the request our code really
builds (schema, system instruction, model path) and that real SDK response and
error handling work with our code, which pure fakes of ``generate_content``
cannot show.
"""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from src.errors import ConfigurationError, MalformedResponseError, ProviderError
from src.service import analyze_message
from tests.conftest import make_output


def _gemini_reply(text: str, finish_reason: str = "STOP") -> dict:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish_reason}
        ]
    }


def _client(handler) -> genai.Client:
    return genai.Client(
        api_key="test-key-not-real",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def test_request_shape_and_reply_is_parsed(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["api_key_header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=_gemini_reply(make_output().model_dump_json()))

    result = analyze_message("I was charged twice.", settings=settings, client=_client(handler))
    assert result.category.value == "billing"

    assert "test-model" in seen["url"] and "generateContent" in seen["url"]
    assert seen["api_key_header"] == "test-key-not-real"  # sent as header, not in the URL
    assert "test-key-not-real" not in seen["url"]

    body = seen["body"]
    gen_config = body["generationConfig"]
    assert gen_config["responseMimeType"] == "application/json"
    schema = gen_config["responseJsonSchema"]  # plain JSON schema, camelCase key
    schema_text = json.dumps(schema)
    assert "draft_reply" in schema_text and "escalation_reason" in schema_text
    assert "responseSchema" not in gen_config  # SDK's pydantic->Schema conversion is not used
    for forbidden in ("additionalProperties", "additional_properties", "property_ordering", "$ref"):
        assert forbidden not in schema_text
    assert "systemInstruction" in body
    user_text = body["contents"][0]["parts"][0]["text"]
    assert user_text.startswith("<customer_message>")


def test_invalid_enum_from_model_is_rejected(settings):
    bad = make_output().model_dump_json().replace("billing", "complaint")
    handler = lambda request: httpx.Response(200, json=_gemini_reply(bad))
    with pytest.raises(MalformedResponseError):
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))


def test_non_json_content_is_rejected(settings):
    handler = lambda request: httpx.Response(200, json=_gemini_reply("Sure! Here is my analysis."))
    with pytest.raises(MalformedResponseError):
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))


def test_truncated_reply_is_rejected(settings):
    handler = lambda request: httpx.Response(
        200, json=_gemini_reply('{"category": "bill', finish_reason="MAX_TOKENS")
    )
    with pytest.raises(MalformedResponseError):
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))


def test_prompt_blocked_by_provider_is_reported(settings):
    handler = lambda request: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(MalformedResponseError):
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))


def test_http_400_bad_key_is_translated_without_leaking_body(settings):
    body = {"error": {"code": 400, "message": "API key not valid. sk-LEAK", "status": "INVALID_ARGUMENT"}}
    handler = lambda request: httpx.Response(400, json=body)
    with pytest.raises(ConfigurationError) as info:
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))
    assert "sk-LEAK" not in info.value.user_message
    assert "GEMINI_API_KEY" in info.value.user_message


def test_http_429_is_provider_error(settings):
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    handler = lambda request: httpx.Response(429, json=body)
    with pytest.raises(ProviderError):
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))


def test_http_503_is_provider_error(settings):
    body = {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
    handler = lambda request: httpx.Response(503, json=body)
    with pytest.raises(ProviderError):
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))


def test_network_failure_is_provider_error(settings):
    def handler(request):
        raise httpx.ConnectError("boom sk-LEAK")

    with pytest.raises(ProviderError) as info:
        analyze_message("I was charged twice.", settings=settings, client=_client(handler))
    assert "sk-LEAK" not in info.value.user_message
