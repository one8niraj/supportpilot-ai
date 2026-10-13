import pytest

from src.config import MAX_MESSAGE_CHARS, get_settings
from src.errors import ConfigurationError, InvalidInputError
from src.service import validate_message


@pytest.mark.parametrize("message", [None, "", "   ", "\n\t  \n"])
def test_empty_messages_rejected(message):
    with pytest.raises(InvalidInputError):
        validate_message(message)


def test_oversized_message_rejected():
    with pytest.raises(InvalidInputError) as info:
        validate_message("a" * (MAX_MESSAGE_CHARS + 1))
    assert str(MAX_MESSAGE_CHARS) in info.value.user_message


def test_message_at_limit_accepted():
    assert len(validate_message("a" * MAX_MESSAGE_CHARS)) == MAX_MESSAGE_CHARS


def test_message_is_stripped():
    assert validate_message("  hello  ") == "hello"


def test_missing_api_key_raises_configuration_error():
    with pytest.raises(ConfigurationError) as info:
        get_settings()
    assert "GEMINI_API_KEY" in info.value.user_message


def test_placeholder_api_key_treated_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your-api-key-here")
    with pytest.raises(ConfigurationError):
        get_settings()


def test_settings_from_env_and_repr_hides_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("SUPPORTPILOT_MODEL", "my-model")
    settings = get_settings()
    assert settings.model == "my-model"
    assert "sk-test-secret-value" not in repr(settings)


@pytest.mark.parametrize("value", ["abc", "0", "500"])
def test_invalid_timeout_rejected(monkeypatch, value):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.setenv("SUPPORTPILOT_TIMEOUT_SECONDS", value)
    with pytest.raises(ConfigurationError):
        get_settings()
