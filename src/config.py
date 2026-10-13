"""Configuration loaded from environment variables and an optional .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from src.errors import ConfigurationError

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_MESSAGE_CHARS = 2000
MAX_RETRIES = 1


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES

    def __repr__(self) -> str:
        return (
            f"Settings(api_key='***', model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries})"
        )


def get_settings() -> Settings:
    load_dotenv(override=False)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        try:
            import streamlit as st
            api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            api_key = ""

    if not api_key or api_key.startswith("your-"):
        raise ConfigurationError(
            "The Gemini API key is not configured. Set GEMINI_API_KEY."
        )

    model = (
        os.environ.get("SUPPORTPILOT_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL
    )

    raw_timeout = os.environ.get("SUPPORTPILOT_TIMEOUT_SECONDS", "").strip()
    timeout = DEFAULT_TIMEOUT_SECONDS

    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            raise ConfigurationError(
                "SUPPORTPILOT_TIMEOUT_SECONDS must be a number."
            ) from None

    if not 1 <= timeout <= 120:
        raise ConfigurationError(
            "SUPPORTPILOT_TIMEOUT_SECONDS must be between 1 and 120."
        )

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout,
    )
