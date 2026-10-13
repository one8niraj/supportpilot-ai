"""Application-level exceptions.

Every exception carries a ``user_message`` that is safe to show in the UI:
it never contains secrets, raw provider error text, or stack traces.
"""


class SupportPilotError(Exception):
    """Base class. ``user_message`` is always safe to display."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class ConfigurationError(SupportPilotError):
    """Missing or invalid configuration (e.g. no API key, unknown model)."""


class InvalidInputError(SupportPilotError):
    """The customer message failed validation (empty or too long)."""


class ProviderError(SupportPilotError):
    """The LLM provider call failed (auth, rate limit, network, timeout...)."""


class MalformedResponseError(SupportPilotError):
    """The model answered, but the result was refused, truncated or invalid."""
