"""Pydantic models for the triage result.

Two models are used on purpose:

* ``ModelOutput`` is what we ask the API to produce. It only has types and
  enums, because providers support only a subset of JSON Schema.
* ``TriageResult`` is what the application accepts. It adds length limits and
  whitespace handling. Anything the model returns must pass this model, so
  the app never shows unvalidated output.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SHIPPING = "shipping"
    REFUND = "refund"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ModelOutput(BaseModel):
    """Schema sent to the API."""

    # Kept deliberately plain: Literal types render as inline JSON-schema enums
    # (no $ref) and there is no extra="forbid" (it adds additionalProperties,
    # which the Gemini API rejected). Strictness is enforced by TriageResult.
    category: Literal["billing", "technical", "account", "shipping", "refund", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str
    recommended_action: str
    draft_reply: str
    escalation_reason: Optional[str]


class TriageResult(BaseModel):
    """Validated triage result shown to the human reviewer."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Category
    priority: Priority
    summary: str = Field(min_length=10, max_length=400)
    recommended_action: str = Field(min_length=10, max_length=600)
    draft_reply: str = Field(min_length=20, max_length=2000)
    escalation_reason: Optional[str] = Field(default=None, max_length=400)

    @field_validator("escalation_reason")
    @classmethod
    def blank_escalation_is_none(cls, value: Optional[str]) -> Optional[str]:
        """Treat an empty/whitespace reason as 'no escalation'."""
        if value is None or not value.strip():
            return None
        return value

    @property
    def needs_escalation(self) -> bool:
        return self.escalation_reason is not None
