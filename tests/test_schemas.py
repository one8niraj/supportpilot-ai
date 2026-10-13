import pytest
from pydantic import ValidationError

from src.schemas import Category, Priority, TriageResult

VALID = dict(
    category="billing",
    priority="high",
    summary="Customer was charged twice for one order.",
    recommended_action="Ask for the order number and both charge dates.",
    draft_reply="Hello, I'm sorry about this. Could you share your order number so we can look into it?",
    escalation_reason=None,
)


def make(**overrides) -> TriageResult:
    return TriageResult.model_validate({**VALID, **overrides})


def test_valid_result_parses():
    result = make()
    assert result.category is Category.BILLING
    assert result.priority is Priority.HIGH
    assert result.needs_escalation is False


@pytest.mark.parametrize("field,value", [("category", "shipping_delay"), ("priority", "critical")])
def test_invalid_enum_values_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", "short"),
        ("summary", "x" * 401),
        ("recommended_action", "x" * 601),
        ("draft_reply", "too short"),
        ("draft_reply", "x" * 2001),
        ("escalation_reason", "x" * 401),
    ],
)
def test_invalid_lengths_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


@pytest.mark.parametrize("missing", ["category", "priority", "summary", "recommended_action", "draft_reply"])
def test_missing_required_fields_rejected(missing):
    data = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(ValidationError):
        TriageResult.model_validate(data)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        make(send_email=True)


def test_blank_escalation_reason_becomes_none():
    assert make(escalation_reason="   ").escalation_reason is None


def test_escalation_reason_sets_flag():
    assert make(escalation_reason="Possible fraud; needs security team.").needs_escalation is True


def test_whitespace_is_stripped():
    assert make(summary="   Customer was charged twice.   ").summary == "Customer was charged twice."


# ---- the schema we send to the API ------------------------------------------

def test_api_schema_uses_only_plain_json_schema_keywords():
    """Regression: Gemini rejected 'additionalProperties' (sent as additional_properties)."""
    import json

    from src.schemas import ModelOutput

    text = json.dumps(ModelOutput.model_json_schema())
    for forbidden in ("additionalProperties", "additional_properties", "$ref", "$defs", "minLength", "maxLength"):
        assert forbidden not in text


def test_api_schema_literals_match_enums():
    from src.schemas import ModelOutput

    props = ModelOutput.model_json_schema()["properties"]
    assert props["category"]["enum"] == [c.value for c in Category]
    assert props["priority"]["enum"] == [p.value for p in Priority]
