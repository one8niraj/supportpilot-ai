"""Deterministic checks on the evaluation dataset and runner (no API calls)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import run_evaluation as ev  # noqa: E402

from tests.conftest import FakeClient, make_output, make_response  # noqa: E402


def test_dataset_loads_and_is_valid():
    cases = ev.load_dataset()
    assert len(cases) >= 12


def test_dataset_covers_required_scenarios():
    cases = ev.load_dataset()
    tags = {tag for case in cases for tag in case["tags"]}
    categories = {c for case in cases for c in case["expected_categories"]}
    priorities = {p for case in cases for p in case["expected_priorities"]}
    assert {"adversarial", "ambiguous", "missing-details"} <= tags
    assert categories == {"billing", "technical", "account", "shipping", "refund", "other"}
    assert priorities == {"low", "medium", "high", "urgent"}


def test_dataset_rejects_invalid_label(tmp_path):
    bad = [{"id": "x", "message": "hello", "expected_categories": ["nope"],
            "expected_priorities": ["low"], "tags": [], "notes": ""}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        ev.load_dataset(path)


def test_run_case_scoring_with_fake_client(settings, monkeypatch):
    client = FakeClient(response=make_response(output=make_output()))
    monkeypatch.setattr("src.service._build_client", lambda s: client)
    case = {"id": "t", "message": "charged twice", "expected_categories": ["billing"],
            "expected_priorities": ["high"], "tags": [], "notes": "", "must_not_claim_action": True}
    record = ev.run_case(case, settings)
    assert record["category_match"] and record["priority_match"] and record["no_claim_flagged"]


def test_summary_counts_errors():
    summary = ev.summarize([{"id": "a", "error": "boom"}])
    assert summary["api_or_validation_errors"] == 1
