"""Model-quality evaluation runner (calls the REAL model).

This is NOT part of the pytest suite. The deterministic tests in ``tests/``
check our code; this script checks what a live model actually does on a small
fictional dataset and compares it with documented expected labels.

Usage (from the project root, with GEMINI_API_KEY configured):

    python evals/run_evaluation.py --dry-run       # validate dataset only, no API calls
    python evals/run_evaluation.py --live          # calls the Gemini API (may use quota or cost money)
    python evals/run_evaluation.py --live --limit 3 --output evals/results/run.json

Results are observations from one run. LLM outputs vary between runs, and a
14-case dataset is far too small to support general accuracy claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.errors import SupportPilotError  # noqa: E402
from src.schemas import Category, Priority  # noqa: E402
from src.service import analyze_message, find_unsupported_claims, validate_message  # noqa: E402

DATASET_PATH = ROOT / "evals" / "dataset.json"
REQUIRED_KEYS = {"id", "message", "expected_categories", "expected_priorities", "tags", "notes"}


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Load and structurally validate the dataset (deterministic, no API)."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for case in cases:
        missing = REQUIRED_KEYS - case.keys()
        if missing:
            raise ValueError(f"Case {case.get('id')} is missing keys: {sorted(missing)}")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        for value in case["expected_categories"]:
            Category(value)
        for value in case["expected_priorities"]:
            Priority(value)
        validate_message(case["message"])
    return cases


def run_case(case: dict, settings) -> dict:
    record = {"id": case["id"], "tags": case["tags"]}
    start = time.perf_counter()
    try:
        result = analyze_message(case["message"], settings=settings)
    except SupportPilotError as exc:
        record.update(error=exc.user_message)
        return record
    record["latency_seconds"] = round(time.perf_counter() - start, 2)

    warnings = find_unsupported_claims(result.draft_reply)
    record.update(
        category=result.category.value,
        priority=result.priority.value,
        escalation_reason=result.escalation_reason,
        summary=result.summary,
        recommended_action=result.recommended_action,
        draft_reply=result.draft_reply,
        category_match=result.category.value in case["expected_categories"],
        priority_match=result.priority.value in case["expected_priorities"],
        unsupported_claim_warnings=warnings,
    )
    if "expected_escalation" in case:
        record["escalation_match"] = result.needs_escalation == case["expected_escalation"]
    if case.get("must_not_claim_action"):
        record["no_claim_flagged"] = not warnings
    return record


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]

    def count(key: str) -> tuple[int, int]:
        rows = [r for r in ok if key in r]
        return sum(1 for r in rows if r[key]), len(rows)

    return {
        "cases_run": len(records),
        "api_or_validation_errors": len(records) - len(ok),
        "category_match": count("category_match"),
        "priority_match": count("priority_match"),
        "escalation_match": count("escalation_match"),
        "no_claim_flagged_on_guarded_cases": count("no_claim_flagged"),
    }


def print_report(records: list[dict], summary: dict) -> None:
    print("\n=== Observed results (single run; not a benchmark) ===\n")
    for r in records:
        print(f"[{r['id']}]")
        if "error" in r:
            print(f"  ERROR: {r['error']}\n")
            continue
        flags = []
        flags.append("category OK" if r["category_match"] else "category MISMATCH")
        flags.append("priority OK" if r["priority_match"] else "priority MISMATCH")
        if "escalation_match" in r:
            flags.append("escalation OK" if r["escalation_match"] else "escalation MISMATCH")
        if r["unsupported_claim_warnings"]:
            flags.append("CLAIM WARNING: " + "; ".join(r["unsupported_claim_warnings"]))
        print(f"  got: {r['category']} / {r['priority']}  | {', '.join(flags)}")
        print(f"  escalation: {r['escalation_reason']}")
        print(f"  action: {r['recommended_action']}")
        print(f"  latency: {r['latency_seconds']}s\n")
    print("=== Summary (matches / cases with that check) ===")
    for key, value in summary.items():
        if isinstance(value, tuple):
            print(f"  {key}: {value[0]}/{value[1]}")
        else:
            print(f"  {key}: {value}")
    print(
        "\nExpected labels are judgment calls and several cases accept more than one "
        "label. Read the draft replies yourself; automated checks are partial.\n"
        "Full drafts are available via --output."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate the dataset only; no API calls")
    mode.add_argument("--live", action="store_true", help="call the real Gemini model (uses quota / may cost money)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    parser.add_argument("--output", type=Path, default=None, help="write full results as JSON")
    args = parser.parse_args()

    cases = load_dataset()
    if args.limit:
        cases = cases[: args.limit]

    if args.dry_run:
        print(f"Dataset OK: {len(cases)} cases validated. No API calls were made.")
        return 0

    try:
        settings = get_settings()
    except SupportPilotError as exc:
        print(f"Configuration error: {exc.user_message}", file=sys.stderr)
        return 2

    print(f"Running {len(cases)} cases against model '{settings.model}' (real API calls)...")
    records = [run_case(case, settings) for case in cases]
    summary = summarize(records)
    print_report(records, summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": settings.model, "summary": summary, "results": records}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Full results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
