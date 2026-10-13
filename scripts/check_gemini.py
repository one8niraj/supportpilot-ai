"""Diagnose Gemini setup: list usable models and test the exact call the app makes.

Usage (project root, venv active, GEMINI_API_KEY set in .env):

    python scripts/check_gemini.py                 # list models + test the configured one
    python scripts/check_gemini.py gemini-2.5-flash   # test a specific model

This makes a few tiny real API calls. It prints Gemini's own error text (with your
key redacted) so you can see exactly why a request was rejected.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.errors import SupportPilotError  # noqa: E402
from src.prompts import SYSTEM_PROMPT, build_user_content  # noqa: E402
from src.schemas import ModelOutput  # noqa: E402


def redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def try_call(client: genai.Client, model: str, key: str, label: str, **kwargs) -> bool:
    start = time.perf_counter()
    try:
        client.models.generate_content(model=model, **kwargs)
    except errors.APIError as exc:
        took = time.perf_counter() - start
        print(f"  [FAIL] ({took:.1f}s) {label}: HTTP {exc.code} {exc.status}: {redact(str(exc.message), key)[:400]}")
        return False
    except Exception as exc:  # noqa: BLE001 - diagnostic tool, show type only
        print(f"  [FAIL] ({time.perf_counter() - start:.1f}s) {label}: {type(exc).__name__}")
        return False
    print(f"  [ OK ] ({time.perf_counter() - start:.1f}s) {label}")
    return True


def main() -> int:
    try:
        settings = get_settings()
    except SupportPilotError as exc:
        print(f"Config error: {exc.user_message}")
        return 2
    model = sys.argv[1] if len(sys.argv) > 1 else settings.model
    key = settings.api_key
    client = genai.Client(api_key=key)

    print("Models on your key that support generateContent:")
    try:
        names = []
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                names.append((m.name or "").removeprefix("models/"))
        for name in sorted(n for n in names if "gemini" in n):
            print(f"  {name}")
    except errors.APIError as exc:
        print(f"  Could not list models: HTTP {exc.code} {redact(str(exc.message), key)[:300]}")

    print(f"\nTesting model: {model}")
    plain_ok = try_call(client, model, key, "1. plain text call", contents="Reply with the word OK.")
    schema_ok = try_call(
        client, model, key, "2. structured output (what the app does)",
        contents=build_user_content("I was charged twice for my order."),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=ModelOutput.model_json_schema(),
            max_output_tokens=4096,
        ),
    )
    if schema_ok:
        try_call(
            client, model, key, "3. same, with thinking_level=MINIMAL (speed test)",
            contents=build_user_content("I was charged twice for my order."),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=ModelOutput.model_json_schema(),
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
            ),
        )
    if plain_ok and schema_ok:
        print("\nThis model works. Put it in .env as SUPPORTPILOT_MODEL=" + model)
    elif not plain_ok:
        print(
            "\nThe plain call failed. HTTP 503 means Google is temporarily overloaded: wait a "
            "minute and rerun, or try another model from the list above. HTTP 404 means the "
            "model name is wrong. Other errors: send me the FAIL line."
        )
    else:
        print("\nPlain call works but structured output fails: send me the FAIL line above.")
    return 0 if (plain_ok and schema_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
