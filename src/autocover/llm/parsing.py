"""Helpers for pulling code and JSON out of model replies."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(text: str, language: str = "python") -> str:
    """Return the code in a reply: the longest fenced block tagged `language`, else the
    longest untagged block, else the whole reply stripped."""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    tagged = [body for lang, body in blocks if lang in (language, language[:2])]
    untagged = [body for lang, body in blocks if not lang]
    for group in (tagged, untagged):
        if group:
            return max(group, key=len).strip() + "\n"
    return text.strip() + "\n"


def extract_json(text: str) -> Any:
    """Parse JSON from a reply that may wrap it in a fence or surrounding prose."""
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON object found in reply: {text[:200]!r}")


def _json_candidates(text: str):
    yield text.strip()
    for lang, body in _FENCE.findall(text):
        if lang.lower() in ("", "json"):
            yield body.strip()
    # Outermost {...} or [...] span as a last resort.
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = text.find(open_), text.rfind(close)
        if 0 <= start < end:
            yield text[start : end + 1]
