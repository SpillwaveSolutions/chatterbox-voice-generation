"""Pronunciation pre-substitution ported from deck2video/tts.py.

Three functions:
- load_pronunciations(data): validates a dict of str->str entries with caps
  (<=1000 entries, <=200 chars per key/value). Raises ValueError on bad input.
- compile_pronunciations(mapping): pre-compiles to a list of
  (re.Pattern, replacement) tuples sorted by descending key length so multi-word
  phrases match before their substrings. Uses re.escape + re.IGNORECASE.
- apply_pronunciations(text, patterns): sequential pattern.sub over the list.
  Output of pass N is input to pass N+1. Empty list returns input unchanged.
  Never mutates the input string (the for-loop rebinds the local `text`).
"""
from __future__ import annotations

import re

PRONUNCIATIONS_MAX_ENTRIES = 1000
PRONUNCIATIONS_MAX_FIELD_LEN = 200


def load_pronunciations(data: dict) -> dict[str, str]:
    """Validate a pronunciation mapping dict.

    Args:
        data: Mapping of phrase -> phonetic respelling (e.g.
            ``{"kubectl": "cube control"}``).

    Returns:
        The validated dict (unchanged).

    Raises:
        ValueError: If ``data`` is not a dict, contains non-string keys/values,
            empty keys, exceeds ``PRONUNCIATIONS_MAX_ENTRIES`` items, or any
            field exceeds ``PRONUNCIATIONS_MAX_FIELD_LEN`` chars.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"pronunciations: must be a dict, got {type(data).__name__}"
        )
    if len(data) > PRONUNCIATIONS_MAX_ENTRIES:
        raise ValueError(
            f"pronunciations: {len(data)} entries; max "
            f"{PRONUNCIATIONS_MAX_ENTRIES} allowed"
        )
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(
                f"pronunciations: keys must be strings, got "
                f"{type(key).__name__}"
            )
        if not isinstance(value, str):
            raise ValueError(
                f"pronunciations: value for {key!r} must be a string, got "
                f"{type(value).__name__}"
            )
        if not key:
            raise ValueError("pronunciations: empty keys not allowed")
        if (
            len(key) > PRONUNCIATIONS_MAX_FIELD_LEN
            or len(value) > PRONUNCIATIONS_MAX_FIELD_LEN
        ):
            raise ValueError(
                f"pronunciations: max {PRONUNCIATIONS_MAX_FIELD_LEN} chars "
                f"per key/value (got {key!r})"
            )
    return data


def compile_pronunciations(
    mapping: dict[str, str],
) -> list[tuple[re.Pattern, str]]:
    """Pre-compile pronunciation patterns sorted by descending key length.

    re.escape protects against regex metacharacters in user input;
    re.IGNORECASE makes matching case-insensitive. The descending-length sort
    ensures multi-word keys ("MySQL") match before their substrings ("SQL").
    """
    return [
        (re.compile(re.escape(word), re.IGNORECASE), mapping[word])
        for word in sorted(mapping, key=len, reverse=True)
    ]


def apply_pronunciations(
    text: str, patterns: list[tuple[re.Pattern, str]]
) -> str:
    """Apply compiled pronunciation patterns to ``text`` sequentially.

    Output of pass N is input to pass N+1. Empty list returns input
    unchanged. The input string is never mutated (the for-loop rebinds the
    local ``text`` variable on each iteration; strings are immutable).
    """
    if not patterns:
        return text
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text
