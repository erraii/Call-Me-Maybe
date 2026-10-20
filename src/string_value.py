"""Validate complete or partial JSON string values."""

STRING_TRANSITIONS: dict[str, dict[str, str]] = {
    "start": {
        "quote": "content",
    },
    "content": {
        "normal": "content",
        "backslash": "escape",
        "quote": "done",
    },
    "escape": {
        "simple_escape": "content",
        "unicode_escape": "unicode_1",
    },
    "unicode_1": {
        "hex": "unicode_2",
    },
    "unicode_2": {
        "hex": "unicode_3",
    },
    "unicode_3": {
        "hex": "unicode_4",
    },
    "unicode_4": {
        "hex": "content",
    },
}

JSON_WHITESPACE = " \t\r\n"
HEX_DIGITS = "0123456789abcdefABCDEF"


def string_transition(state: str, character: str) -> str | None:
    """Return the next string state, or None for an invalid character."""
    if state == "escape":
        if character in '"\\/bfnrt':
            category = "simple_escape"
        elif character == "u":
            category = "unicode_escape"
        else:
            return None

    elif state.startswith("unicode_"):
        if character not in HEX_DIGITS:
            return None

        category = "hex"

    elif character == '"':
        category = "quote"

    elif character == "\\":
        category = "backslash"

    elif ord(character) < 0x20:
        # JSON forbids unescaped control characters inside strings.
        return None

    else:
        category = "normal"

    return STRING_TRANSITIONS.get(state, {}).get(category)


def check_string_value(text: str) -> tuple[bool, bool]:
    """Check whether text is a valid prefix of a JSON string.

    Args:
        text: JSON source text, including its opening/closing quotes
            when present.

    Returns:
        A pair: (valid_prefix, complete_string).

        Leading and trailing JSON whitespace is allowed.
        String content, including its whitespace, is preserved.
    """
    state = "start"

    for character in text:
        # Whitespace is allowed before the opening quote.
        if state == "start" and character in JSON_WHITESPACE:
            continue

        # After closing, only JSON whitespace is allowed.
        if state == "done":
            if character not in JSON_WHITESPACE:
                return False, False

            continue

        next_state = string_transition(state, character)

        if next_state is None:
            return False, False

        state = next_state

    return True, state == "done"
