"""Check prefixes of JSON objects containing numeric parameters."""

import json


# States in the JSON number grammar and their valid transitions.
NUMBER_TRANSITIONS: dict[str, dict[str, str]] = {
    "start": {
        "-": "sign",
        "zero": "zero",
        "digit": "integer",
    },
    "sign": {
        "zero": "zero",
        "digit": "integer",
    },
    "zero": {
        ".": "point",
        "e": "exponent",
    },
    "integer": {
        "zero": "integer",
        "digit": "integer",
        ".": "point",
        "e": "exponent",
    },
    "point": {
        "zero": "fraction",
        "digit": "fraction",
    },
    "fraction": {
        "zero": "fraction",
        "digit": "fraction",
        "e": "exponent",
    },
    "exponent": {
        "+": "exponent_sign",
        "-": "exponent_sign",
        "zero": "exponent_digits",
        "digit": "exponent_digits",
    },
    "exponent_sign": {
        "zero": "exponent_digits",
        "digit": "exponent_digits",
    },
    "exponent_digits": {
        "zero": "exponent_digits",
        "digit": "exponent_digits",
    },
}

COMPLETE_NUMBER_STATES = {
    "zero",
    "integer",
    "fraction",
    "exponent_digits",
}

JSON_WHITESPACE = " \t\r\n"


def number_transition(state: str, character: str) -> str | None:
    """Return the next numeric state, or None for an invalid character."""
    if character == "0":
        category = "zero"
    elif character in "123456789":
        category = "digit"
    elif character in "eE":
        category = "e"
    else:
        category = character

    return NUMBER_TRANSITIONS[state].get(category)


def check_number_object(
    text: str,
    parameter_names: list[str],
) -> tuple[bool, bool]:
    """Check a JSON object's prefix against ordered numeric parameters.

    Args:
        text: The generated text, possibly incomplete.
        parameter_names: Required keys in generation order.

    Returns:
        A pair: (valid_prefix, complete_object).

        An incomplete but extendable prefix returns (True, False).
        An invalid prefix returns (False, False).
        A complete object returns (True, True).
    """
    position = 0

    def skip_whitespace() -> None:
        """Skip JSON whitespace outside keys and numbers."""
        nonlocal position

        while (
            position < len(text)
            and text[position] in JSON_WHITESPACE
        ):
            position += 1

    def consume_literal(expected: str) -> str:
        """Consume fixed text and report complete, partial, or invalid."""
        nonlocal position

        for character in expected:
            if position == len(text):
                return "partial"

            if text[position] != character:
                return "invalid"

            position += 1

        return "complete"

    skip_whitespace()
    status = consume_literal("{")

    if status != "complete":
        return status == "partial", False

    for index, name in enumerate(parameter_names):
        if index > 0:
            skip_whitespace()
            status = consume_literal(",")

            if status != "complete":
                return status == "partial", False

        skip_whitespace()

        # json.dumps safely quotes the declared parameter name.
        status = consume_literal(json.dumps(name, ensure_ascii=True))

        if status != "complete":
            return status == "partial", False

        skip_whitespace()
        status = consume_literal(":")

        if status != "complete":
            return status == "partial", False

        skip_whitespace()
        state = "start"

        while position < len(text):
            character = text[position]

            if character in JSON_WHITESPACE or character in ",}":
                break

            next_state = number_transition(state, character)

            if next_state is None:
                return False, False

            state = next_state
            position += 1

        # At the end, even "-" or "2." can still be completed.
        if position == len(text):
            return True, False

        # Before a separator, the number must already be complete.
        if state not in COMPLETE_NUMBER_STATES:
            return False, False

    skip_whitespace()
    status = consume_literal("}")

    if status != "complete":
        return status == "partial", False

    skip_whitespace()

    if position != len(text):
        return False, False

    return True, True
