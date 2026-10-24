"""Validate JSON argument objects with number and string parameters."""

import json

from src.number_object import (
    COMPLETE_NUMBER_STATES,
    JSON_WHITESPACE,
    number_transition,
)
from src.string_value import string_transition


def check_argument_object(
    text: str,
    parameters: dict[str, str],
) -> tuple[bool, bool]:
    """Check an argument object's syntax and declared parameter types.

    Args:
        text: Complete or partial JSON source text.
        parameters: Parameter names and types in generation order.

    Returns:
        A pair: (valid_prefix, complete_object).

    Raises:
        ValueError: If a parameter has an unsupported type.
    """
    if any(
        parameter_type not in ("number", "string")
        for parameter_type in parameters.values()
    ):
        raise ValueError("Only number and string parameters are supported.")

    position = 0

    def skip_whitespace() -> None:
        """Skip whitespace outside JSON values."""
        nonlocal position

        while (
            position < len(text)
            and text[position] in JSON_WHITESPACE
        ):
            position += 1

    def consume_literal(expected: str) -> str:
        """Consume fixed syntax: complete, partial, or invalid."""
        nonlocal position

        for character in expected:
            if position == len(text):
                return "partial"

            if text[position] != character:
                return "invalid"

            position += 1

        return "complete"

    def consume_number() -> str:
        """Read a number, leaving its separator unconsumed."""
        nonlocal position

        state = "start"

        while position < len(text):
            character = text[position]

            if character in JSON_WHITESPACE or character in ",}":
                break

            next_state = number_transition(state, character)

            if next_state is None:
                return "invalid"

            state = next_state
            position += 1

        # At EOF, a valid numeric prefix can still be extended.
        # The surrounding object is not complete yet.
        if position == len(text):
            return "partial"

        if state not in COMPLETE_NUMBER_STATES:
            return "invalid"

        return "complete"

    def consume_string() -> str:
        """Read through the closing quote using JSON escape rules."""
        nonlocal position

        state = "start"

        while position < len(text):
            next_state = string_transition(state, text[position])

            if next_state is None:
                return "invalid"

            state = next_state
            position += 1

            # Commas and braces inside the string are content.
            # Only an unescaped closing quote ends this value.
            if state == "done":
                return "complete"

        return "partial"

    skip_whitespace()
    status = consume_literal("{")

    if status != "complete":
        return status == "partial", False

    for index, (name, parameter_type) in enumerate(parameters.items()):
        if index > 0:
            skip_whitespace()
            status = consume_literal(",")

            if status != "complete":
                return status == "partial", False

        skip_whitespace()
        status = consume_literal(json.dumps(name, ensure_ascii=True))

        if status != "complete":
            return status == "partial", False

        skip_whitespace()
        status = consume_literal(":")

        if status != "complete":
            return status == "partial", False

        skip_whitespace()

        # Select value rules from the declared parameter type.
        if parameter_type == "number":
            status = consume_number()
        else:
            status = consume_string()

        if status != "complete":
            return status == "partial", False

    skip_whitespace()
    status = consume_literal("}")

    if status != "complete":
        return status == "partial", False

    skip_whitespace()

    # Reject anything except whitespace after the closing brace.
    if position != len(text):
        return False, False

    return True, True
