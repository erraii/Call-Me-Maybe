"""Generate JSON argument objects with numeric values."""

import json
import math

from llm_sdk import Small_LLM_Model

from src.argument_object import check_argument_object
from src.validation import FunctionDefinition


def generate_arguments(
    model: Small_LLM_Model,
    request: str,
    function: FunctionDefinition,
    token_texts: dict[int, str],
) -> dict[str, int | float | str]:
    """Generate a constrained JSON object with number/string arguments.

    Raises:
        ValueError: If the schema is unsupported, generation fails,
            or a numeric value cannot be represented safely.
    """

    parameters = {
        name: definition.type
        for name, definition in function.parameters.items()
    }

    if any(
        parameter_type not in ("number", "string")
        for parameter_type in parameters.values()
    ):
        raise ValueError(
            "This experiment supports number and string parameters only."
        )

    definition_json = json.dumps(
        function.model_dump(),
        ensure_ascii=True,
    )

    prompt = (
        "<|im_start|>system\n"
        "You extract function input arguments from user requests.\n"
        "Return only the JSON argument object, without explanation.\n"
        "Use exactly the parameter names and order in the definition.\n"
        "Parameter names are labels, not argument values.\n"
        "Extract values from the request; never execute the function.\n"
        "Preserve input strings exactly, including spaces and slashes.\n"
        "For regex parameters, produce a regex pattern that matches "
        "the requested category or text, not a list of examples.\n"
        "\n"
        "Examples:\n"
        'Parameters: {"person": "string"}\n'
        "Request: Say hello to Alice.\n"
        'Arguments: {"person": "Alice"}\n'
        "\n"
        'Parameters: {"text": "string"}\n'
        "Request: Reverse 'garden'.\n"
        'Arguments: {"text": "garden"}\n'
        "\n"
        'Parameters: {"x": "number", "y": "number"}\n'
        "Request: Add 7 and 12.\n"
        'Arguments: {"x": 7, "y": 12}\n'
        "\n"
        'Parameters: {"source": "string", "pattern": "string", '
        '"replacement": "string"}\n'
        "Request: Replace every digit sequence in 'Room 82, floor 6' "
        "with '#'.\n"
        'Arguments: {"source": "Room 82, floor 6", '
        '"pattern": "[0-9]+", "replacement": "#"}\n'
        "\n"
        'Parameters: {"source": "string", "pattern": "string", '
        '"replacement": "string"}\n'
        "Request: Replace each comma or semicolon in 'red,blue;green' "
        "with '|'.\n"
        'Arguments: {"source": "red,blue;green", '
        '"pattern": "[,;]", "replacement": "|"}\n'
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Function definition: {definition_json}\n"
        f"Request: {request}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    input_ids: list[int] = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []

    # Operational limit: reaching it is a failure, not valid completion.
    max_new_tokens = 128

    for step in range(max_new_tokens):
        current_text = model.decode(generated_ids)
        _, complete = check_argument_object(
            current_text, parameters
        )

        # A complete object needs no further model call.
        if complete:
            break

        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids: list[int] = []

        for token_id, token_text in token_texts.items():
            candidate_text = current_text + token_text
            valid_prefix, _ = check_argument_object(
                candidate_text, parameters
            )

            if valid_prefix:
                allowed_ids.append(token_id)

        if not allowed_ids:
            raise ValueError(
                f"No valid continuation for {current_text!r}"
            )

        next_id = max(
            allowed_ids,
            key=lambda token_id: logits[token_id],
        )

        input_ids.append(next_id)
        generated_ids.append(next_id)

        print(
            f"Argument step {step + 1:02d} | "
            f"Token: {token_texts[next_id]!r}"
        )

    text = model.decode(generated_ids)
    _, complete = check_argument_object(text, parameters)

    if not complete:
        raise ValueError(
            f"Argument generation did not complete: {text!r}"
        )

    # Syntax is checked above; parse and verify the resulting values too.
    parsed = json.loads(text)
    arguments: dict[str, int | float | str] = {}

    for name, parameter_type in parameters.items():
        value = parsed[name]

        if parameter_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"Parameter {name!r} is not a string.")

        else:
            if type(value) not in (int, float):
                raise ValueError(f"Parameter {name!r} is not a number.")

            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"Parameter {name!r} is outside the supported float range."
                )

        arguments[name] = value

    return arguments
