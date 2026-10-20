"""Generate JSON argument objects with numeric values."""

import json
import math

from llm_sdk import Small_LLM_Model

from src.number_object import check_number_object
from src.validation import FunctionDefinition


def generate_number_arguments(
    model: Small_LLM_Model,
    request: str,
    function: FunctionDefinition,
    token_texts: dict[int, str],
) -> dict[str, int | float]:
    """Generate a complete numeric argument object using token constraints.

    Raises:
        ValueError: If the schema is unsupported, generation fails,
            or a generated value cannot be represented safely.
    """
    if any(
        definition.type != "number"
        for definition in function.parameters.values()
    ):
        raise ValueError("This experiment supports number parameters only.")

    parameter_names = list(function.parameters)

    definition_json = json.dumps(
        function.model_dump(),
        ensure_ascii=True,
    )

    prompt = (
        "<|im_start|>user\n"
        "Extract the arguments for the selected function.\n"
        "Return only a JSON object containing its parameters.\n"
        "Use the parameter order shown in the definition.\n"
        "Extract input values; do not calculate the function result.\n"
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
        _, complete = check_number_object(
            current_text, parameter_names
        )

        # A complete object needs no further model call.
        if complete:
            break

        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids: list[int] = []

        for token_id, token_text in token_texts.items():
            candidate_text = current_text + token_text
            valid_prefix, _ = check_number_object(
                candidate_text, parameter_names
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
    _, complete = check_number_object(text, parameter_names)

    if not complete:
        raise ValueError(
            f"Argument generation did not complete: {text!r}"
        )

    # Syntax is checked above; parse and verify the resulting values too.
    parsed = json.loads(text)
    arguments: dict[str, int | float] = {}

    for name in parameter_names:
        value = parsed[name]

        if type(value) not in (int, float):
            raise ValueError(f"Parameter {name!r} is not a number.")

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Parameter {name!r} is outside the supported float range."
            )

        arguments[name] = value

    return arguments
