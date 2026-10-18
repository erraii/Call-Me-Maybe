"""Select function names using constrained token generation."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.file_io import read_json
from src.validation import validate_functions, validate_prompts


def main() -> None:
    """Read inputs and select an available function for each request."""
    functions_path = Path("data/input/functions_definition.json")
    prompts_path = Path("data/input/function_calling_tests.json")

    try:
        functions_data = read_json(functions_path)
        prompts_data = read_json(prompts_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Could not read input files: {error}")
        return

    try:
        functions = validate_functions(functions_data)
        requests = validate_prompts(prompts_data)
    except ValidationError as error:
        print(f"Invalid input structure: {error}")
        return

    if not functions:
        print("No function definitions were provided.")
        return

    if not requests:
        print("No requests were provided.")
        return

    function_names = [function.name for function in functions]

    if any(not name or not name.isascii() for name in function_names):
        print("This experiment requires non-empty ASCII function names.")
        return

    available_functions = "\n".join(
        f"{function.name}: {function.description}"
        for function in functions
    )

    # Load the model only after input loading and validation succeed.
    model = Small_LLM_Model()

    stop_ids = {
        model.encode(marker)[0].tolist()[0]
        for marker in ("<|im_end|>", "<|endoftext|>")
    }

    instruction = (
        "Select the function that best matches the request.\n"
        "Return only the function name, without explanation."
    )

    # Cache token text once and reuse it for all requests.
    # Keep pieces occurring anywhere in a function name:
    # "_g" and "reet" are needed even though names start with "fn".
    token_texts: dict[int, str] = {}
    tokens_prepared = False

    for request in requests:
        prompt = (
            "<|im_start|>user\n"
            f"{instruction}\n"
            "Available functions:\n"
            f"{available_functions}\n"
            f"Request: {request.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        input_ids: list[int] = model.encode(prompt)[0].tolist()
        generated_ids: list[int] = []

        # Each accepted token adds at least one ASCII character.
        # The longest name bounds the steps; +1 allows a stop token
        max_new_tokens = max(map(len, function_names)) + 1

        print(f"\nRequest: {request.prompt}")

        for step in range(max_new_tokens):
            logits = model.get_logits_from_input_ids(input_ids)

            if not tokens_prepared:
                for token_id in range(len(logits)):
                    if token_id in stop_ids:
                        continue

                    text = model.decode([token_id])

                    # Reject empty text so every token makes progress.
                    # This is a broad filter; prefix checks happen later.
                    if text and any(
                        text in name for name in function_names
                    ):
                        token_texts[token_id] = text

                tokens_prepared = True

            current_text = model.decode(generated_ids)
            allowed_ids: list[int] = []

            for token_id, token_text in token_texts.items():
                # Test the full generated prefix, not just the new token.
                candidate_text = current_text + token_text

                # Accept only prefixes that can become an available name.
                if any(
                    name.startswith(candidate_text)
                    for name in function_names
                ):
                    allowed_ids.append(token_id)

            # Allow stopping only when the generated text is a complete name.
            # A complete name may also be a prefix of another valid name.
            if current_text in function_names:
                allowed_ids.extend(stop_ids)

            if not allowed_ids:
                print(f"No valid continuation for {current_text!r}")
                break

            # Let the model choose among valid tokens only.
            # Equivalent to masking forbidden logits with -infinity.
            next_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )

            if next_id in stop_ids:
                print("End of reply.")
                break

            # Update the model context and separately collect the reply.
            input_ids.append(next_id)
            generated_ids.append(next_id)

            token_text = model.decode([next_id])
            print(
                f"Step {step + 1:02d} | "
                f"ID: {next_id:>6} | "
                f"Text: {token_text!r}"
            )
        else:
            print("Token limit reached; reply may be incomplete.")

        reply = model.decode(generated_ids)

        if reply in function_names:
            print(f"Selected function: {reply}")
        else:
            print(f"Incomplete or invalid function name: {reply!r}")


if __name__ == "__main__":
    main()
