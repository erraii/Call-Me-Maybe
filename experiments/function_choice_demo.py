"""Generate a short reply using Qwen's chat format."""

import json
from pathlib import Path
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.file_io import read_json
from src.validation import validate_functions


def main() -> None:
    """Generate a reply one token at a time."""
    
    # Read and display the functions.
    functions_path = Path("data/input/functions_definition.json")

    try:
        functions_data = read_json(functions_path)
    except FileNotFoundError:
        print("Error: 'functions_definition.json' file was not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
        return
    except (OSError, UnicodeError) as error:
        print(f"Could not read input file: {error}")
        return

    try:
        validated_functions = validate_functions(functions_data)
    except ValidationError as error:
        print(f"Invalid input structure: {error}")
        return

    model = Small_LLM_Model()

    instruction = (
        """Select the function that best matches the request.
        Return only the function name, without explanation."""
    )

    available_functions = ""
    for function in validated_functions:
        # print(f"{function.name}: {function.description}")
        available_functions += function.name + ":" + function.description + "\n"

    request = "Greet shrek"
    # Mark the user message and the start of the assistant reply.
    # The empty think block selects the non-thinking chat format.
    prompt = (
        "<|im_start|>user\n"
        f"{instruction}\n"
        "Available functions:\n"
        f"{available_functions}\n"
        f"Request: {request}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    input_ids: list[int] = model.encode(prompt)[0].tolist()
    generated_ids: list[int] = []

    # Obtain stop-token IDs through the public SDK interface.
    stop_ids = {
        model.encode(marker)[0].tolist()[0]
        for marker in ("<|im_end|>", "<|endoftext|>")
    }

    max_new_tokens = 50

    print(f"Instruction: {instruction}")

    for step in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        # Argmax: select the token ID with the highest score.
        next_id = max(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
        )

        if next_id in stop_ids:
            print("\nEnd of reply.")
            break

        input_ids.append(next_id)
        generated_ids.append(next_id)

        token_text = model.decode([next_id])
        print(
            f"Step {step + 1:02d} | "
            f"ID: {next_id:>6} | "
            f"Text: {token_text!r}"
        )
    else:
        print("\nToken limit reached; reply may be incomplete.")

    reply = model.decode(generated_ids)
    print(f"\nReply: {reply}")


if __name__ == "__main__":
    main()
