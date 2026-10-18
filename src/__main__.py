import json
from pathlib import Path
from pydantic import ValidationError
from src.validation import validate_prompts, validate_functions
from src.file_io import read_json


def main() -> None:
    """Run the application."""
    print(
        "\n"
        "  ┌──────────────────────────────────────────┐\n"
        "  │             CALL ME MAYBE                │\n"
        "  │     Natural language → Function calls    │\n"
        "  └──────────────────────────────────────────┘\n"
    )

    # Read and display the input prompts.
    input_path = Path("data/input/function_calling_tests.json")

    try:
        prompts = read_json(input_path)
    except FileNotFoundError:
        print("Error: 'function_calling_tests.json' file was not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
        return
    except (OSError, UnicodeError) as error:
        print(f"Could not read input file: {error}")
        return

    # for item in prompts:
    #     print(item["prompt"])
    try:
        validated_prompts = validate_prompts(prompts)
    except ValidationError as error:
        print(f"Invalid input structure: {error}")
        return
    for item in validated_prompts:
        print(item.prompt)

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
    for function in validated_functions:
        print(f"{function.name}: {function.description}")


if __name__ == "__main__":
    main()
