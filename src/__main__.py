import json
from pathlib import Path
from pydantic import ValidationError
from src.validation import validate_prompts


def main() -> None:
    """Run the application."""
    print("Welcome to my project!")
    print("Call Me Maybe")
    """Read and display the input prompts."""
    input_path = Path("data/input/function_calling_tests.json")
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            prompts = json.load(file)
    except FileNotFoundError:
        print("Error: 'function_calling_tests.json' file was not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
        return
    except (OSError, UnicodeError) as error:
        print(f"Could not read input file: {error}")
        return

    else:
        # for item in prompts:
        #     print(item["prompt"])
        try:
            validated_prompts = validate_prompts(prompts)
        except ValidationError as error:
            print(f"Invalid input structure: {error}")
            return
        for item in validated_prompts:
            print(item.prompt)


if __name__ == "__main__":
    main()
