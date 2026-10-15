"""Validate application inputs using Pydantic models."""

from pydantic import BaseModel, ConfigDict, TypeAdapter


class PromptInput(BaseModel):
    """Represent a request with a required string prompt.

    Reject unexpected fields and disable automatic type coercion.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str


def validate_prompts(data: object) -> list[PromptInput]:
    """Validate the input list and convert its entries into models.

    Args:
        data: Python data loaded from the input JSON file.

    Returns:
        A list of validated PromptInput objects.

    Raises:
        ValidationError: If the input is not a list or an entry
            does not match the PromptInput schema.
    """
    # Validate the outer list as well as each request it contains.
    adapter = TypeAdapter(list[PromptInput])
    return adapter.validate_python(data, strict=True)
