"""Look up validated function definitions by name."""

from src.validation import FunctionDefinition


def find_function(
    functions: list[FunctionDefinition],
    name: str,
) -> FunctionDefinition:
    """Return the definition matching the selected function name.

    Args:
        functions: Validated function definitions.
        name: Exact function name selected during generation.

    Returns:
        The matching function definition.

    Raises:
        ValueError: If the selected name is not in the catalog.
    """
    for function in functions:
        if function.name == name:
            return function

    raise ValueError(
        f"Selected function {name!r} was not found "
        "in the function definitions."
    )
