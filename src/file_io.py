import json
from pathlib import Path


def read_json(path: Path) -> object:
    """Read a JSON file and return its parsed contents."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
