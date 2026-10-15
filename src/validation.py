"""Define validation models for application inputs."""

from pydantic import BaseModel, ConfigDict


class PromptInput(BaseModel):
    """Validate a single natural-language request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
