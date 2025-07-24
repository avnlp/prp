from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_config(path: Path, model: type[T]) -> T:
    """Load a YAML file and validate it against a Pydantic model.

    Args:
        path: Path to the YAML file.
        model: Pydantic model to validate against.

    Returns:
        The loaded and validated model.
    """
    raw_data = yaml.safe_load(path.read_text())

    return model.model_validate(raw_data)
