"""Validation errors that carry the path of the offending field."""

from __future__ import annotations

from dataclasses import dataclass

# ``ConfigError`` lives in ``robotdesigner`` (the lowest layer); it is re-exported here so there is one class.
from robotdesigner.errors import ConfigError

__all__ = ["ConfigError", "Problem"]


@dataclass(frozen=True)
class Problem:
    """One reason a config cannot be run, with the path of the field to change (for example ``control.mode``)."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"
