"""Plain-text rendering shared by the ``__repr__`` of the config objects. Imports nothing from the package.

Two shapes: ``kv`` (aligned ``name  value`` lines) and ``table`` (one row per item). Both leave out what is empty: a
``kv`` line whose value is empty is not printed, and a ``table`` column that is empty in every row is dropped. An
empty cell in a column that stays is left blank. ``None`` therefore prints as nothing, never as the word ``None``.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

__all__ = ["fmt", "indent", "kv", "num", "oneline", "pre_html", "table"]


def num(x: float | int) -> str:
    """A number for people: six significant digits, no trailing zeros, ``-0`` written as ``0``."""
    if isinstance(x, float):
        return f"{x + 0.0:.6g}"
    return str(x)


def fmt(v: Any) -> str:
    """A value as it appears in a table cell or a ``kv`` line; ``""`` for ``None``.

    An object may define ``_short()`` to say how it prints here (``Pose``, ``Capsule``, ...); anything else falls back to
    its ``repr``.
    """
    if v is None:
        return ""
    short = getattr(v, "_short", None)
    if short is not None:
        return short()
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return num(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    return repr(v)


def oneline(v: Any) -> str:
    """``repr(v)`` on one line, for the places that embed a value in a row (``explain()``, messages). An object whose
    ``repr`` spans lines defines ``_oneline()``."""
    one = getattr(v, "_oneline", None)
    return one() if one is not None else repr(v)


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def kv(pairs: Sequence[tuple[str, Any]], prefix: str = "  ") -> list[str]:
    """``name  value`` lines with the names padded to one width; pairs whose value is empty are skipped."""
    shown = [(k, fmt(v)) for k, v in pairs]
    shown = [(k, s) for k, s in shown if s != ""]
    if not shown:
        return []
    width = max(len(k) for k, _ in shown)
    return [f"{prefix}{k:<{width}}  {s}" for k, s in shown]


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], prefix: str = "  ") -> list[str]:
    """A header line and one line per row. Cells are formatted with ``fmt``; a column (other than the first, which
    names the row) with no content in any row is dropped."""
    cells = [[fmt(c) for c in row] for row in rows]
    keep = [j for j in range(len(headers)) if j == 0 or any(row[j] for row in cells)]
    widths = {j: max([len(headers[j]), *(len(row[j]) for row in cells)]) for j in keep}

    def line(parts: Sequence[str]) -> str:
        return (prefix + "  ".join(f"{parts[j]:<{widths[j]}}" for j in keep)).rstrip()

    return [line(headers), *(line(row) for row in cells)]


def pre_html(text: str) -> str:
    """The text in a ``<pre>`` block, for ``_repr_html_`` (a notebook shows it in a fixed-width font)."""
    return f"<pre>{html.escape(text)}</pre>"
