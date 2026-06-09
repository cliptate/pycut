from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Highlight:
    """Video highlight segment."""

    start: float
    end: float
    title: str
    subtitle: str
    content: str
