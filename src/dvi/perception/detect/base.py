"""Detect stage contract (ARCHITECTURE.md §3 stage 2)."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Detection:
    box_xywh: tuple[float, float, float, float]  # top-left x, y, width, height (pixels)
    score: float
    label: str


@runtime_checkable
class PersonDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...
