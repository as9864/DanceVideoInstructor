"""CalibrationProfile — the output of gravity estimation (ARCHITECTURE.md §5.2).

Kept separate from gravity.py's estimators so it can be persisted/reused
("같은 장소·같은 카메라면 재활용") once there's a real session store to put
it in — that wiring comes with the actual GravityAlign pipeline stage.
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

GravityMethod = Literal["model_provided", "floor_plane_ransac", "long_axis_prior", "manual"]


class CalibrationProfile(BaseModel):
    method: GravityMethod
    rotation: np.ndarray  # (3, 3) — maps pre-alignment 3D points onto WORLD_G
    confidence: float  # 0..1; low confidence should suppress absolute-angle claims (§5.2)
    metadata: dict[str, float] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check(self) -> "CalibrationProfile":
        if self.rotation.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {self.rotation.shape}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        return self

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Rotate (..., 3) points into WORLD_G. Does not translate — gravity
        alignment fixes orientation only, not the coordinate origin."""
        return points @ self.rotation.T
