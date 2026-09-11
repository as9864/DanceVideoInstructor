"""FeatureSeries — raw geometric quantities derived from a PoseSequence (ARCHITECTURE.md §4.4).

One named scalar time series per feature (e.g. "torso_pitch_deg",
"com_offset_norm"), each with its own validity mask — a feature can be
invalid independently of the joints it was computed from (e.g. undefined
during a flight phase).
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class FeatureSeries(BaseModel):
    subject_id: str
    t: np.ndarray  # (T,) seconds
    features: dict[str, np.ndarray]  # name -> (T,)
    valid: dict[str, np.ndarray]  # name -> (T,) bool

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check_shapes_and_coverage(self) -> "FeatureSeries":
        t_len = self.t.shape[0]

        for name, arr in self.features.items():
            if arr.shape != (t_len,):
                raise ValueError(f"features[{name!r}].shape={arr.shape} != ({t_len},)")

        missing_valid = set(self.features) - set(self.valid)
        if missing_valid:
            raise ValueError(f"features missing a valid mask: {sorted(missing_valid)}")

        for name, arr in self.valid.items():
            if arr.shape != (t_len,):
                raise ValueError(f"valid[{name!r}].shape={arr.shape} != ({t_len},)")

        return self
