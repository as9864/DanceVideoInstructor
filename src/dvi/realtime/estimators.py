"""LivePoseEstimator adapters over the existing perception backends."""

import numpy as np

from dvi.realtime.session import BoxXYWH


class ViTPoseLiveEstimator:
    """Tier A ViTPose (perception/pose/tier_a_vitpose.py), one box per frame.

    Imported lazily so `dvi live --help` and the test suite don't pay for
    torch/transformers at import time.
    """

    def __init__(self, device: str | None = None):
        from dvi.perception.pose.tier_a_vitpose import ViTPoseTierABackend

        self.backend = ViTPoseTierABackend(device=device)
        self.joint_names = self.backend.skeleton.joint_names
        self.device = self.backend.device

    def estimate(self, image_rgb: np.ndarray, box: BoxXYWH | None) -> tuple[np.ndarray, np.ndarray]:
        if box is None:
            h, w = image_rgb.shape[:2]
            box = (0.0, 0.0, float(w), float(h))
        (kpts, conf), = self.backend.infer_frame_boxes(image_rgb, [box])
        return kpts, conf
