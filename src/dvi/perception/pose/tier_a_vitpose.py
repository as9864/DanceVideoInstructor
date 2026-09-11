"""Tier A pose backend: ViTPose (ADR-0001 §3 M1 scaffold candidate).

usyd-community/vitpose-plus-base outputs COCO-17 (no feet) — fine for the
M1 milestone ("skeleton renders on a single-person video"), not for AXI-*
metrics. Swap for a Halpe-26 model (e.g. RTMPose ONNX) before M3, per
ADR-0001 §3.

Runs one whole-frame box per image — no detector is wired up yet (that's
Detect, a separate later milestone), so this only makes sense for
single-person footage for now.
"""

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, VitPoseForPoseEstimation

from dvi.io.video_reader import VideoFrame
from dvi.perception.pose.base import Pose2D
from dvi.perception.pose.skeleton import COCO17_SKELETON_SPEC, map_coco17_to_canonical

DEFAULT_MODEL_ID = "usyd-community/vitpose-plus-base"

# vitpose-plus-base is a mixture-of-experts backbone trained on 6 pose
# datasets and requires a dataset_index at inference time. Its id2label is
# the 17 COCO keypoint names, and the ViTPose++ paper's canonical dataset
# ordering is COCO(0)/AIC(1)/MPII(2)/AP-10K(3)/APT-36K(4)/WholeBody(5) --
# so 0 selects the COCO-17 expert this backend is built around.
COCO_DATASET_INDEX = 0


class ViTPoseTierABackend:
    name = "vitpose_plus_base"
    provides_3d = False
    provides_world_frame = False
    skeleton = COCO17_SKELETON_SPEC

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, frames: Sequence[VideoFrame]) -> Pose2D:
        ts: list[float] = []
        joints_px: list[np.ndarray] = []
        confidence: list[np.ndarray] = []
        width = height = 0

        for vf in frames:
            image = Image.fromarray(vf.image)
            width, height = image.size
            box = [[0.0, 0.0, float(width), float(height)]]  # whole-frame box, COCO xywh

            inputs = self.processor(image, boxes=[box], return_tensors="pt").to(self.device)
            dataset_index = torch.tensor([COCO_DATASET_INDEX], device=self.device)
            outputs = self.model(**inputs, dataset_index=dataset_index)
            result = self.processor.post_process_pose_estimation(outputs, boxes=[box])[0][0]

            kpts = result["keypoints"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            canon_kpts, canon_conf, _ = map_coco17_to_canonical(kpts, scores)

            ts.append(vf.t)
            joints_px.append(canon_kpts)
            confidence.append(canon_conf)

        confidence_arr = np.stack(confidence)
        return Pose2D(
            subject_id="subject_0",
            t=np.array(ts),
            joints_px=np.stack(joints_px),
            confidence=confidence_arr,
            valid=confidence_arr >= 0.3,
            skeleton=COCO17_SKELETON_SPEC,
            image_size=(width, height),
        )
