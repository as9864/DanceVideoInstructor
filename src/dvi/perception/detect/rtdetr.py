"""RT-DETRv2 person detector (ADR-0001 §4 C4-1).

Apache-2.0, transformers-native — chosen specifically to avoid Ultralytics
YOLO's AGPL-3.0 (docs/licenses.md §3: incompatible with the FastAPI/viewer
plans in ARCHITECTURE.md §2).
"""

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

from dvi.perception.detect.base import Detection

DEFAULT_MODEL_ID = "PekingU/rtdetr_v2_r50vd"


class RTDETRPersonDetector:
    name = "rtdetr_v2_r50vd"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        score_threshold: float = 0.5,
        max_persons: int | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.max_persons = max_persons

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self._person_label_id = next(i for i, name in self.model.config.id2label.items() if name == "person")

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list[Detection]:
        pil_image = Image.fromarray(image)
        width, height = pil_image.size

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=torch.tensor([[height, width]])
        )[0]

        detections = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            if int(label) != self._person_label_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(box_xywh=(x1, y1, x2 - x1, y2 - y1), score=float(score), label="person")
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        if self.max_persons is not None:
            detections = detections[: self.max_persons]
        return detections

    def detect_video(self, frames: Sequence) -> list[list[Detection]]:
        return [self.detect(f.image) for f in frames]
