"""Experimental two-person axis comparison from a single camera.

Not a pipeline stage — a tunable script for trying out whether
detect-then-per-box-pose can separate two dancers well enough to compare
their axes, ahead of ARCHITECTURE.md §13-10 (Track + RoleAssigner, M4).

Background (see also docs/PROGRESS.md §4.1's real-footage finding):
running Tier A (ViTPose) on a whole-frame box fails on couple footage —
either it locks onto one figure (often a mirror reflection) or blends two
overlapping people into one nonsense skeleton. The academic literature on
reconstructing two people in close contact from a SINGLE image is BUDDI
("Generative Proxemics", CVPR 2024, github.com/muelea/buddi) — a diffusion
prior over two people's SMPL-X parameters, built for exactly hugs/embraces.
It needs the same gated SMPL-X body model as GVHMR (docs/licenses.md §2,
manual registration at smpl-x.is.tue.mpg.de) and operates on single images,
not natively on video, so it's a heavier follow-up, not this script.

This script instead does the practical, fully-open-license thing available
right now: RT-DETRv2 (person detection, ADR-0001 C4-1) finds N person boxes
per frame, then the ALREADY-BUILT ViTPose backend runs per box. There's no
identity tracking across frames yet (that's Track, M4) — each frame is
detected independently, so "person 0"/"person 1" can swap between frames
if boxes cross or scores flip. Good enough for a first look, not for a
real per-subject time series.

Axis here means the 2D image-plane torso-lean angle (hip_mid -> neck vs.
image-up), NOT the gravity-corrected WORLD_G POS-01 — there's no 3D lift
or gravity alignment in this 2D-only path.

Usage:
    uv run python scripts/two_person_axis.py VIDEO [OPTIONS]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import typer

from dvi.io.video_reader import VideoReader
from dvi.io.video_writer import VideoWriter
from dvi.perception.detect.rtdetr import RTDETRPersonDetector
from dvi.perception.pose.tier_a_vitpose import ViTPoseTierABackend
from dvi.schemas.skeleton import CANONICAL_BONES

app = typer.Typer(add_completion=False)

PERSON_COLORS = [(255, 128, 0), (0, 200, 255), (255, 0, 200), (128, 255, 0)]  # RGB, per detection slot


def _lean_angle_deg(kpts: np.ndarray, conf: np.ndarray, joint_index: dict[str, int], conf_threshold: float) -> float | None:
    needed = ("hip_l", "hip_r", "neck")
    if any(conf[joint_index[n]] < conf_threshold for n in needed):
        return None
    hip_mid = (kpts[joint_index["hip_l"]] + kpts[joint_index["hip_r"]]) / 2
    v = kpts[joint_index["neck"]] - hip_mid
    # image y grows downward, so "up" is -y; angle from vertical, +right-lean
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def _draw_person(image: np.ndarray, kpts: np.ndarray, conf: np.ndarray, joint_names: tuple[str, ...],
                  conf_threshold: float, color: tuple[int, int, int], label: str) -> None:
    joint_index = {name: i for i, name in enumerate(joint_names)}
    valid = conf >= conf_threshold

    for name_a, name_b in CANONICAL_BONES:
        if name_a not in joint_index or name_b not in joint_index:
            continue
        ia, ib = joint_index[name_a], joint_index[name_b]
        if not (valid[ia] and valid[ib]):
            continue
        pa = tuple(int(round(c)) for c in kpts[ia])
        pb = tuple(int(round(c)) for c in kpts[ib])
        cv2.line(image, pa, pb, color=color, thickness=2, lineType=cv2.LINE_AA)

    for name, idx in joint_index.items():
        if not valid[idx]:
            continue
        p = tuple(int(round(c)) for c in kpts[idx])
        cv2.circle(image, p, radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

    if valid[joint_index["neck"]]:
        p = tuple(int(round(c)) for c in kpts[joint_index["neck"]])
        cv2.putText(image, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


@app.command()
def main(
    video: Path = typer.Argument(..., help="Input video (couple footage)"),
    out: Path = typer.Option(None, help="Output overlay path (default: <video>_two_person.mp4 next to input)"),
    score_threshold: float = typer.Option(0.5, help="RT-DETRv2 person-detection confidence threshold"),
    max_persons: int = typer.Option(2, help="Keep only the top-N highest-confidence person boxes per frame"),
    keypoint_conf_threshold: float = typer.Option(0.3, help="ViTPose keypoint confidence to count as valid"),
    max_long_edge: int = typer.Option(960, help="Downscale frames to this long edge before inference"),
    roi: str = typer.Option(
        None,
        help='Restrict detection to "x,y,w,h" pixels (post max-long-edge scaling) — '
        "use this to crop out a mirror or bystanders that the detector also picks up",
    ),
    stride: int = typer.Option(1, help="Process every Nth frame (speed vs. coverage while experimenting)"),
    max_frames: int = typer.Option(None, help="Stop after this many processed frames"),
) -> None:
    out = out or video.parent / f"{video.stem}_two_person.mp4"
    roi_box = tuple(float(v) for v in roi.split(",")) if roi else None  # (x, y, w, h)

    typer.echo(f"reading {video} (max_long_edge={max_long_edge}, stride={stride})...")
    reader = VideoReader(video, max_long_edge=max_long_edge)

    typer.echo("loading RT-DETRv2 (person detection) + ViTPose...")
    detector = RTDETRPersonDetector(score_threshold=score_threshold, max_persons=max_persons)
    pose_backend = ViTPoseTierABackend()
    joint_names = pose_backend.skeleton.joint_names
    joint_index = {name: i for i, name in enumerate(joint_names)}

    lean_series: list[list[float | None]] = []
    pelvis_distance_series: list[float | None] = []

    with VideoWriter(out, fps=reader.fps / stride, width=reader.width, height=reader.height) as writer:
        t0 = time.time()
        n_processed = 0
        for frame in reader:
            if frame.index % stride != 0:
                continue
            if max_frames is not None and n_processed >= max_frames:
                break

            image = frame.image.copy()
            detect_image = image
            offset = np.array([0.0, 0.0])
            if roi_box is not None:
                x, y, w, h = (int(v) for v in roi_box)
                detect_image = image[y : y + h, x : x + w]
                offset = np.array([x, y])

            detections = detector.detect(detect_image)
            boxes = [
                (d.box_xywh[0] + offset[0], d.box_xywh[1] + offset[1], d.box_xywh[2], d.box_xywh[3])
                for d in detections
            ]

            frame_leans: list[float | None] = []
            pelvis_points: list[np.ndarray] = []

            if boxes:
                results = pose_backend.infer_frame_boxes(image, boxes)
                for i, (kpts, conf) in enumerate(results):
                    color = PERSON_COLORS[i % len(PERSON_COLORS)]
                    _draw_person(image, kpts, conf, joint_names, keypoint_conf_threshold, color, f"p{i}")
                    frame_leans.append(_lean_angle_deg(kpts, conf, joint_index, keypoint_conf_threshold))
                    if conf[joint_index["hip_l"]] >= keypoint_conf_threshold and conf[joint_index["hip_r"]] >= keypoint_conf_threshold:
                        pelvis_points.append((kpts[joint_index["hip_l"]] + kpts[joint_index["hip_r"]]) / 2)

            lean_series.append(frame_leans)
            if len(pelvis_points) == 2:
                pelvis_distance_series.append(float(np.linalg.norm(pelvis_points[0] - pelvis_points[1])))
            else:
                pelvis_distance_series.append(None)

            if roi_box is not None:
                x, y, w, h = (int(v) for v in roi_box)
                cv2.rectangle(image, (x, y), (x + w, y + h), (80, 80, 80), 1, cv2.LINE_AA)

            writer.write(image)
            n_processed += 1

        elapsed = time.time() - t0

    typer.echo(f"processed {n_processed} frames in {elapsed:.1f}s ({n_processed / elapsed:.1f} fps)")
    for i in range(max_persons):
        vals = [frame[i] for frame in lean_series if len(frame) > i and frame[i] is not None]
        if vals:
            typer.echo(f"person {i}: lean angle mean={np.mean(vals):.1f} deg, std={np.std(vals):.1f} deg, n={len(vals)}")
        else:
            typer.echo(f"person {i}: no valid readings")

    dist_vals = [d for d in pelvis_distance_series if d is not None]
    if dist_vals:
        typer.echo(
            f"inter-pelvis pixel distance: mean={np.mean(dist_vals):.1f}px, std={np.std(dist_vals):.1f}px, "
            f"n={len(dist_vals)}/{n_processed} frames had exactly 2 people"
        )
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
