"""Self-contained demo of the POS-*/AXI-* metrics engine + the V2
(camera-angle-consistency) evaluation harness.

No GPU, no model download, no video file needed — this builds a synthetic
3-second "walk" (PROJECT.md §7 V1: exact, hand-known geometry, the same
technique tests/fixtures/synthetic/skeletons.py uses for unit tests) and
runs it through the real metric functions in src/dvi/metrics/. It exists
to answer a question the rest of the repo can't yet: since `dvi analyze`
isn't wired up (docs/PROGRESS.md §2, M3 column) and the real Tier A
backend's COCO-17 output can't feed AXI-01/06 (no foot joints, ADR-0001
§2 C1), this is the shortest path to seeing every implemented metric run
end-to-end and produce numbers.

The synthetic sequence has three 1-second segments, all built with
build_synthetic_standing_pose() (exact torso pitch/roll/kyphosis, so the
"true" POS-01/02/03 values are known by construction):

    1. normal salon lean, left foot supporting
    2. a deliberate "collapse": torso pitch AND spine kyphosis both climb,
       support knee sinks, right foot supporting
    3. recovery back toward segment 1's numbers, left foot supporting

Segment 2 is designed to demonstrate the POS-01-vs-POS-03 distinction
ARCHITECTURE.md §6.1 calls out: pitch (whole-torso lean) and kyphosis
(spine kink) are varied independently, so a real "collapse" shows up as
BOTH metrics moving, not one.

Usage:
    uv run python samples/metrics_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# Windows consoles often default to a non-UTF-8 codepage (e.g. cp949 on Korean
# locales) and this demo prints em-dashes/degree signs (same fix as cli.py).
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
import yaml

from dvi.geometry.support_foot import SupportFootResult
from dvi.metrics import axis, posture
from dvi.schemas import BodyScale, Frame, MetricResult, PoseSequence, Provenance, TargetRange
from dvi.schemas.skeleton import DVI_CANONICAL_V1
from eval.harness import angle_consistency
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose

FPS = 30.0
SEGMENT_FRAMES = 30  # 1s/segment @30fps, 3 segments = 90 frames total


def _lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def _segment_params(frame_idx: int) -> tuple[float, float, float, float, str]:
    """(pitch_deg, roll_deg, kyphosis_deg, support_knee_bend_deg, support_side) for one frame."""
    seg, local = divmod(frame_idx, SEGMENT_FRAMES)
    frac = local / (SEGMENT_FRAMES - 1)
    roll = 2.0 * np.sin(2 * np.pi * frame_idx / SEGMENT_FRAMES)  # small constant sway -> POS-02 demo

    if seg == 0:  # normal salon lean, left support
        return 3.0, roll, 0.0, 0.0, "left"
    if seg == 1:  # deliberate collapse: pitch AND kyphosis both climb, support leg sinks, right support
        return _lerp(3.0, 9.0, frac), roll, _lerp(0.0, 14.0, frac), _lerp(0.0, 8.0, frac), "right"
    return _lerp(9.0, 4.0, frac), roll, _lerp(14.0, 2.0, frac), _lerp(8.0, 0.0, frac), "left"  # recovery


def _apply_support_knee_bend(joints: np.ndarray, side: str, bend_deg: float) -> None:
    """Re-place the support-side ankle so hip-knee-ankle = 180 - bend_deg exactly
    (same hand-derivable construction as tests/unit/test_metrics_axis.py)."""
    if bend_deg == 0.0:
        return
    knee_idx = DVI_CANONICAL_V1.joint_index(f"knee_{side}")
    ankle_idx = DVI_CANONICAL_V1.joint_index(f"ankle_{side}")
    knee = joints[knee_idx]
    bend_rad = np.radians(bend_deg)
    joints[ankle_idx] = knee + 0.45 * np.array([0.0, -np.cos(bend_rad), np.sin(bend_rad)])


def build_demo_sequence(n_frames: int = 3 * SEGMENT_FRAMES) -> tuple[PoseSequence, SupportFootResult]:
    skeleton = DVI_CANONICAL_V1
    frame_joints = []
    support_states = []
    for i in range(n_frames):
        pitch, roll, kyphosis, knee_bend, support = _segment_params(i)
        single_frame_pose = build_synthetic_standing_pose(
            torso_pitch_deg=pitch, torso_roll_deg=roll, spine_kyphosis_deg=kyphosis
        )
        joints = single_frame_pose.joints[0].copy()
        _apply_support_knee_bend(joints, "l" if support == "left" else "r", knee_bend)
        frame_joints.append(joints)
        support_states.append(support)

    joints = np.stack(frame_joints)  # (T, J, 3)
    t = np.arange(n_frames, dtype=float) / FPS
    pose = PoseSequence(
        subject_id="demo",
        frame=Frame.WORLD_G,
        fps=FPS,
        t=t,
        joints=joints,
        confidence=np.ones((n_frames, skeleton.n_joints)),
        valid=np.ones((n_frames, skeleton.n_joints), dtype=bool),
        skeleton=skeleton,
        scale=BodyScale(torso_length_m=0.5, height_estimate_m=1.7),
        provenance=Provenance(backend_name="synthetic-demo", backend_version="0", config_hash="demo"),
    )
    state = np.array(support_states, dtype=object)
    support = SupportFootResult(
        state=state, left_grounded=state == "left", right_grounded=state == "right", floor_y=0.0
    )
    return pose, support


def load_style_targets(style_path: Path) -> dict[str, TargetRange]:
    raw = yaml.safe_load(style_path.read_text(encoding="utf-8"))
    return {metric_id: TargetRange(**spec) for metric_id, spec in raw["targets"].items()}


def _out_of_range_fraction(result: MetricResult, target: TargetRange) -> float:
    """Not part of MetricResult today (ARCHITECTURE.md §4.5 only specs mean/p50/p95/...
    in `aggregate`) — computed here to show the kind of pass/fail quantification
    a Finding-detection stage (§7.2, M5) would consume."""
    observed = result.series[result.valid]
    if observed.size == 0:
        return float("nan")
    lo = -np.inf if target.min is None else target.min
    hi = np.inf if target.max is None else target.max
    return float(np.mean((observed < lo) | (observed > hi)))


_METRIC_FNS = {
    "POS-01": lambda pose, support, target: posture.pos01(pose, target),
    "POS-02": lambda pose, support, target: posture.pos02(pose, target),
    "POS-03": lambda pose, support, target: posture.pos03(pose, target),
    "POS-04": lambda pose, support, target: posture.pos04(pose, target),
    "POS-05a": lambda pose, support, target: posture.pos05a(pose, target),
    "POS-05b": lambda pose, support, target: posture.pos05b(pose, target),
    "POS-06": lambda pose, support, target: posture.pos06(pose, target),
    "AXI-01": lambda pose, support, target: axis.axi01(pose, support, target),
    "AXI-02": lambda pose, support, target: axis.axi02(pose, support, target),
    "AXI-04": lambda pose, support, target: axis.axi04(pose, support, target),
    "AXI-06": lambda pose, support, target: axis.axi06(pose, support, target),
}


def print_metrics_table(pose: PoseSequence, support: SupportFootResult, targets: dict[str, TargetRange]) -> None:
    header = f"{'metric':8} {'unit':11} {'coverage':>9} {'mean':>8} {'p95':>8} {'out-of-range':>13}"
    print(header)
    print("-" * len(header))
    for metric_id, fn in _METRIC_FNS.items():
        target = targets[metric_id]
        result = fn(pose, support, target)
        oor = _out_of_range_fraction(result, target)
        print(
            f"{metric_id:8} {target.unit:11} {result.coverage:9.1%} "
            f"{result.aggregate['mean']:8.3f} {result.aggregate['p95']:8.3f} {oor:13.1%}"
        )


def demo_angle_consistency(pose: PoseSequence, target: TargetRange, rng: np.random.Generator) -> None:
    """Stands in for PROJECT.md §7 V2 / ADR-0001 §5's headline evaluation axis:
    does the same motion, seen from different camera angles, produce the same
    metric value? Real usage needs simultaneously-filmed multi-angle footage,
    which doesn't exist yet (docs/PROGRESS.md §4.2) — here the "3 angles" are
    the true POS-01 series plus independent per-angle Gaussian noise, standing
    in for viewpoint-dependent estimation error (side views are harder for a
    sagittal-plane angle than front views, hence the larger std)."""
    true_series, valid = posture.compute_pos01_series(pose)
    noise_std_deg_by_angle = {"front": 1.0, "45deg": 2.2, "side": 3.5}

    results = []
    for angle_name, std in noise_std_deg_by_angle.items():
        noisy = true_series + rng.normal(0.0, std, size=true_series.shape)
        results.append(
            MetricResult(
                metric_id="POS-01",
                series=noisy,
                valid=valid,
                coverage=float(valid.mean()),
                target=target,
                aggregate={"mean": float(noisy[valid].mean())},
                causal=True,
            )
        )

    consistency = angle_consistency(results)
    print(f"\nangle_consistency(POS-01, {len(results)} synthetic viewpoints: {list(noise_std_deg_by_angle)}):")
    print(
        f"  mean_std={consistency.mean_std:.2f} deg, max_std={consistency.max_std:.2f} deg, "
        f"coverage={consistency.coverage:.1%}"
    )
    print(
        "  (this only demonstrates the harness on synthetic noise, not a real camera-robustness"
        " result — ADR-0001 §5 needs real simultaneous multi-angle footage to answer that.)"
    )


def main() -> None:
    pose, support = build_demo_sequence()
    targets = load_style_targets(_REPO_ROOT / "configs" / "styles" / "salon.yaml")

    print("synthetic 3-second sequence: normal stance -> deliberate torso collapse -> recovery")
    print(f"{pose.joints.shape[0]} frames @ {pose.fps:.0f} fps, style profile = salon\n")
    print_metrics_table(pose, support, targets)

    rng = np.random.default_rng(0)
    demo_angle_consistency(pose, targets["POS-01"], rng)


if __name__ == "__main__":
    main()
