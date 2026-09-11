"""Evaluation harness (PROJECT.md §7, ADR-0001 §5).

There's no ground truth in this domain (PROJECT.md R4) — "is POS-01 = 7.3°
correct" is unanswerable without a motion-capture rig. What IS answerable:
does the same motion, filmed from a different angle, produce the same
number? That's ADR-0001 §5's headline evaluation axis for the M2 backend
bake-off (V2 in PROJECT.md §7), and this module implements exactly that
comparison — nothing else yet, since V4 (label correlation) needs expert-
tagged clips that don't exist until PROJECT.md §7 V4 labeling happens.

Clips being compared are assumed simultaneously recorded (ADR-0001 §5 step
1: "동일 동작을 정면/45°/측면 동시 촬영"), so they're already frame-
synchronized — no DTW alignment is attempted here. Once clips aren't
synchronized (e.g. comparing separate takes), that alignment step belongs
in reference/ (ARCHITECTURE.md §7.4), not here.
"""

from dataclasses import dataclass

import numpy as np

from dvi.schemas.metrics import MetricResult


@dataclass
class AngleConsistencyResult:
    metric_id: str
    n_angles: int
    n_frames_compared: int  # frames valid in every angle
    per_frame_std: np.ndarray  # (n_frames_compared,) cross-angle std at each such frame
    mean_std: float  # ADR-0001 §5's headline comparison number
    max_std: float
    coverage: float  # fraction of the synchronized timeline that was usable


def angle_consistency(results: list[MetricResult]) -> AngleConsistencyResult:
    """One MetricResult per camera angle, same metric_id, same synchronized clip.

    Only frames valid in *every* angle are compared — a frame occluded in
    one angle isn't evidence against the other angles' agreement.
    """
    if len(results) < 2:
        raise ValueError("need at least 2 angles to measure consistency")

    metric_id = results[0].metric_id
    if any(r.metric_id != metric_id for r in results):
        raise ValueError(f"all results must share metric_id; got {[r.metric_id for r in results]}")

    lengths = {r.series.shape[0] for r in results}
    if len(lengths) != 1:
        raise ValueError(
            f"angle series have mismatched lengths {sorted(lengths)} — clips must be "
            "synchronized (ADR-0001 §5 step 1: simultaneous recording)"
        )
    t_len = lengths.pop()

    stacked_series = np.stack([r.series for r in results])  # (A, T)
    stacked_valid = np.stack([r.valid for r in results])  # (A, T)
    all_valid = np.all(stacked_valid, axis=0)  # (T,)
    n_frames_compared = int(all_valid.sum())

    if n_frames_compared == 0:
        return AngleConsistencyResult(
            metric_id=metric_id,
            n_angles=len(results),
            n_frames_compared=0,
            per_frame_std=np.array([]),
            mean_std=float("nan"),
            max_std=float("nan"),
            coverage=0.0,
        )

    per_frame_std = stacked_series[:, all_valid].std(axis=0)

    return AngleConsistencyResult(
        metric_id=metric_id,
        n_angles=len(results),
        n_frames_compared=n_frames_compared,
        per_frame_std=per_frame_std,
        mean_std=float(per_frame_std.mean()),
        max_std=float(per_frame_std.max()),
        coverage=n_frames_compared / t_len,
    )
