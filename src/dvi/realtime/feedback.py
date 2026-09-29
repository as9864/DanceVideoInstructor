"""Threshold -> on-screen/audible cue logic for the live runner (ARCHITECTURE.md §11).

A raw per-frame threshold check would flicker: pose jitter crosses a
bound for one frame, a warning flashes, and the dancer learns to ignore
it. So each metric runs a small hysteresis state machine, the causal
counterpart of the offline Finding rules (ARCHITECTURE.md §7.2):

  - an alert fires only after the value stays out of range (major) for
    `hold_s` — default.yaml's segmentation.min_duration_s, the same
    "ignore spikes shorter than this" rule the offline path uses;
  - it clears only after `release_s` back in range (merge_gap_s);
  - a sound/voice cue is emitted when an alert fires, at most once per
    `cooldown_s` per metric, so a sustained fault doesn't beep every step.
"""

from dataclasses import dataclass, field
from typing import Literal

from dvi.realtime.metrics2d import ABSOLUTE_METRICS
from dvi.schemas.metrics import TargetRange

Status = Literal["ok", "minor", "major", "unknown"]
Direction = Literal["high", "low"]

# How far past a bound still counts as "minor" when the style profile sets
# no explicit tolerance. Hypotheses, like every target (PROJECT.md §7).
DEFAULT_MINOR_BAND: dict[str, float] = {"deg": 2.0, "normalized": 0.03}

# Most fundamental first: when several alerts are active, the HUD headline
# shows the top one — fixing the axis usually fixes the rest.
PRIORITY: tuple[str, ...] = ("AXI-06", "POS-01", "POS-02", "POS-04", "POS-05a", "POS-05b", "AXI-04")


def judge(metric_id: str, value: float | None, target: TargetRange) -> tuple[Status, Direction | None]:
    if value is None:
        return "unknown", None
    v = abs(value) if metric_id in ABSOLUTE_METRICS else value
    if target.max is not None and v > target.max:
        excess, direction = v - target.max, "high"
    elif target.min is not None and v < target.min:
        excess, direction = target.min - v, "low"
    else:
        return "ok", None
    band = target.tolerance or DEFAULT_MINOR_BAND.get(target.unit, 0.0)
    return ("minor" if excess <= band else "major"), direction


@dataclass
class MetricFeedback:
    metric_id: str
    value: float | None = None
    status: Status = "unknown"
    direction: Direction | None = None
    alert: bool = False
    message: str | None = None
    _bad_since: float | None = None
    _good_since: float | None = None
    _last_cue_t: float | None = None


@dataclass
class FeedbackUpdate:
    metrics: dict[str, MetricFeedback]
    headline: MetricFeedback | None  # highest-priority active alert
    cues: list[MetricFeedback] = field(default_factory=list)  # alerts that just fired (play a sound)


class FeedbackEngine:
    def __init__(
        self,
        targets: dict[str, TargetRange],
        messages: dict[str, dict[str, str]],
        *,
        hold_s: float = 0.3,
        release_s: float = 0.5,
        cooldown_s: float = 4.0,
        mirrored: bool = False,
    ):
        self.targets = targets
        self.messages = messages
        self.hold_s = hold_s
        self.release_s = release_s
        self.cooldown_s = cooldown_s
        # The live runner feeds pose the mirrored frame in self-view mode; then
        # image-right is the dancer's right. Without mirroring it's their left.
        self.mirrored = mirrored
        self._state: dict[str, MetricFeedback] = {}

    def reset(self) -> None:
        self._state.clear()

    def _message(self, fb: MetricFeedback) -> str | None:
        templates = self.messages.get(fb.metric_id, {})
        template = templates.get(fb.direction or "high")
        if template is None:
            return None
        image_right = (fb.value or 0.0) > 0
        dancer_right = image_right if self.mirrored else not image_right
        return template.format(side="오른쪽" if dancer_right else "왼쪽")

    def update(self, values: dict[str, float | None], t: float) -> FeedbackUpdate:
        cues: list[MetricFeedback] = []
        for metric_id, value in values.items():
            target = self.targets.get(metric_id)
            fb = self._state.setdefault(metric_id, MetricFeedback(metric_id))
            fb.value = value
            if target is None:
                fb.status, fb.direction = "unknown", None
                continue
            status, direction = judge(metric_id, value, target)
            direction_changed = direction != fb.direction
            fb.status, fb.direction = status, direction

            if status == "major":
                fb._good_since = None
                if fb._bad_since is None or direction_changed:
                    fb._bad_since = t
                if not fb.alert and t - fb._bad_since >= self.hold_s:
                    fb.alert = True
                    if fb._last_cue_t is None or t - fb._last_cue_t >= self.cooldown_s:
                        fb._last_cue_t = t
                        cues.append(fb)
                if fb.alert:
                    fb.message = self._message(fb)  # refreshed: the side can flip mid-alert
            else:
                # ok / minor / unknown all count toward releasing an alert
                fb._bad_since = None
                if fb._good_since is None:
                    fb._good_since = t
                if fb.alert and t - fb._good_since >= self.release_s:
                    fb.alert = False
                    fb.message = None

        # Metrics not reported this frame (view change / recalibration) drop out.
        for metric_id in list(self._state):
            if metric_id not in values:
                del self._state[metric_id]

        active = [self._state[m] for m in PRIORITY if m in self._state and self._state[m].alert]
        return FeedbackUpdate(metrics=dict(self._state), headline=active[0] if active else None, cues=cues)

    def update_absent(self, t: float) -> FeedbackUpdate:
        """Nobody in frame: every tracked metric reads unknown, so alerts age
        out on the normal release timer instead of freezing on screen."""
        return self.update({m: None for m in self._state}, t)
