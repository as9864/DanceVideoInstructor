"""Live HUD rendering (ARCHITECTURE.md §11 "HUD + 임계값 초과 시 경고").

Korean text: OpenCV 5's built-in Unicode font (cv2.FontFace("uni")) covers
Hangul, so no system font lookup is needed. On an older OpenCV without it,
the HUD falls back to ASCII metric IDs rather than printing '?????'.
"""

import cv2
import numpy as np

from dvi.realtime.feedback import FeedbackUpdate, MetricFeedback
from dvi.realtime.metrics2d import METRIC_LABELS_KO, Keypoints2D, support_side
from dvi.schemas.metrics import TargetRange
from dvi.schemas.skeleton import CANONICAL_BONES

# RGB — frames are RGB end to end (io/video_reader.py); conversion to BGR
# happens only at the display/writer boundary.
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "ok": (60, 200, 90),
    "minor": (250, 190, 40),
    "major": (235, 60, 50),
    "unknown": (150, 150, 150),
}
BONE_COLOR = (80, 220, 255)
JOINT_COLOR = (255, 255, 255)
PANEL_BG = (20, 20, 20)
GRAVITY_COLOR = (200, 200, 200)

try:
    _UNI_FONT = cv2.FontFace("uni")
except (AttributeError, cv2.error):  # OpenCV < 5
    _UNI_FONT = None

HAS_UNICODE_FONT = _UNI_FONT is not None

UNSIGNED_METRICS = frozenset({"AXI-04"})  # a joint angle; +/- would only add noise


def put_text(image: np.ndarray, text: str, org: tuple[int, int], color, size: int = 20) -> int:
    """Draw `text` with its baseline at `org`; returns the x where it ended."""
    x, y = int(org[0]), int(org[1])
    if _UNI_FONT is not None:
        end, _ = cv2.putText(image, text, (x, y), color, _UNI_FONT, size)
        return int(end[0])
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    scale = size / 30.0
    cv2.putText(image, ascii_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, size // 12), cv2.LINE_AA)
    (w, _), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, size // 12))
    return x + w


def text_width(text: str, size: int) -> int:
    """Width of `text` as put_text would draw it, measured off-screen."""
    scratch = np.zeros((size * 2, size * (len(text) + 2), 3), dtype=np.uint8)
    return put_text(scratch, text, (0, int(size * 1.4)), (255, 255, 255), size)


def _blend_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, alpha: float = 0.6) -> None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def _label(metric_id: str) -> str:
    return METRIC_LABELS_KO.get(metric_id, metric_id) if HAS_UNICODE_FONT else metric_id


def _format_value(fb: MetricFeedback, target: TargetRange | None) -> str:
    if fb.value is None:
        return "측정 불가" if HAS_UNICODE_FONT else "n/a"
    unit = target.unit if target else ""
    sign = "" if fb.metric_id in UNSIGNED_METRICS else "+"
    if unit == "deg":
        return f"{fb.value:{sign}.1f}" + ("°" if HAS_UNICODE_FONT else "deg")
    return f"{fb.value:{sign}.2f}"


def _format_target(metric_id: str, target: TargetRange | None) -> str:
    if target is None:
        return ""
    suffix = "°" if target.unit == "deg" and HAS_UNICODE_FONT else ""
    if target.min is not None and target.max is not None:
        return f"{target.min:g}~{target.max:g}{suffix}"
    if target.max is not None:
        return f"<{target.max:g}{suffix}"
    return f">{target.min:g}{suffix}"


def draw_skeleton(image: np.ndarray, kp: Keypoints2D) -> None:
    index = {n: i for i, n in enumerate(kp.joint_names)}
    for a, b in CANONICAL_BONES:
        if a in index and b in index and kp.valid[index[a]] and kp.valid[index[b]]:
            pa = tuple(int(round(c)) for c in kp.xy[index[a]])
            pb = tuple(int(round(c)) for c in kp.xy[index[b]])
            cv2.line(image, pa, pb, BONE_COLOR, 3, cv2.LINE_AA)
    for i, ok in enumerate(kp.valid):
        if ok:
            p = tuple(int(round(c)) for c in kp.xy[i])
            cv2.circle(image, p, 4, JOINT_COLOR, -1, cv2.LINE_AA)


def draw_axis(image: np.ndarray, kp: Keypoints2D, status: str) -> None:
    """Support ankle -> trunk center (colored by AXI-06 status), next to a
    true-vertical reference from the same ankle — the gap between the two
    lines IS the axis error, readable at a glance mid-step."""
    side = support_side(kp)
    if side is None or not kp.has("shoulder_l", "shoulder_r", "hip_l", "hip_r"):
        return
    if side == "both":
        if not kp.has("ankle_l", "ankle_r"):
            return
        base = (kp.get("ankle_l") + kp.get("ankle_r")) / 2
    else:
        base = kp.get(f"ankle_{side}")
    center = (kp.get("hip_l") + kp.get("hip_r") + kp.get("shoulder_l") + kp.get("shoulder_r")) / 4
    height = max(1.0, base[1] - center[1]) * 1.25
    b = (int(base[0]), int(base[1]))
    cv2.line(image, b, (b[0], int(base[1] - height)), GRAVITY_COLOR, 1, cv2.LINE_AA)
    direction = (center - base) / max(np.linalg.norm(center - base), 1e-6)
    tip = base + direction * np.linalg.norm(center - base) * 1.25
    cv2.line(image, b, (int(tip[0]), int(tip[1])), STATUS_COLORS[status], 3, cv2.LINE_AA)


def render_hud(
    image: np.ndarray,
    kp: Keypoints2D | None,
    feedback: FeedbackUpdate | None,
    targets: dict[str, TargetRange],
    *,
    header: str = "",
    calibration_progress: float | None = None,
    fps: float | None = None,
    latency_ms: float | None = None,
) -> np.ndarray:
    """Draw skeleton, axis line, metric panel and headline onto a copy of `image`."""
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.6, min(1.6, h / 720))
    fs = int(20 * scale)  # base font size

    if kp is not None:
        draw_skeleton(out, kp)
        axis_status = "unknown"
        if feedback and "AXI-06" in feedback.metrics:
            axis_status = feedback.metrics["AXI-06"].status
        draw_axis(out, kp, axis_status)

    # Metric panel (top-left)
    if feedback is not None and feedback.metrics:
        row_h = int(fs * 1.5)
        rows = list(feedback.metrics.values())
        gap = fs
        x_label = 40
        x_value = x_label + max(text_width(_label(fb.metric_id), fs) for fb in rows) + gap
        x_target = x_value + text_width("+180.0°", fs) + gap
        panel_right = x_target + max(
            text_width(_format_target(fb.metric_id, targets.get(fb.metric_id)), int(fs * 0.8)) for fb in rows
        ) + gap
        _blend_rect(out, 8, 8, panel_right, 16 + row_h * (len(rows) + 1), PANEL_BG)
        put_text(out, header, (18, 8 + row_h - fs // 3), (255, 255, 255), int(fs * 0.85))
        for i, fb in enumerate(rows, start=1):
            y = 8 + row_h * (i + 1) - fs // 3
            color = STATUS_COLORS[fb.status]
            cv2.circle(out, (24, y - fs // 3), max(4, fs // 3), color, -1, cv2.LINE_AA)
            put_text(out, _label(fb.metric_id), (x_label, y), (255, 255, 255), fs)
            target = targets.get(fb.metric_id)
            put_text(out, _format_value(fb, target), (x_value, y), color, fs)
            put_text(out, _format_target(fb.metric_id, target), (x_target, y), (170, 170, 170), int(fs * 0.8))

    # Performance (top-right)
    perf = []
    if fps is not None:
        perf.append(f"{fps:4.1f} fps")
    if latency_ms is not None:
        perf.append(f"{latency_ms:3.0f} ms")
    if perf:
        text = "  ".join(perf)
        tw = text_width(text, int(fs * 0.8))
        _blend_rect(out, w - tw - 24, 8, w - 8, 8 + int(fs * 1.5), PANEL_BG)
        put_text(out, text, (w - tw - 16, 8 + int(fs * 1.1)), (220, 220, 220), int(fs * 0.8))

    # Headline (bottom): calibration prompt, or the top active alert
    headline, color = None, None
    if calibration_progress is not None:
        pct = int(round(calibration_progress * 100))
        headline = (
            f"기준 자세 측정 중 — 편하게 바로 서 주세요 ({pct}%)" if HAS_UNICODE_FONT else f"Calibrating... {pct}%"
        )
        color = (60, 120, 220)
    elif kp is None:
        headline = "사람이 보이지 않아요 — 전신이 화면에 들어오게 서 주세요" if HAS_UNICODE_FONT else "No person detected"
        color = STATUS_COLORS["unknown"]
    elif feedback is not None and feedback.headline is not None:
        fb = feedback.headline
        headline = fb.message if (fb.message and HAS_UNICODE_FONT) else f"{fb.metric_id}: {fb.direction}"
        color = STATUS_COLORS["major"]

    if headline:
        hs = int(fs * 1.4)
        tw = text_width(headline, hs)
        x0 = max(8, (w - tw) // 2 - 16)
        y1 = h - 16
        _blend_rect(out, x0, y1 - int(hs * 1.8), min(w - 8, x0 + tw + 32), y1, color, alpha=0.8)
        put_text(out, headline, (x0 + 16, y1 - int(hs * 0.55)), (255, 255, 255), hs)

    return out
