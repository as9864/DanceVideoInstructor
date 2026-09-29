"""Stream runner — the live counterpart of batch_runner.py (ARCHITECTURE.md §3.1, §11).

capture -> LiveSession (pose, One-Euro, 2D metrics, feedback) -> HUD ->
display/recording/sound. Frame-at-a-time and strictly causal: nothing
here waits for future frames.
"""

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from dvi.io.video_writer import VideoWriter
from dvi.realtime.display import Display
from dvi.realtime.feedback import MetricFeedback
from dvi.realtime.hud import render_hud
from dvi.realtime.session import LiveSession


class FrameSource(Protocol):
    def read(self, timeout_s: float = 2.0) -> tuple[np.ndarray, float] | None: ...
    def now(self) -> float: ...
    def close(self) -> None: ...


@dataclass
class LiveStats:
    frames: int = 0
    wall_s: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    cues: list[tuple[float, str, str | None]] = field(default_factory=list)  # (t, metric_id, message)

    @property
    def fps(self) -> float:
        return self.frames / self.wall_s if self.wall_s > 0 else 0.0

    def latency_p50_p95(self) -> tuple[float, float]:
        if not self.latencies_ms:
            return float("nan"), float("nan")
        arr = np.array(self.latencies_ms)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def beep() -> None:
    """Short, non-blocking alert sound (PROJECT.md S4 "화면/사운드 경고")."""
    if sys.platform == "win32":
        import winsound

        threading.Thread(target=winsound.Beep, args=(880, 150), daemon=True).start()
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def run_live(
    source: FrameSource,
    session: LiveSession,
    display: Display,
    *,
    header: str = "",
    record_path: Path | None = None,
    record_fps: float = 30.0,
    max_frames: int | None = None,
    on_cue: Callable[[MetricFeedback], None] | None = None,
) -> LiveStats:
    stats = LiveStats()
    writer: VideoWriter | None = None
    frame_times: deque[float] = deque(maxlen=30)
    t_start = time.monotonic()
    try:
        while max_frames is None or stats.frames < max_frames:
            cmd = display.poll_command()
            if cmd == "quit":
                break
            if cmd == "recalibrate":
                session.recalibrate()

            got = source.read()
            if got is None:
                break
            image, t = got
            t_proc = time.monotonic()
            result = session.process(image, t)

            if result.feedback is not None:
                for fb in result.feedback.cues:
                    stats.cues.append((t, fb.metric_id, fb.message))
                    if on_cue is not None:
                        on_cue(fb)

            frame_times.append(time.monotonic())
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else None
            # HUD shows the previous frame's end-to-end latency (this one isn't finished yet).
            latency_ms = stats.latencies_ms[-1] if stats.latencies_ms else None
            hud = render_hud(
                image,
                result.keypoints,
                result.feedback,
                session.targets,
                header=header,
                calibration_progress=result.calibration_progress,
                fps=fps,
                latency_ms=latency_ms,
            )
            display.show(hud)
            # Live source: capture clock -> on screen. Clip: processing time only.
            stats.latencies_ms.append(max(source.now() - t, time.monotonic() - t_proc) * 1000.0)

            if record_path is not None:
                if writer is None:
                    record_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = hud.shape[:2]
                    writer = VideoWriter(record_path, fps=record_fps, width=w - w % 2, height=h - h % 2)
                writer.write(hud[: hud.shape[0] - hud.shape[0] % 2, : hud.shape[1] - hud.shape[1] % 2])
            stats.frames += 1
    except KeyboardInterrupt:
        pass
    finally:
        stats.wall_s = time.monotonic() - t_start
        if writer is not None:
            writer.close()
    return stats
