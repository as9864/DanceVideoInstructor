"""Manual smoke test for the Tier A (ViTPose) backend + overlay renderer.

Not part of the automated test suite — needs network (HF model download,
cached after the first run) and downloads ~350MB of weights the first time.
This is the shortest path to seeing the pipeline work on a real clip before
the full `dvi analyze` command exists (that's M1-M5, ARCHITECTURE.md §13).

Single-person footage only for now — no detector is wired up yet, so this
always uses a whole-frame box (ADR-0001 §3 M1 scope).

Usage:
    uv run python scripts/tier_a_smoke.py <video_path> [out_dir]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvi.io.video_reader import VideoReader  # noqa: E402
from dvi.perception.pose.tier_a_vitpose import ViTPoseTierABackend  # noqa: E402
from dvi.report.overlay import render_overlay  # noqa: E402

MAX_LONG_EDGE = 960  # keep inference fast; render_overlay rescales back to source res


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    video_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else video_path.parent
    out_path = out_dir / f"{video_path.stem}_overlay.mp4"

    print(f"reading {video_path} (max_long_edge={MAX_LONG_EDGE})...")
    frames = list(VideoReader(video_path, max_long_edge=MAX_LONG_EDGE))
    print(f"{len(frames)} frames")
    if not frames:
        print("no frames decoded — is this a valid video file?")
        return 1

    print("loading ViTPose (usyd-community/vitpose-plus-base)...")
    t0 = time.time()
    backend = ViTPoseTierABackend()
    print(f"loaded on {backend.device} in {time.time() - t0:.1f}s")

    t0 = time.time()
    pose = backend.infer(frames)
    elapsed = time.time() - t0
    print(f"inference: {elapsed:.1f}s for {len(frames)} frames ({len(frames) / elapsed:.1f} fps)")
    print(f"mean confidence: {pose.confidence.mean():.3f}, coverage: {pose.valid.mean():.1%}")

    render_overlay(video_path, pose, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
