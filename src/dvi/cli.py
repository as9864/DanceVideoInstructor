"""dvi — DanceVideoInstructor CLI entry point (ARCHITECTURE.md §9)."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

# Windows consoles often default to a non-UTF-8 codepage (e.g. cp949 on
# Korean locales); coaching output (FR-08) is Korean text, so force UTF-8
# stdout/stderr instead of crashing on the first non-ASCII character.
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _not_implemented(command: str, milestone: str) -> None:
    typer.echo(f"'{command}' is not implemented yet (planned for {milestone}).")
    raise typer.Exit(code=1)


@app.command("check-env")
def check_env() -> None:
    """Verify Python/CUDA/torch setup (see scripts/check_env.py)."""
    import runpy

    script = Path(__file__).resolve().parents[2] / "scripts" / "check_env.py"
    runpy.run_path(str(script), run_name="__main__")


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Input video file (mp4/mov)"),
    style: str = typer.Option("salon", help="Style profile: salon | milonguero | nuevo"),
    backend: str = typer.Option("tier_a", help="Pose backend tier: tier_a | tier_b | tier_c"),
    out: Path = typer.Option(..., help="Output session directory"),
    resume: bool = typer.Option(False, help="Reuse cached pose artifacts, recompute metrics only"),
) -> None:
    """Run the full analysis pipeline on a video (PROJECT.md S1)."""
    _not_implemented("analyze", "M1-M5")


@app.command()
def calibrate(
    video: Path = typer.Argument(...),
    vertical_hint: str = typer.Option(None, help="Two pixel coords marking a known vertical line"),
) -> None:
    """Manual gravity calibration fallback (ARCHITECTURE.md section 5.2 priority 4)."""
    _not_implemented("calibrate", "M2-M3")


@app.command()
def overlay(
    session: Path = typer.Argument(..., help="Session output directory"),
    metrics: str = typer.Option(None, help="Comma-separated metric IDs to render"),
) -> None:
    """Render skeleton + metric HUD overlay video for a session."""
    _not_implemented("overlay", "M1")


@app.command()
def compare(
    session_a: Path = typer.Argument(...),
    session_b: Path = typer.Argument(...),
) -> None:
    """Compare metrics across two sessions for the same subject (FR-16)."""
    _not_implemented("compare", "M5")


@app.command("reference-compare")
def reference_compare(
    session: Path = typer.Argument(...),
    ref: Path = typer.Option(..., help="Reference clip directory"),
) -> None:
    """DTW-align a session against a reference clip (PROJECT.md S3, M7)."""
    _not_implemented("reference-compare", "M7")


@app.command()
def live(
    source: str = typer.Option(
        "0", help="Camera index (0, 1, ...), phone IP-camera/RTSP URL, or a video file to replay"
    ),
    view: str = typer.Option(
        "side", help="Camera placement: side (forward lean, head, knee) | front (side lean, shoulder tilt)"
    ),
    style: str = typer.Option("salon", help="Style profile: salon | milonguero | nuevo"),
    backend: str = typer.Option("tier_a", help="Pose backend tier (only tier_a is causal/fast enough)"),
    display: str = typer.Option("auto", help="auto | window | browser | none"),
    host: str = typer.Option("127.0.0.1", help="Browser display bind address (0.0.0.0 = viewable from a phone)"),
    port: int = typer.Option(8765, help="Browser display port"),
    mirror: bool = typer.Option(None, help="Mirror the image (default: on for a local webcam)"),
    vertical_hint: str = typer.Option(
        None, help='Camera roll fix: "x1,y1,x2,y2" pixels on a truly vertical line (door frame, pillar)'
    ),
    calibration_s: float = typer.Option(3.0, help="Seconds of neutral standing to record the shoulder baseline"),
    max_long_edge: int = typer.Option(960, help="Downscale frames to this long edge before inference"),
    beep: bool = typer.Option(True, help="Sound a short beep when a warning fires"),
    record: Path = typer.Option(None, help="Also save the HUD video to this .mp4"),
    max_frames: int = typer.Option(None, help="Stop after N frames (testing)"),
    device: str = typer.Option(None, help="torch device override (cuda / cpu)"),
) -> None:
    """Real-time camera feedback on tango posture and axis (PROJECT.md S4, ARCHITECTURE.md §11)."""
    from dvi.config.loader import load_defaults, load_live_messages, load_style_targets
    from dvi.geometry.gravity import manual_calibration_from_vertical_line
    from dvi.io.webcam import open_source
    from dvi.pipeline import stream_runner
    from dvi.realtime import display as displays
    from dvi.realtime.metrics2d import CameraView

    if backend != "tier_a":
        typer.echo(f"backend {backend!r} is not causal/real-time capable; only tier_a is supported for live.")
        raise typer.Exit(code=2)
    try:
        camera_view = CameraView(view)
        targets = load_style_targets(style)
    except ValueError as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=2)

    roll_deg = 0.0
    if vertical_hint:
        try:
            x1, y1, x2, y2 = (float(v) for v in vertical_hint.split(","))
            roll_deg = manual_calibration_from_vertical_line((x1, y1), (x2, y2)).metadata["roll_deg"]
        except ValueError as e:
            typer.echo(f"error: --vertical-hint must be 'x1,y1,x2,y2' ({e})")
            raise typer.Exit(code=2)

    segmentation = load_defaults().get("segmentation", {})

    typer.echo("loading pose model (Tier A ViTPose, first run downloads weights)...")
    from dvi.realtime.estimators import ViTPoseLiveEstimator
    from dvi.realtime.session import LiveSession

    estimator = ViTPoseLiveEstimator(device=device)
    typer.echo(f"pose model on {estimator.device}")

    try:
        frames = open_source(source, max_long_edge=max_long_edge, mirror=mirror)
    except (RuntimeError, FileNotFoundError) as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(code=1)
    mirrored = bool(getattr(frames, "mirror", False))

    session = LiveSession(
        estimator,
        targets,
        load_live_messages(),
        view=camera_view,
        roll_deg=roll_deg,
        calibration_s=calibration_s,
        mirrored=mirrored,
        hold_s=float(segmentation.get("min_duration_s", 0.3)),
        release_s=float(segmentation.get("merge_gap_s", 0.5)),
    )

    if display == "auto":
        display = "window" if displays.gui_available() else "browser"
    if display == "window":
        out: displays.Display = displays.WindowDisplay()
        typer.echo("window display: q = quit, c = re-measure neutral posture")
    elif display == "browser":
        out = displays.BrowserDisplay(host=host, port=port)
        typer.echo(f"open {out.url} (q = quit, c = re-measure neutral posture, Ctrl+C here also quits)")
    elif display == "none":
        out = displays.NullDisplay()
    else:
        typer.echo(f"error: unknown display {display!r}")
        raise typer.Exit(code=2)

    def on_cue(fb) -> None:
        typer.echo(f"[{fb.metric_id}] {fb.message or fb.direction}")
        if beep:
            stream_runner.beep()

    typer.echo(f"style={style} view={camera_view.value} — stand relaxed for {calibration_s:g}s to calibrate")
    try:
        stats = stream_runner.run_live(
            frames,
            session,
            out,
            header=f"{style} · {camera_view.value}",
            record_path=record,
            record_fps=float(getattr(frames, "fps", 30.0)),
            max_frames=max_frames,
            on_cue=on_cue,
        )
    finally:
        frames.close()
        out.close()

    p50, p95 = stats.latency_p50_p95()
    typer.echo(
        f"{stats.frames} frames, {stats.fps:.1f} fps, latency p50={p50:.0f}ms p95={p95:.0f}ms, "
        f"{len(stats.cues)} warnings"
    )
    if record:
        typer.echo(f"wrote {record}")


@app.command("eval")
def eval_run(
    clips: Path = typer.Option(Path("eval/clips.yaml"), help="Validation clip manifest"),
) -> None:
    """Run the metric validation harness (PROJECT.md section 7, M6)."""
    _not_implemented("eval run", "M6")


if __name__ == "__main__":
    app()
