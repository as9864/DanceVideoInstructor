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
    backend: str = typer.Option("tier_a", help="Pose backend tier (causal only)"),
    style: str = typer.Option("salon"),
) -> None:
    """Real-time webcam feedback (PROJECT.md S4, M8)."""
    _not_implemented("live", "M8")


@app.command("eval")
def eval_run(
    clips: Path = typer.Option(Path("eval/clips.yaml"), help="Validation clip manifest"),
) -> None:
    """Run the metric validation harness (PROJECT.md section 7, M6)."""
    _not_implemented("eval run", "M6")


if __name__ == "__main__":
    app()
