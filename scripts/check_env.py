"""Environment verification for DanceVideoInstructor (M0).

Checks the things ARCHITECTURE.md §9 and PROJECT.md R8 flag as fragile:
Python version, NVIDIA driver/CUDA, and (if installed) a working
CUDA-enabled torch build for the Blackwell (RTX 50xx) GPU generation.

Run with: uv run scripts/check_env.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys

REQUIRED_PYTHON = (3, 12)


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) == REQUIRED_PYTHON
    status = "OK" if ok else "MISMATCH"
    print(f"[{status}] Python {major}.{minor} (required: {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x)")
    return ok


def check_nvidia_driver() -> bool:
    if shutil.which("nvidia-smi") is None:
        print("[FAIL] nvidia-smi not found on PATH — no NVIDIA driver detected")
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[FAIL] nvidia-smi failed: {exc}")
        return False

    if not out:
        print("[FAIL] nvidia-smi returned no GPU")
        return False

    for line in out.splitlines():
        print(f"[OK] GPU: {line.strip()}")
    return True


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        print(
            "[SKIP] torch not installed yet. RTX 50xx (Blackwell, sm_120) needs a "
            "CUDA 12.8+ build. Install with, e.g.:\n"
            "    uv add torch --index-url https://download.pytorch.org/whl/cu128"
        )
        return False

    print(f"[INFO] torch {torch.__version__} (built for CUDA {torch.version.cuda})")

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False - CUDA build/driver mismatch")
        return False

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"[OK] CUDA available - device: {device_name}, compute capability: {capability}")

    try:
        x = torch.randn(4, 4, device="cuda")
        (x @ x).sum().item()
        print("[OK] CUDA matmul smoke test passed")
    except Exception as exc:  # noqa: BLE001 — smoke test, report any failure
        print(f"[FAIL] CUDA smoke test raised: {exc}")
        return False

    return True


def main() -> int:
    # Windows consoles often default to a non-UTF-8 codepage (e.g. cp949 on
    # Korean locales); this tool prints Korean text elsewhere (FR-08 coaching
    # messages), so force UTF-8 stdout instead of crashing on the first
    # non-ASCII character it hits.
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    print("=== DanceVideoInstructor environment check ===")
    results = {
        "python": check_python(),
        "nvidia_driver": check_nvidia_driver(),
        "torch_cuda": check_torch(),
    }
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{name}: {'OK' if ok else 'NOT READY'}")

    # torch_cuda is allowed to be unready at M0 (not installed yet); the rest must pass.
    hard_requirements = ["python", "nvidia_driver"]
    return 0 if all(results[k] for k in hard_requirements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
