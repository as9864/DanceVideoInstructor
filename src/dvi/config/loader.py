"""YAML config loading for configs/ (ARCHITECTURE.md §8)."""

from pathlib import Path
from typing import Any

import yaml

from dvi.schemas.metrics import TargetRange

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_style_targets(style: str, config_dir: Path = CONFIG_DIR) -> dict[str, TargetRange]:
    path = config_dir / "styles" / f"{style}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (config_dir / "styles").glob("*.yaml"))
        raise ValueError(f"unknown style profile {style!r} (available: {', '.join(available)})")
    raw = load_yaml(path).get("targets", {})
    return {metric_id: TargetRange(**spec) for metric_id, spec in raw.items()}


def load_live_messages(config_dir: Path = CONFIG_DIR, lang: str = "ko") -> dict[str, dict[str, str]]:
    return load_yaml(config_dir / "coaching" / f"{lang}.yaml").get("live", {})


def load_defaults(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    return load_yaml(config_dir / "default.yaml")
