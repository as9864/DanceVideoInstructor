"""Artifact cache store (ARCHITECTURE.md §10).

"캐싱이 개발 속도를 결정한다" — pose extraction is slow, metric tuning is fast
and repeated hundreds of times. Keying stage output on (input hash + the
subset of config that stage actually reads) is what makes that split real
instead of aspirational.

Pickle is fine here: this cache is written and read only by the same local
user on the same machine (NFR-05), never exposed to untrusted input.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ArtifactStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def load(self, key: str) -> Any:
        with open(self._path_for(key), "rb") as f:
            return pickle.load(f)

    def save(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(value, f)
        tmp.replace(path)  # atomic on the same filesystem — no torn cache entries
