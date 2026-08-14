"""
ACS score-swap cache helpers (v2).

Prefer ``recompute_acs_scores_from_cache.py`` for CLI replay.
This module keeps lightweight path/serialize helpers used by the runner docs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 2


def cache_path(cache_dir: Path, replicate_idx: int) -> Path:
    return Path(cache_dir) / f"rep_{int(replicate_idx):05d}.pkl"


def save_cache(cache_dir: Path, replicate_idx: int, payload: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["version"] = CACHE_VERSION
    payload["replicate_idx"] = int(replicate_idx)
    with cache_path(cache_dir, replicate_idx).open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(cache_dir: Path, replicate_idx: int) -> dict[str, Any]:
    with cache_path(cache_dir, replicate_idx).open("rb") as f:
        return pickle.load(f)


def serialize_group_data(group_data: dict) -> dict:
    return {
        int(grp): {
            "X": np.asarray(data["X"], dtype=float),
            "Y": np.asarray(data["Y"], dtype=float),
            "income": np.asarray(data["income"], dtype=float),
        }
        for grp, data in group_data.items()
    }
