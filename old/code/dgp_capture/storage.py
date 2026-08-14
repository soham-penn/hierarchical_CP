"""CSV artifact writers for DGP RF capture experiments."""

from __future__ import annotations

import json
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path

import numpy as np
import pandas as pd


class DgpCaptureStore:
    """Append-only CSV store for one DGP capture experiment root."""

    def __init__(self, root: Path, *, lock: LockType | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = lock
        self._x_names: list[str] | None = None
        self._u_names: list[str] | None = None

    def set_feature_names(self, x_names: list[str], u_names: list[str]) -> None:
        self._x_names = list(x_names)
        self._u_names = list(u_names)
        rows = (
            [{"kind": "x", "feature_idx": i, "feature_name": n} for i, n in enumerate(x_names)]
            + [{"kind": "u", "feature_idx": i, "feature_name": n} for i, n in enumerate(u_names)]
        )
        pd.DataFrame(rows).to_csv(self.root / "feature_names.csv", index=False)

    def write_config(self, config: dict) -> None:
        rows = [
            {"key": str(k), "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
            for k, v in sorted(config.items())
        ]
        pd.DataFrame(rows).to_csv(self.root / "experiment_config.csv", index=False)

    def _append(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        # Defensive: recreate root if another process wiped capture dirs mid-run.
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{table}.csv"
        df = pd.DataFrame(rows)
        if self._lock is not None:
            self._lock.acquire()
        try:
            header = not path.exists()
            df.to_csv(path, mode="a", header=header, index=False)
        finally:
            if self._lock is not None:
                self._lock.release()

    def write_observations(
        self,
        *,
        experiment: int,
        group_id: int,
        role: str,
        calib_ord: int,
        U_group: np.ndarray,
        Z_group: list[dict],
    ) -> None:
        if self._x_names is None or self._u_names is None:
            raise RuntimeError("feature_names must be set before writing observations")
        u = np.asarray(U_group, dtype=float).ravel()
        rows = []
        for row_idx, z in enumerate(Z_group):
            x = np.asarray(z["X"], dtype=float).ravel()
            row = {
                "experiment": int(experiment),
                "group_id": int(group_id),
                "role": role,
                "calib_ord": int(calib_ord),
                "row_idx": int(row_idx),
                "y": float(z["Y"]),
            }
            for j, name in enumerate(self._x_names):
                row[name] = float(x[j])
            for j, name in enumerate(self._u_names):
                row[name] = float(u[j])
            rows.append(row)
        self._append("observations", rows)

    def write_experiment_design(
        self,
        *,
        experiment: int,
        n_calib: int,
        n_test: int = 1,
    ) -> None:
        rows = [
            {
                "experiment": int(experiment),
                "group_id": int(g),
                "role": "calib",
                "calib_ord": int(g),
            }
            for g in range(n_calib)
        ]
        for t in range(n_test):
            rows.append({
                "experiment": int(experiment),
                "group_id": int(n_calib + t),
                "role": "test",
                "calib_ord": -1,
            })
        self._append("experiment_design", rows)

    def write_baseline_fit(
        self,
        *,
        experiment: int,
        train_slots: list[int],
        calib_slots: list[int],
        mu_global_rows: list[dict],
    ) -> None:
        self._append("baseline_hcp_split", [{
            "experiment": int(experiment),
            "train_slots_json": json.dumps([int(x) for x in train_slots]),
            "calib_slots_json": json.dumps([int(x) for x in calib_slots]),
        }])
        self._append("baseline_mu_global", mu_global_rows)

    def write_global_fit(
        self,
        *,
        fit_id: str,
        experiment: int,
        call_id: str,
        hcp_method: str,
        o: int,
        comp_slots: list[int],
        n_groups_total: int,
        mu_rows: list[dict],
    ) -> None:
        self._append("global_fits", [{
            "fit_id": fit_id,
            "experiment": int(experiment),
            "call_id": call_id,
            "hcp_method": hcp_method,
            "o": int(o),
            "comp_slots_json": json.dumps([int(s) for s in comp_slots]),
            "n_comp": len(comp_slots),
            "n_groups_total": int(n_groups_total),
        }])
        self._append("mu_global", mu_rows)

    def write_mu_pred(self, rows: list[dict]) -> None:
        self._append("mu_predictions", rows)

    def write_hcp_call(self, row: dict) -> None:
        self._append("hcp_calls", [row])


class NullDgpCaptureStore:
    """No-op store for results-only runs (no μ-capture CSV I/O)."""

    def set_feature_names(self, x_names: list[str], u_names: list[str]) -> None:
        return None

    def write_config(self, config: dict) -> None:
        return None

    def write_observations(self, **kwargs) -> None:
        return None

    def write_experiment_design(self, **kwargs) -> None:
        return None

    def write_baseline_fit(self, **kwargs) -> None:
        return None

    def write_global_fit(self, **kwargs) -> None:
        return None

    def write_mu_pred(self, rows: list[dict]) -> None:
        return None

    def write_hcp_call(self, row: dict) -> None:
        return None
