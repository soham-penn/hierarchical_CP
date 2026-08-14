"""CSV artifact writers for ACS capture experiments."""

from __future__ import annotations

import json
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path

import numpy as np
import pandas as pd


class CaptureStore:
    """Append-only CSV store for one capture experiment root."""

    def __init__(self, root: Path, *, lock: LockType | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = lock
        self._feature_names: list[str] | None = None

    def set_feature_names(self, names: list[str]) -> None:
        self._feature_names = list(names)
        pd.DataFrame({"feature_idx": range(len(names)), "feature_name": names}).to_csv(
            self.root / "feature_names.csv", index=False
        )

    def write_config(self, config: dict) -> None:
        rows = [{"key": str(k), "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
                for k, v in sorted(config.items())]
        pd.DataFrame(rows).to_csv(self.root / "experiment_config.csv", index=False)

    def _append(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
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
        replicate: int,
        puma: int,
        group_data: dict,
    ) -> None:
        if self._feature_names is None:
            raise RuntimeError("feature_names must be set before writing observations")
        rows = []
        n_feat = len(self._feature_names)
        for row_idx in range(len(group_data[puma]["Y"])):
            feats = group_data[puma]["X"][row_idx]
            row = {
                "replicate": int(replicate),
                "puma": int(puma),
                "row_idx": int(row_idx),
                "y": float(group_data[puma]["Y"][row_idx]),
                "income": float(group_data[puma]["income"][row_idx]),
            }
            for j in range(n_feat):
                row[f"f_{j}"] = float(feats[j])
            rows.append(row)
        self._append("observations", rows)

    def write_replicate_design(
        self,
        *,
        replicate: int,
        calib_groups: list[int],
        test_groups: list[int],
    ) -> None:
        rows = []
        for ord_i, puma in enumerate(calib_groups):
            rows.append({
                "replicate": int(replicate),
                "puma": int(puma),
                "role": "calib",
                "calib_ord": int(ord_i),
            })
        for puma in test_groups:
            rows.append({
                "replicate": int(replicate),
                "puma": int(puma),
                "role": "test",
                "calib_ord": -1,
            })
        self._append("replicate_design", rows)

    def write_baseline_fit(
        self,
        *,
        replicate: int,
        predictor: str,
        train_slots: list[int],
        calib_slots: list[int],
        calib_pumas: list[int],
        mu_global_rows: list[dict],
        ols_coef: np.ndarray | None = None,
        ols_intercept: float | None = None,
    ) -> None:
        self._append("baseline_hcp_split", [{
            "replicate": int(replicate),
            "predictor": predictor,
            "train_slots_json": json.dumps([int(x) for x in train_slots]),
            "calib_slots_json": json.dumps([int(x) for x in calib_slots]),
        }])
        for slot in train_slots:
            puma = int(calib_pumas[slot])
            self._append("baseline_train_rows", [{
                "replicate": int(replicate),
                "puma": puma,
                "calib_slot": int(slot),
            }])
        if ols_coef is not None:
            self._append("baseline_ols_intercept", [{
                "replicate": int(replicate),
                "intercept": float(ols_intercept),
            }])
            for j, coef in enumerate(ols_coef):
                self._append("baseline_ols_coef", [{
                    "replicate": int(replicate),
                    "feature_idx": int(j),
                    "coef": float(coef),
                }])
        for row in mu_global_rows:
            self._append("baseline_mu_global", [row])

    def write_global_fit(
        self,
        *,
        fit_id: str,
        replicate: int,
        call_id: str,
        hcp_method: str,
        o: int,
        test_puma: int,
        comp_slots: list[int],
        calib_pumas: list[int],
        test_puma_slot: int | None,
        mu_rows: list[dict],
    ) -> None:
        comp_pumas = []
        for slot in comp_slots:
            if test_puma_slot is not None and slot == test_puma_slot:
                comp_pumas.append(int(test_puma))
            else:
                comp_pumas.append(int(calib_pumas[slot]))
        self._append("global_fits", [{
            "fit_id": fit_id,
            "replicate": int(replicate),
            "call_id": call_id,
            "hcp_method": hcp_method,
            "o": int(o),
            "test_puma": int(test_puma),
            "comp_slots_json": json.dumps([int(s) for s in comp_slots]),
            "comp_pumas_json": json.dumps(comp_pumas),
            "n_comp": len(comp_slots),
        }])
        self._append("mu_global", mu_rows)

    def write_mu_pred(
        self,
        rows: list[dict],
    ) -> None:
        self._append("mu_predictions", rows)

    def append_mu_global(self, rows: list[dict]) -> None:
        """Append μ_RF rows to mu_global.csv (e.g. backfill calib/test during apply)."""
        self._append("mu_global", rows)

    def write_hcp_call(
        self,
        row: dict,
    ) -> None:
        self._append("hcp_calls", [row])

    def write_hcp_scores(
        self,
        rows: list[dict],
    ) -> None:
        self._append("hcp_scores", rows)

    def flush(self) -> None:
        pass
