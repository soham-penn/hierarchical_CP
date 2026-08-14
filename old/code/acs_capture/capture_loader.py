"""Load capture CSV tables into memory-efficient structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from methods.mu_methods import (
    create_mu_method_ols_residual_correction,
    create_mu_method_random_forest_residual_correction,
    create_mu_method_ols_offset,
    create_mu_method_random_forest_offset,
    global_weight,
)
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class CaptureTables:
    root: Path
    config: dict
    observations: pd.DataFrame
    replicate_design: pd.DataFrame
    global_fits: pd.DataFrame
    mu_global: pd.DataFrame
    mu_by_fit_id: dict[str, pd.DataFrame]
    hcp_calls: pd.DataFrame
    baseline_mu_global: pd.DataFrame
    baseline_hcp_split: pd.DataFrame
    n_replicates: int


def _read_config(root: Path) -> dict:
    cfg = pd.read_csv(root / "experiment_config.csv")
    out = {}
    for _, row in cfg.iterrows():
        key = row["key"]
        val = row["value"]
        try:
            out[key] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def load_capture_tables(root: Path, *, load_mu_global: bool = True) -> CaptureTables:
    root = Path(root)
    print("Loading observations...", flush=True)
    obs = pd.read_csv(root / "observations.csv")
    n_rep = int(obs["replicate"].max()) + 1
    print("Loading replicate_design, global_fits, hcp_calls...", flush=True)
    replicate_design = pd.read_csv(root / "replicate_design.csv")
    global_fits = pd.read_csv(root / "global_fits.csv")
    hcp_calls = pd.read_csv(root / "hcp_calls.csv")
    baseline_mu_global = pd.read_csv(root / "baseline_mu_global.csv")
    baseline_hcp_split = pd.read_csv(root / "baseline_hcp_split.csv")
    mu_global = pd.DataFrame()
    mu_by_fit_id: dict[str, pd.DataFrame] = {}
    if load_mu_global:
        print("Loading mu_global.csv (large; may take several minutes)...", flush=True)
        mu_global = pd.read_csv(
            root / "mu_global.csv",
            usecols=["fit_id", "replicate", "call_id", "puma", "row_idx", "mu_global"],
        )
        print(f"  mu_global rows: {len(mu_global):,}; indexing by fit_id...", flush=True)
        mu_by_fit_id = {str(k): v for k, v in mu_global.groupby("fit_id", sort=False)}
    print("Capture tables loaded.", flush=True)
    return CaptureTables(
        root=root,
        config=_read_config(root),
        observations=obs,
        replicate_design=replicate_design,
        global_fits=global_fits,
        mu_global=mu_global,
        mu_by_fit_id=mu_by_fit_id,
        hcp_calls=hcp_calls,
        baseline_mu_global=baseline_mu_global,
        baseline_hcp_split=baseline_hcp_split,
        n_replicates=n_rep,
    )


def build_group_data_from_observations(
    obs: pd.DataFrame, replicate: int, pumas: list[int],
) -> dict:
    feat_cols = [c for c in obs.columns if c.startswith("f_")]
    group_data = {}
    sub = obs[obs["replicate"] == replicate]
    for puma in pumas:
        g = sub[sub["puma"] == puma].sort_values("row_idx")
        group_data[puma] = {
            "X": g[feat_cols].to_numpy(dtype=float),
            "Y": g["y"].to_numpy(dtype=float),
            "income": g["income"].to_numpy(dtype=float),
            "row_idx": g["row_idx"].to_numpy(dtype=int),
        }
    return group_data


class _MuGlobalLookupModel:
    """Stand-in for sklearn RF: predict() uses stored mu_global by (puma, row_idx)."""

    def __init__(
        self,
        fit_id: str,
        n_comp: int,
        mu_table: pd.DataFrame,
    ):
        self._capture_fit_id = fit_id
        self._capture_n_comp = n_comp
        self._n_comp_groups = n_comp
        sub = mu_table
        self._mu_map = {
            (int(puma), int(row_idx)): float(mu)
            for puma, row_idx, mu in zip(sub["puma"], sub["row_idx"], sub["mu_global"])
        }

    def lookup(self, puma: int, row_idx: int) -> float:
        return self._mu_map.get((int(puma), int(row_idx)), 0.0)

    def predict(self, Xu: np.ndarray) -> np.ndarray:
        # Fallback for callers without puma/row_idx context (rare in replay path).
        Xu = np.asarray(Xu, dtype=float)
        if Xu.ndim == 1:
            return np.array([0.0])
        return np.zeros(len(Xu), dtype=float)


def get_mu_for_replicate(
    replicate: int,
    *,
    within_group_mode: str,
    global_predictor: str,
) -> dict:
    if within_group_mode == "correction":
        if global_predictor == "ols":
            base = create_mu_method_ols_residual_correction(local_adjustment_clip=50_000.0)
        else:
            base = create_mu_method_random_forest_residual_correction(local_adjustment_clip=50_000.0)
    else:
        if global_predictor == "ols":
            base = create_mu_method_ols_offset()
        else:
            base = create_mu_method_random_forest_offset()
    out = dict(base)
    out["_global_predictor"] = global_predictor
    return out


def build_mu_from_capture(
    tables: CaptureTables,
    *,
    within_group_mode: str,
    global_predictor: str,
):
    """Return callable(replicate) -> base mu_method dict (wrap with _wrap_mu_lookup before use)."""

    def for_replicate(replicate: int) -> dict:
        return get_mu_for_replicate(
            replicate,
            within_group_mode=within_group_mode,
            global_predictor=global_predictor,
        )

    return for_replicate
