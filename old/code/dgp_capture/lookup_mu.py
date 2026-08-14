"""Pure-lookup mu methods: use stored mu_global, never refit RF."""

from __future__ import annotations

from typing import Any

import numpy as np

from methods.mu_methods import (
    _fit_within_group_correction,
    create_mu_method_random_forest_offset,
    create_mu_method_random_forest_residual_correction,
    global_weight,
)


class XWithMeta(np.ndarray):
    """X vector carrying (group_id, row_idx) through donor/sample-HCP calls."""

    def __new__(cls, input_array, group_id: int, row_idx: int):
        obj = np.asarray(input_array, dtype=float).view(cls)
        obj.group_id = int(group_id)
        obj.row_idx = int(row_idx)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.group_id = getattr(obj, "group_id", -1)
        self.row_idx = getattr(obj, "row_idx", -1)


class MuGlobalLookupModel:
    """Stand-in for sklearn RF: lookup μ by (group_id, row_idx)."""

    def __init__(self, fit_id: str, n_comp: int, mu_map: dict[tuple[int, int], float]):
        self._capture_fit_id = fit_id
        self._capture_n_comp = n_comp
        self._n_comp_groups = n_comp
        self._mu_map = mu_map

    def lookup(self, group_id: int, row_idx: int) -> float:
        key = (int(group_id), int(row_idx))
        if key not in self._mu_map:
            raise KeyError(f"Missing mu_global for fit_id={self._capture_fit_id} key={key}")
        return self._mu_map[key]

    def predict(self, Xu: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "LookupModel.predict() must not be called; use predict_global with XWithMeta."
        )


def attach_meta_to_z(Z_list: list[list[dict]], group_ids: list[int] | None = None) -> list[list[dict]]:
    """Return Z_list copies whose X entries are XWithMeta(group_id, row_idx)."""
    out = []
    for slot, Zg in enumerate(Z_list):
        gid = int(group_ids[slot]) if group_ids is not None else int(slot)
        out.append([
            {
                "X": XWithMeta(z["X"], gid, i),
                "Y": float(z["Y"]),
                "group_id": gid,
                "row_idx": int(i),
            }
            for i, z in enumerate(Zg)
        ])
    return out


def _lookup_mu_g(model, x_vector) -> float:
    if model is None:
        return 0.0
    if hasattr(x_vector, "group_id") and hasattr(x_vector, "row_idx"):
        return float(model.lookup(int(x_vector.group_id), int(x_vector.row_idx)))
    raise RuntimeError("predict_global requires XWithMeta (group_id, row_idx)")


def make_lookup_mu_method(
    *,
    within_group_mode: str,
    mu_by_fit_id: dict[str, dict[tuple[int, int], float]],
    experiment: int,
    call_id: str,
    local_adjustment_clip: float | None = 50_000.0,
) -> dict:
    """Build a mu_method that loads μ_RF from capture and applies within-group mode.

    within_group_mode:
      - mean: sample-mean shrinkage (default DGP RF)
      - correction: ridge residual correction
      - none: pure global (tau ignored; use tau_override=0 in HCP call)
    """
    mode = str(within_group_mode).lower()
    if mode == "correction":
        base = create_mu_method_random_forest_residual_correction(
            local_adjustment_clip=local_adjustment_clip,
        )
    else:
        # mean and none share the mean-shrinkage API; none uses tau_override=0
        base = create_mu_method_random_forest_offset()

    out = dict(base)
    fit_seq = [0]
    c = float(base.get("c", 0.5))

    def fit_global(U_matrix, Z_list, group_index_vector):
        fit_seq[0] += 1
        fit_id = f"e{experiment}_{call_id}_f{fit_seq[0]}"
        if fit_id not in mu_by_fit_id:
            raise KeyError(
                f"Missing fit_id {fit_id} in mu_global capture "
                f"(have {len(mu_by_fit_id)} fits for this experiment)"
            )
        n_comp = len(group_index_vector)
        model = MuGlobalLookupModel(fit_id, n_comp, mu_by_fit_id[fit_id])
        return model

    def predict_global(model_global, x_vector, u_vector):
        return _lookup_mu_g(model_global, x_vector)

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        idx = list(training_index_vector)
        if mode == "none" or len(idx) == 0 or model_global is None:
            gid = int(Z_group_list[0]["group_id"]) if Z_group_list else -1
            return {"mode": "none", "group_mean": 0.0, "tau": 0, "group_id": gid}

        gid = int(Z_group_list[0]["group_id"])
        if mode == "mean":
            group_mean = float(np.mean([Z_group_list[i]["Y"] for i in idx], dtype=float))
            return {"mode": "mean", "group_mean": group_mean, "tau": len(idx), "group_id": gid}

        # correction: g_j on residuals Y - mu_global
        x_local = np.array([np.asarray(Z_group_list[i]["X"], dtype=float).ravel() for i in idx])
        y_local = np.array([Z_group_list[i]["Y"] for i in idx], dtype=float)
        mu_g = np.array([
            model_global.lookup(gid, int(Z_group_list[i]["row_idx"])) for i in idx
        ], dtype=float)
        residuals = y_local - mu_g
        adj = _fit_within_group_correction(x_local, residuals, ridge_alpha=1.0)
        if local_adjustment_clip is not None:
            adj["clip"] = float(local_adjustment_clip)
        adj["mode"] = "correction"
        adj["group_id"] = gid
        adj["tau"] = len(idx)
        return adj

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        mu_g = _lookup_mu_g(model_global, x_vector)
        if not isinstance(group_adjustment, dict):
            return mu_g
        mode_adj = group_adjustment.get("mode", "mean")
        if mode_adj == "none" or mode == "none":
            return mu_g
        if mode_adj == "correction" or mode == "correction":
            # residual correction path from create_mu_method_random_forest_residual_correction
            return float(base["predict_group_mu"](model_global, group_adjustment, x_vector, u_group_vector))
        # mean shrinkage
        group_mean = float(group_adjustment.get("group_mean", 0.0))
        tau_eff = int(group_adjustment.get("tau", 0))
        if tau_eff <= 0:
            return mu_g
        n_comp = int(getattr(model_global, "_n_comp_groups", 0))
        w_g = global_weight(n_comp, tau_eff, c)
        return w_g * mu_g + (1.0 - w_g) * group_mean

    # For correction mode, predict_group_mu on base expects model.predict — override fully.
    if mode == "correction":
        def predict_group_mu_corr(model_global, group_adjustment, x_vector, u_group_vector):
            mu_g = _lookup_mu_g(model_global, x_vector)
            if not isinstance(group_adjustment, dict) or group_adjustment.get("mode") in (None, "none"):
                return mu_g
            # Replicate residual-correction predict without calling model.predict
            from methods.mu_methods import _predict_within_group_correction
            delta = _predict_within_group_correction(group_adjustment, x_vector)
            clip = group_adjustment.get("clip")
            if clip is not None:
                delta = float(np.clip(delta, -float(clip), float(clip)))
            return mu_g + float(delta)

        out["predict_group_mu"] = predict_group_mu_corr
    else:
        out["predict_group_mu"] = predict_group_mu

    out["fit_global"] = fit_global
    out["predict_global"] = predict_global
    out["fit_group_adjustment"] = fit_group_adjustment
    out["_lookup_mode"] = mode
    return out


def index_mu_global(mu_global_df) -> dict[str, dict[tuple[int, int], float]]:
    """fit_id -> {(group_id, row_idx): mu_global}."""
    out: dict[str, dict[tuple[int, int], float]] = {}
    for fit_id, sub in mu_global_df.groupby("fit_id", sort=False):
        out[str(fit_id)] = {
            (int(g), int(r)): float(m)
            for g, r, m in zip(sub["group_id"], sub["row_idx"], sub["mu_global"])
        }
    return out
