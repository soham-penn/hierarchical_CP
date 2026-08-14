"""Instrument mu_method dicts to log every global fit and prediction to CSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from methods.mu_methods import global_weight


@dataclass
class CaptureCallContext:
    replicate: int
    call_id: str
    hcp_method: str
    o: int
    test_puma: int
    calib_pumas: list[int]
    group_data: dict
    fit_seq: int = 0

    def slot_to_puma(self, slot: int, *, n_calib: int, test_slot: int | None) -> int:
        if test_slot is not None and slot == test_slot:
            return int(self.test_puma)
        return int(self.calib_pumas[slot])


def _stack_u(X_mat, u_vector):
    X_mat = np.asarray(X_mat, dtype=float)
    u = np.asarray(u_vector, dtype=float).ravel()
    U_rep = np.repeat(u.reshape(1, -1), len(X_mat), axis=0)
    return np.hstack([X_mat, U_rep])


def wrap_mu_for_capture(
    mu_method: dict,
    *,
    store,
    ctx: CaptureCallContext,
    n_calib: int,
    test_slot: int | None,
) -> dict:
    """Return a shallow copy of mu_method that logs fits and point predictions."""
    out = dict(mu_method)
    orig_fit = mu_method["fit_global"]
    orig_pred_global = mu_method["predict_global"]
    orig_pred_group = mu_method["predict_group_mu"]
    orig_fit_adj = mu_method["fit_group_adjustment"]
    pred_log: list[dict] = []

    def fit_global(U_matrix, Z_list, group_index_vector):
        ctx.fit_seq += 1
        model = orig_fit(U_matrix, Z_list, group_index_vector)
        fit_id = f"r{ctx.replicate}_{ctx.call_id}_f{ctx.fit_seq}"
        comp_slots = [int(g) for g in group_index_vector]
        mu_rows = []
        n_comp = len(comp_slots)
        # Store μ_RF for every row in every group slot in this HCP call (calib + test +
        # complement), so later apply can swap within-group correction without refitting RF.
        from methods.mu_methods import _predict_batch
        for slot, Zg in enumerate(Z_list):
            if not Zg:
                continue
            puma = ctx.slot_to_puma(slot, n_calib=n_calib, test_slot=test_slot)
            Xg = np.array([z["X"] for z in Zg], dtype=float)
            u = U_matrix[slot, :]
            mu_batch = _predict_batch(model, Xg, u)
            for row_idx, mu_val in enumerate(mu_batch):
                mu_rows.append({
                    "fit_id": fit_id,
                    "replicate": int(ctx.replicate),
                    "call_id": ctx.call_id,
                    "puma": int(puma),
                    "row_idx": int(row_idx),
                    "mu_global": float(mu_val),
                })
        store.write_global_fit(
            fit_id=fit_id,
            replicate=ctx.replicate,
            call_id=ctx.call_id,
            hcp_method=ctx.hcp_method,
            o=ctx.o,
            test_puma=ctx.test_puma,
            comp_slots=comp_slots,
            calib_pumas=ctx.calib_pumas,
            test_puma_slot=test_slot,
            mu_rows=mu_rows,
        )
        model._capture_fit_id = fit_id  # type: ignore[attr-defined]
        model._capture_n_comp = n_comp  # type: ignore[attr-defined]
        return model

    def predict_global(model_global, x_vector, u_vector):
        return float(orig_pred_global(model_global, x_vector, u_vector))

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        mu_g = orig_pred_global(model_global, x_vector, u_group_vector)
        mu_pred = float(orig_pred_group(model_global, group_adjustment, x_vector, u_group_vector))
        if not isinstance(group_adjustment, dict):
            return mu_pred
        group_mean = float(group_adjustment.get("group_mean", 0.0))
        tau_eff = int(group_adjustment.get("tau", 0))
        n_comp = int(getattr(model_global, "_capture_n_comp", 0))
        c = float(mu_method.get("c", 0.5))
        w_g = float(global_weight(n_comp, tau_eff, c)) if tau_eff > 0 else 1.0
        pred_log.append({
            "replicate": int(ctx.replicate),
            "call_id": ctx.call_id,
            "hcp_method": ctx.hcp_method,
            "o": int(ctx.o),
            "test_puma": int(ctx.test_puma),
            "fit_id": getattr(model_global, "_capture_fit_id", ""),
            "mu_global": mu_g,
            "group_mean": group_mean,
            "tau": tau_eff,
            "w_g": w_g,
            "mu_pred": mu_pred,
        })
        return mu_pred

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        return orig_fit_adj(model_global, u_group_vector, Z_group_list, training_index_vector)

    out["fit_global"] = fit_global
    out["predict_global"] = predict_global
    out["predict_group_mu"] = predict_group_mu
    out["fit_group_adjustment"] = fit_group_adjustment
    out["_capture_pred_log"] = pred_log
    return out


def drain_pred_log(mu_wrapped: dict) -> list[dict]:
    log = mu_wrapped.pop("_capture_pred_log", [])
    return list(log)
