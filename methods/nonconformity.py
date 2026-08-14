"""
Alternate nonconformity scores for hierarchical CP.

score_type:
  absolute     — |Y - μ|  (default)
  studentized  — |Y - μ| / σ̂(x,u) with RF scale models
  cqr          — max(q_lo - Y, Y - q_hi) with quantile models on [X|U]

Studentized dual-scale blend (GHCP / mean-shrinkage μ)
------------------------------------------------------
The shrunk center is
  μ̂ = w_g μ_g + (1 - w_g) μ_ℓ ,
with the same global weight as the mean,
  w_g = N_comp^c / (N_comp^c + τ)   (default c = 1/2).

Residuals then satisfy the convex identity
  Y - μ̂ = w_g (Y - μ_g) + (1 - w_g) (Y - μ_ℓ),
so by the triangle inequality
  E|Y - μ̂| ≤ w_g E|Y - μ_g| + (1 - w_g) E|Y - μ_ℓ|.
Matching that bound, we fit two RF scale models
  σ_g ≈ E[|Y - μ_g| | x,u] ,   σ_ℓ ≈ E[|Y - μ_ℓ| | x,u]
and studentize with the *same* convex weights as the mean:
  σ̂ = w_g σ_g + (1 - w_g) σ_ℓ .
(Using √w weights would misalign scale shrinkage with location shrinkage;
the L1 bound above selects the mean weights.)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


EPS = 1e-6


def get_score_type(mu_method) -> str:
    raw = str((mu_method or {}).get("score_type", "absolute")).lower().replace("-", "_")
    if raw in ("standardized", "studentised", "studentized", "std", "sigma"):
        return "studentized"
    if raw in ("cqr", "conformalized_quantile", "quantile"):
        return "cqr"
    return "absolute"


def _rf_hyperparams(mu_method) -> dict:
    ntree = int((mu_method or {}).get("rf_ntree", 50))
    nodesize = int((mu_method or {}).get("rf_nodesize", 5))
    mtry = (mu_method or {}).get("rf_mtry", None)
    rs = int((mu_method or {}).get("rf_random_state", 123))
    return {"ntree": ntree, "nodesize": nodesize, "mtry": mtry, "random_state": rs}


def _resolve_mtry(mtry, n_features: int):
    if mtry is None:
        return max(1, int(np.sqrt(n_features)))
    if isinstance(mtry, float) and 0 < mtry <= 1:
        return mtry  # sklearn fraction
    return int(mtry)


def _design_matrix(U_all, Z_all, group_index_vector):
    feats, y = [], []
    for g in group_index_vector:
        g = int(g)
        if g < 0 or g >= len(Z_all):
            continue
        Ug = np.asarray(U_all[g, :], dtype=float).ravel()
        for z in Z_all[g]:
            x = np.asarray(z["X"], dtype=float).ravel()
            feats.append(np.concatenate([x, Ug]))
            y.append(float(z["Y"]))
    if not feats:
        return None, None
    return np.asarray(feats, dtype=float), np.asarray(y, dtype=float)


def _feat(x_vector, u_vector) -> np.ndarray:
    x = np.asarray(x_vector, dtype=float).ravel()
    u = np.asarray(u_vector, dtype=float).ravel()
    return np.concatenate([x, u]).reshape(1, -1)


def _fit_rf_scale(hp: dict, X_fit: np.ndarray, y_fit: np.ndarray):
    max_features = _resolve_mtry(hp["mtry"], X_fit.shape[1])
    rf = RandomForestRegressor(
        n_estimators=hp["ntree"],
        max_features=max_features,
        min_samples_leaf=max(1, min(hp["nodesize"], len(y_fit))),
        random_state=hp["random_state"] + 17,
        n_jobs=1,
    )
    rf.fit(X_fit, y_fit)
    return rf


def _within_abs_residuals(U_all, Z_all, group_index_vector, mu_method, global_model, tau: int):
    """Absolute residuals under within-group μ on indices tau:N_j."""
    feats, abs_resid = [], []
    for g in group_index_vector:
        g = int(g)
        if g < 0 or g >= len(Z_all):
            continue
        Zg = Z_all[g]
        Ng = len(Zg)
        if Ng <= tau:
            continue
        Ug = np.asarray(U_all[g, :], dtype=float).ravel()
        offset = mu_method["fit_group_adjustment"](
            model_global=global_model,
            u_group_vector=Ug,
            Z_group_list=Zg,
            training_index_vector=list(range(tau)),
        )
        for i in range(tau, Ng):
            z = Zg[i]
            mu = float(mu_method["predict_group_mu"](
                model_global=global_model,
                group_adjustment=offset,
                x_vector=z["X"],
                u_group_vector=Ug,
            ))
            feats.append(_feat(z["X"], Ug).ravel())
            abs_resid.append(abs(float(z["Y"]) - mu))
    if not abs_resid:
        return None, None
    return np.asarray(feats, dtype=float), np.asarray(abs_resid, dtype=float)


def fit_score_aux(
    U_all,
    Z_all,
    group_index_vector,
    mu_method,
    global_model,
    alpha: float,
    *,
    tau: int = 0,
):
    """Fit auxiliary models needed for the configured score type.

    Studentized: always fit a *global* RF-σ on |Y − μ_global|.  When tau > 0,
    also fit a *local/within* RF-σ on within-adjusted absolute residuals.
    At scoring time, ``predict_scale`` blends them with the same w_g used for μ.
    """
    st = get_score_type(mu_method)
    if st == "absolute" or global_model is None or len(group_index_vector) == 0:
        return {"score_type": "absolute"}

    hp = _rf_hyperparams(mu_method)
    tau = int(max(0, tau))

    if st == "studentized":
        Xg, yg = _design_matrix(U_all, Z_all, group_index_vector)
        if Xg is None or len(yg) < 20:
            return {"score_type": "absolute"}
        mu_g = np.asarray(global_model.predict(Xg), dtype=float)
        rf_global = _fit_rf_scale(hp, Xg, np.abs(yg - mu_g))

        rf_within = None
        if tau > 0:
            Xw, yw = _within_abs_residuals(
                U_all, Z_all, group_index_vector, mu_method, global_model, tau,
            )
            if Xw is not None and len(yw) >= 20:
                rf_within = _fit_rf_scale(hp, Xw, yw)

        return {
            "score_type": "studentized",
            "scale_model": rf_global,  # back-compat alias (= global)
            "scale_model_global": rf_global,
            "scale_model_within": rf_within,
            "scale_blend": "mean_weights",
            "tau_fit": tau,
        }

    X, y = _design_matrix(U_all, Z_all, group_index_vector)
    if X is None or len(y) < 20:
        return {"score_type": "absolute"}

    # CQR: pinball quantile models at α/2 and 1-α/2 (HistGradientBoosting).
    a = float(alpha)
    a = min(max(a, 1e-4), 0.49)
    q_lo_lvl = a / 2.0
    q_hi_lvl = 1.0 - a / 2.0
    common = dict(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=max(5, hp["nodesize"]),
        random_state=hp["random_state"] + 23,
    )
    q_lo = HistGradientBoostingRegressor(loss="quantile", quantile=q_lo_lvl, **common)
    q_hi = HistGradientBoostingRegressor(loss="quantile", quantile=q_hi_lvl, **common)
    q_lo.fit(X, y)
    q_hi.fit(X, y)
    return {
        "score_type": "cqr",
        "q_lo_model": q_lo,
        "q_hi_model": q_hi,
        "q_lo_level": q_lo_lvl,
        "q_hi_level": q_hi_lvl,
    }


def _predict_one_scale(model, x_vector, u_vector) -> float:
    if model is None:
        return 1.0
    s = float(model.predict(_feat(x_vector, u_vector))[0])
    return float(np.clip(s, EPS, 1e12))


def predict_scale(aux, x_vector, u_vector, *, w_g: float = 1.0) -> float:
    """Predict residual scale; blend global/within σ with weight w_g when both exist."""
    if not aux or aux.get("score_type") != "studentized":
        return 1.0

    rf_g = aux.get("scale_model_global", aux.get("scale_model"))
    rf_w = aux.get("scale_model_within")
    s_g = _predict_one_scale(rf_g, x_vector, u_vector)
    if rf_w is None:
        return s_g
    w = float(np.clip(w_g, 0.0, 1.0))
    s_w = _predict_one_scale(rf_w, x_vector, u_vector)
    return float(np.clip(w * s_g + (1.0 - w) * s_w, EPS, 1e12))


def predict_quantiles(aux, x_vector, u_vector, delta: float = 0.0):
    """Return (q_lo, q_hi), optionally shifted by within-group delta."""
    if not aux or aux.get("score_type") != "cqr":
        return None, None
    f = _feat(x_vector, u_vector)
    lo = float(aux["q_lo_model"].predict(f)[0]) + float(delta)
    hi = float(aux["q_hi_model"].predict(f)[0]) + float(delta)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def cqr_within_delta(mu_global: float, mu_shrunk: float) -> float:
    """Additive shift so CQR bands track the same within-group recentering as μ."""
    return float(mu_shrunk - mu_global)


def conformity_score(y, mu, x_vector, u_vector, aux, *, mu_global=None, w_g: float = 1.0) -> float:
    """Scalar nonconformity score for one observation."""
    st = (aux or {}).get("score_type", "absolute")
    y = float(y)
    mu = float(mu)
    if st == "studentized":
        return float(abs(y - mu) / predict_scale(aux, x_vector, u_vector, w_g=w_g))
    if st == "cqr":
        delta = 0.0
        if mu_global is not None:
            delta = cqr_within_delta(float(mu_global), mu)
        lo, hi = predict_quantiles(aux, x_vector, u_vector, delta=delta)
        if lo is None:
            return float(abs(y - mu))
        return float(max(lo - y, y - hi))
    return float(abs(y - mu))


def interval_from_threshold(q, mu, x_vector, u_vector, aux, *, mu_global=None, w_g: float = 1.0):
    """Map conformal threshold q to a prediction interval."""
    if not np.isfinite(q):
        return (-np.inf, np.inf)
    st = (aux or {}).get("score_type", "absolute")
    mu = float(mu)
    if st == "cqr":
        delta = 0.0
        if mu_global is not None:
            delta = cqr_within_delta(float(mu_global), mu)
        lo, hi = predict_quantiles(aux, x_vector, u_vector, delta=delta)
        if lo is None:
            return (mu - q, mu + q)
        return (lo - q, hi + q)
    s = predict_scale(aux, x_vector, u_vector, w_g=w_g) if st == "studentized" else 1.0
    return (mu - q * s, mu + q * s)


def absolute_or_studentized_baseline_scores(y, mu, sigma=None):
    """Vector scores for baseline HCP (absolute or studentized)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if sigma is None:
        return np.abs(y - mu)
    sigma = np.asarray(sigma, dtype=float)
    return np.abs(y - mu) / np.clip(sigma, EPS, 1e12)


def cqr_baseline_scores(y, q_lo, q_hi):
    y = np.asarray(y, dtype=float)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    return np.maximum(q_lo - y, y - q_hi)
