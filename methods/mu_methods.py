"""
Mu-Estimation Method Objects (Random Forest + OLS)

Global RF / OLS model
---------------------
Fit once on all training groups (complement of S); produces
  mu_global(x, u) = model.predict([x, u])

Group-level shrinkage
---------------------
For each group j we observe the first tau observations Z_j[0..tau-1] as a
within-group training history.  The within-group mean is

  mu_bar_j = mean(Y_j[0], ..., Y_j[tau-1])     (first tau obs only)

The shrinkage prediction for observation (x, u) in group j is

  mu_hat(x, u | group j) = w_g * mu_global(x, u)  +  (1 - w_g) * mu_bar_j

where (manuscript Eq. (4) with λ = tau / (|Strain| + tau))

  w_g = |S_comp|^c / (|S_comp|^c + tau)

and
  tau      : number of within-group training obs (= floor(o/2) in GHCP)
  c        : exponent; default 1.0 matches Eq. (4)  (c=0.5 is the legacy sqrt rule)
  |S_comp| : number of groups used to fit the global model (= |Strain|)

When tau = 0 → pure global (no within-group history used).
When |S_comp| → ∞ → pure global (many groups → trust global more).

Performance note
----------------
All inner-loop score computations should use the batched `predict_global_batch`
and `predict_shrunk_batch` functions, which call model.predict once per group
rather than once per observation.  This gives 10-100x speedup for RF models.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stack_features(U_group, Z_group):
    """Build feature matrix [X | repeated U] for a group."""
    X_mat = np.asarray([z["X"] for z in Z_group], dtype=float)
    U_rep = np.repeat(
        np.asarray(U_group, dtype=float).reshape(1, -1),
        X_mat.shape[0], axis=0
    )
    return np.hstack([X_mat, U_rep])


def compute_group_mean(Z_group, tau):
    """
    Compute within-group mean Y using the first `tau` observations.

    Parameters
    ----------
    Z_group : list of {'X': ..., 'Y': ...}
    tau : int
        Number of leading observations to average.  If tau <= 0 or the group
        has fewer than tau observations, returns 0.0.

    Returns
    -------
    float
    """
    if tau <= 0 or len(Z_group) < tau:
        return 0.0
    return float(np.mean([Z_group[i]["Y"] for i in range(tau)]))


def global_weight(N_comp_groups, tau, c):
    """
    Global weight w_g = N_comp^c / (N_comp^c + tau).

    tau = 0  → w_g = 1 (pure global, no within-group history).
    N_comp = 0 → w_g = 0 (no global model; fall back to group mean).
    """
    if tau <= 0:
        return 1.0
    if N_comp_groups <= 0:
        return 0.0
    numer = float(N_comp_groups) ** c
    return numer / (numer + float(tau))


def estimate_re_variance_components(
    U_matrix,
    Z_list,
    group_index_vector,
    mu_method,
    global_model,
    *,
    ntree=50,
    nodesize=5,
    random_state=123,
    rho=0.5,
):
    """
    Classical one-way MoM estimators of σ² (within) and τ_B² (between).

    Uses residuals from the *already fitted* global predictor:

        e_{ji} = Y_{ji} - μ̂(X_{ji}, U_j)

    ``group_index_vector`` should be groups that were *not* used to train
    ``global_model`` (e.g. S_η / S_tilde in GHCP). Then μ̂ is out-of-sample for
    those groups and no LOGO is needed.

        σ̂² = pooled within-group variance of e
        τ̂_B² = max(0, s²_{between} - σ̂² / n̄_h)
    """
    groups = [int(j) for j in group_index_vector if len(Z_list[int(j)]) > 0]
    if len(groups) == 0:
        return 1.0, 0.0
    if global_model is None or mu_method is None:
        return 1.0, 0.0

    predict_global = mu_method["predict_global"]
    group_means = []
    ns = []
    within_ss = 0.0
    within_df = 0
    for g in groups:
        Zg = Z_list[g]
        n = len(Zg)
        if n <= 0:
            continue
        Uj = U_matrix[g, :]
        resid = np.empty(n, dtype=float)
        for i, z in enumerate(Zg):
            mu_hat = float(predict_global(
                model_global=global_model,
                x_vector=z["X"],
                u_vector=Uj,
            ))
            resid[i] = float(z["Y"]) - mu_hat
        group_means.append(float(np.mean(resid)))
        ns.append(n)
        if n > 1:
            within_ss += float(np.sum((resid - resid.mean()) ** 2))
            within_df += n - 1

    if within_df > 0:
        sigma2 = within_ss / within_df
    else:
        sigma2 = float(np.var(group_means, ddof=0)) if group_means else 1.0
    sigma2 = float(max(sigma2, 1e-12))

    if len(group_means) >= 2:
        s2_between = float(np.var(group_means, ddof=1))
        n_harm = len(ns) / sum(1.0 / n for n in ns)
        tau_B2 = float(max(s2_between - sigma2 / n_harm, 0.0))
    else:
        tau_B2 = 0.0
    return sigma2, tau_B2


def bayes_re_global_weight(sigma2, tau_B2, n_local):
    """
    Random-effects / empirical-Bayes weight on the global predictor:

        w_g = (σ²/n) / (τ_B² + σ²/n)

    with n = n_local (= τ = floor(o/2) in GHCP). n_local <= 0 → pure global.
    """
    n = int(n_local)
    if n <= 0:
        return 1.0
    sigma2 = float(max(sigma2, 1e-12))
    tau_B2 = float(max(tau_B2, 0.0))
    return float((sigma2 / n) / (tau_B2 + sigma2 / n))


def resolve_shrinkage_weight(mu_method, global_model, group_adjustment) -> float:
    """w_g for μ/σ blend; honors mu_method['w_g_override'] (e.g. 0 = pure local)."""
    if isinstance(group_adjustment, dict) and group_adjustment.get("w_g_override") is not None:
        return float(np.clip(group_adjustment["w_g_override"], 0.0, 1.0))
    if mu_method is not None:
        if mu_method.get("w_g_override") is not None:
            return float(np.clip(mu_method["w_g_override"], 0.0, 1.0))
        state = mu_method.get("_w_g_state")
        if isinstance(state, dict) and state.get("w_g_override") is not None:
            return float(np.clip(state["w_g_override"], 0.0, 1.0))
    if not isinstance(group_adjustment, dict):
        return 1.0
    tau_eff = int(group_adjustment.get("tau", 0))
    if tau_eff <= 0:
        return 1.0
    N_comp = 0
    if global_model is not None:
        N_comp = int(getattr(global_model, "_n_comp_groups", 0))
    c = float((mu_method or {}).get("c", 1.0))
    return float(global_weight(N_comp, tau_eff, c))


# ---------------------------------------------------------------------------
# Model-agnostic batch prediction helpers
# ---------------------------------------------------------------------------

def _predict_batch(model, X_mat, u_vector):
    """
    Batch prediction for n observations in one group.

    Parameters
    ----------
    model    : fitted sklearn model  (or None → returns zeros)
    X_mat    : ndarray (n, p_X)
    u_vector : ndarray (d,)  — shared group covariate

    Returns
    -------
    ndarray (n,)
    """
    if model is None:
        return np.zeros(len(X_mat))
    X_mat = np.asarray(X_mat, dtype=float)
    u = np.asarray(u_vector, dtype=float).ravel()
    U_rep = np.repeat(u.reshape(1, -1), len(X_mat), axis=0)
    return model.predict(np.hstack([X_mat, U_rep]))


def _predict_shrunk_batch(model, X_mat, u_vector, group_mean, N_comp_groups, tau, c):
    """
    Shrinkage prediction for a batch.

      mu_shrunk_i = w_g * mu_global(x_i, u) + (1-w_g) * group_mean

    Returns ndarray (n,).
    """
    mu_g = _predict_batch(model, X_mat, u_vector)
    if tau <= 0:
        return mu_g
    w_g = global_weight(N_comp_groups, tau, c)
    return w_g * mu_g + (1.0 - w_g) * float(group_mean)


# ---------------------------------------------------------------------------
# Common factory builder
# ---------------------------------------------------------------------------

def _make_method(fit_fn, tau, c):
    """
    Build the method dict given a fit function and shrinkage params.
    All predict functions are built here for both RF and OLS.
    """

    def predict_global(model_global, x_vector, u_vector):
        """Single-obs prediction (no shrinkage)."""
        if model_global is None:
            return 0.0
        x = np.asarray(x_vector, dtype=float).ravel()
        u = np.asarray(u_vector, dtype=float).ravel()
        return float(model_global.predict(
            np.concatenate([x, u]).reshape(1, -1))[0])

    def predict_global_batch(model_global, X_mat, u_vector):
        """Batched global prediction — one model.predict call for the whole group."""
        return _predict_batch(model_global, X_mat, u_vector)

    def predict_shrunk(model_global, x_vector, u_vector, group_mean, N_comp_groups):
        """Single-obs shrinkage prediction."""
        mu_g = predict_global(model_global, x_vector, u_vector)
        if tau <= 0:
            return mu_g
        w_g = global_weight(N_comp_groups, tau, c)
        return w_g * mu_g + (1.0 - w_g) * float(group_mean)

    def predict_shrunk_batch(model_global, X_mat, u_vector, group_mean, N_comp_groups):
        """Batched shrinkage prediction — use this in inner loops for speed."""
        return _predict_shrunk_batch(
            model_global, X_mat, u_vector, group_mean, N_comp_groups, tau, c)

    def _compute_group_mean(Z_group):
        return compute_group_mean(Z_group, tau)

    def fit_group_adjustment(model_global, u_group_vector,
                              Z_group_list, training_index_vector):
        """Within-group history: mean Y on training indices (first tau obs)."""
        idx = list(training_index_vector)
        if len(idx) == 0:
            return {"group_mean": 0.0, "tau": 0}
        group_mean = float(
            np.mean([Z_group_list[i]["Y"] for i in idx], dtype=float)
        )
        adj = {"group_mean": group_mean, "tau": len(idx)}
        if w_g_state["w_g_override"] is not None:
            adj["w_g_override"] = float(w_g_state["w_g_override"])
        return adj

    w_g_state = {"w_g_override": None}

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        """Shrink global prediction toward within-group mean (legacy donor-HCP API)."""
        if not isinstance(group_adjustment, dict):
            return predict_global(model_global, x_vector, u_group_vector)
        group_mean = float(group_adjustment.get("group_mean", 0.0))
        tau_eff = int(group_adjustment.get("tau", 0))
        if tau_eff <= 0:
            return predict_global(model_global, x_vector, u_group_vector)
        mu_g = predict_global(model_global, x_vector, u_group_vector)
        if group_adjustment.get("w_g_override") is not None:
            w_g = float(np.clip(group_adjustment["w_g_override"], 0.0, 1.0))
        elif w_g_state["w_g_override"] is not None:
            w_g = float(np.clip(w_g_state["w_g_override"], 0.0, 1.0))
        else:
            N_comp = 0
            if model_global is not None:
                N_comp = int(getattr(model_global, "_n_comp_groups", 0))
            w_g = global_weight(N_comp, tau_eff, c)
        return w_g * mu_g + (1.0 - w_g) * group_mean

    return {
        "fit_global": fit_fn,
        "predict_global": predict_global,
        "predict_global_batch": predict_global_batch,
        "predict_shrunk": predict_shrunk,
        "predict_shrunk_batch": predict_shrunk_batch,
        "compute_group_mean": _compute_group_mean,
        "global_weight": lambda N_comp: global_weight(N_comp, tau, c),
        "tau": tau,
        "c": c,
        "_w_g_state": w_g_state,
        # Legacy
        "fit_group_adjustment": fit_group_adjustment,
        "predict_group_mu": predict_group_mu,
    }


# ---------------------------------------------------------------------------
# Random Forest method factory
# ---------------------------------------------------------------------------

def create_mu_method_random_forest(ntree=50, mtry=None, nodesize=5,
                                    random_state=123, tau=0, c=1.0):
    """
    Create a mu-estimation method using a global Random Forest with optional
    within-group shrinkage.

    Parameters
    ----------
    ntree : int
    mtry  : int or None  (None → sqrt(p_total))
    nodesize : int
    random_state : int
    tau : int
        Number of within-group history observations to use for group mean
        shrinkage.  tau=0 disables shrinkage (pure global RF).
    c : float
        Exponent in w_g = N_comp^c / (N_comp^c + tau).
    """

    def fit_global(U_matrix, Z_list, group_index_vector):
        if len(group_index_vector) == 0:
            return None

        first_g = group_index_vector[0]
        p_X = len(Z_list[first_g][0]["X"])
        d_U = int(np.asarray(U_matrix).shape[1])
        local_mtry = mtry if mtry is not None else max(1, int(np.sqrt(p_X + d_U)))

        X_blocks, y_blocks = [], []
        for g in group_index_vector:
            Zg = Z_list[g]
            if len(Zg) == 0:
                continue
            X_blocks.append(_stack_features(U_matrix[g, :], Zg))
            y_blocks.append(np.array([z["Y"] for z in Zg], dtype=float))

        if not X_blocks:
            return None

        rf = RandomForestRegressor(
            n_estimators=ntree,
            max_features=local_mtry,
            min_samples_leaf=nodesize,
            random_state=random_state,
            n_jobs=1,          # single-threaded; avoids macOS fork overhead
        )
        rf.fit(np.vstack(X_blocks), np.concatenate(y_blocks))
        # record number of groups used to fit the global model
        try:
            rf._n_comp_groups = len(group_index_vector)
        except Exception:
            rf._n_comp_groups = 0
        return rf

    return _make_method(fit_global, tau, c)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def create_mu_method_random_forest_offset(ntree=50, mtry=None, nodesize=5,
                                           random_state=123, tau=0, c=1.0):
    """RF with optional within-group shrinkage (tau controls history depth)."""
    return create_mu_method_random_forest(
        ntree=ntree, mtry=mtry, nodesize=nodesize,
        random_state=random_state, tau=tau, c=c
    )


def create_mu_method_random_forest_global_only(ntree=50, mtry=None, nodesize=5,
                                                random_state=123):
    """RF, pure global (no within-group shrinkage; tau=0)."""
    return create_mu_method_random_forest(
        ntree=ntree, mtry=mtry, nodesize=nodesize,
        random_state=random_state, tau=0, c=0.5
    )


def create_mu_method_random_forest_residual_correction(
    ntree=50,
    mtry=None,
    nodesize=5,
    random_state=123,
    ridge_alpha: float = 1.0,
    local_adjustment_clip: float | None = None,
):
    """RF global mu with per-group residual correction on training indices."""
    base = create_mu_method_random_forest(
        ntree=ntree, mtry=mtry, nodesize=nodesize,
        random_state=random_state, tau=0, c=0.5,
    )
    predict_global = base["predict_global"]

    def fit_group_adjustment(
        model_global,
        u_group_vector,
        Z_group_list,
        training_index_vector,
    ):
        idx = list(training_index_vector)
        if len(idx) <= 0 or model_global is None:
            return {"mode": "none", "tau": 0}

        x_local = np.array([Z_group_list[i]["X"] for i in idx], dtype=float)
        y_local = np.array([Z_group_list[i]["Y"] for i in idx], dtype=float)
        mu_global_train = np.array([
            predict_global(model_global, Z_group_list[i]["X"], u_group_vector)
            for i in idx
        ], dtype=float)
        residuals = y_local - mu_global_train
        return _fit_within_group_correction(
            x_local, residuals, ridge_alpha=ridge_alpha
        )

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        mu_g = predict_global(model_global, x_vector, u_group_vector)
        if not isinstance(group_adjustment, dict):
            return mu_g

        mode = group_adjustment.get("mode", "none")
        if mode == "none" or int(group_adjustment.get("tau", 0)) <= 0:
            return mu_g

        g_x = _predict_within_group_correction(group_adjustment, x_vector)
        if not np.isfinite(g_x):
            return mu_g
        clip = float(local_adjustment_clip if local_adjustment_clip is not None else 0.5)
        g_x = float(np.clip(g_x, -clip, clip))
        return mu_g + g_x

    base["fit_group_adjustment"] = fit_group_adjustment
    base["predict_group_mu"] = predict_group_mu
    return base


def create_mu_method_random_forest_local_rf_offset(
    ntree=50,
    mtry=None,
    nodesize=5,
    random_state=123,
    c=1.0,
    local_ntree=None,
    local_nodesize=None,
    local_mtry="sqrt",
):
    """
    Global RF + within-group local RF (on X only), shrunk like the mean offset.

    Replaces the constant group-mean correction with a RandomForest fit on the
    group's training indices (first tau rows). Prediction is

        w_g * μ_global(x,u) + (1 - w_g) * μ_local_RF(x)

    with the same w_g = N_comp^c / (N_comp^c + tau) as the mean-shrinkage method.
    """
    base = create_mu_method_random_forest(
        ntree=ntree, mtry=mtry, nodesize=nodesize,
        random_state=random_state, tau=0, c=c,
    )
    predict_global = base["predict_global"]
    loc_ntree = int(ntree if local_ntree is None else local_ntree)
    loc_nodesize = int(nodesize if local_nodesize is None else local_nodesize)
    w_g_state = base.get("_w_g_state") or {"w_g_override": None}
    base["_w_g_state"] = w_g_state
    stdcp_inject_state = {
        "Z_group_list": None,
        "local_rf": None,
        "local_rf_scale": None,
    }
    base["_stdcp_inject_state"] = stdcp_inject_state

    def fit_group_adjustment(
        model_global,
        u_group_vector,
        Z_group_list,
        training_index_vector,
    ):
        idx = list(training_index_vector)
        if len(idx) <= 0:
            return {"mode": "none", "tau": 0}

        # Optional: inject a pre-fit Std-CP local RF for this exact group list.
        inj = stdcp_inject_state
        if (
            inj.get("Z_group_list") is Z_group_list
            and inj.get("local_rf") is not None
        ):
            adj = {
                "mode": "local_rf",
                "local_rf": inj["local_rf"],
                "tau": len(idx),
                "stdcp_injected": True,
            }
            if w_g_state["w_g_override"] is not None:
                adj["w_g_override"] = float(w_g_state["w_g_override"])
            return adj

        x_local = np.asarray([Z_group_list[i]["X"] for i in idx], dtype=float)
        y_local = np.asarray([Z_group_list[i]["Y"] for i in idx], dtype=float)
        if x_local.ndim == 1:
            x_local = x_local.reshape(-1, 1)

        # Tiny local samples: fall back to constant mean (RF cannot split).
        if len(y_local) < 2:
            adj = {
                "mode": "mean",
                "group_mean": float(y_local[0]) if len(y_local) else 0.0,
                "tau": len(idx),
            }
            if w_g_state["w_g_override"] is not None:
                adj["w_g_override"] = float(w_g_state["w_g_override"])
            return adj

        min_leaf = max(1, min(loc_nodesize, len(y_local)))
        max_features = local_mtry
        if max_features is None:
            max_features = "sqrt"
        rf_loc = RandomForestRegressor(
            n_estimators=loc_ntree,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            random_state=int(random_state) + 31,
            n_jobs=1,
        )
        rf_loc.fit(x_local, y_local)
        adj = {"mode": "local_rf", "local_rf": rf_loc, "tau": len(idx)}
        if w_g_state["w_g_override"] is not None:
            adj["w_g_override"] = float(w_g_state["w_g_override"])
        return adj

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        mu_g = predict_global(model_global, x_vector, u_group_vector)
        if not isinstance(group_adjustment, dict):
            return mu_g
        tau_eff = int(group_adjustment.get("tau", 0))
        mode = group_adjustment.get("mode", "none")
        if tau_eff <= 0 or mode in ("none", None):
            return mu_g

        if mode == "mean":
            mu_loc = float(group_adjustment.get("group_mean", 0.0))
        elif mode == "local_rf":
            rf_loc = group_adjustment.get("local_rf")
            if rf_loc is None:
                return mu_g
            x = np.asarray(x_vector, dtype=float).reshape(1, -1)
            mu_loc = float(rf_loc.predict(x)[0])
        else:
            return mu_g

        if group_adjustment.get("w_g_override") is not None:
            w_g = float(np.clip(group_adjustment["w_g_override"], 0.0, 1.0))
        elif w_g_state["w_g_override"] is not None:
            w_g = float(np.clip(w_g_state["w_g_override"], 0.0, 1.0))
        else:
            N_comp = int(getattr(model_global, "_n_comp_groups", 0)) if model_global is not None else 0
            w_g = global_weight(N_comp, tau_eff, c)
        return float(w_g * mu_g + (1.0 - w_g) * mu_loc)

    base["fit_group_adjustment"] = fit_group_adjustment
    base["predict_group_mu"] = predict_group_mu
    base["c"] = c
    base["within_group_mode"] = "local_rf"
    return base


def set_w_g_override(mu_method: dict, w_g: float | None) -> dict:
    """Force shrinkage weight (0 = pure local, 1 = pure global). Survives shallow copies."""
    out = dict(mu_method)
    state = out.get("_w_g_state")
    if state is None:
        state = mu_method.get("_w_g_state") or {"w_g_override": None}
        mu_method["_w_g_state"] = state
        out["_w_g_state"] = state
    if w_g is None:
        state["w_g_override"] = None
        out.pop("w_g_override", None)
        mu_method.pop("w_g_override", None)
    else:
        state["w_g_override"] = float(w_g)
        out["w_g_override"] = float(w_g)
        mu_method["w_g_override"] = float(w_g)
    return out


def attach_stdcp_local_predictor(mu_method: dict, *, Z_test_list, local_rf, local_rf_scale=None) -> dict:
    """Reuse Std-CP's fitted local RF when GHCP adjusts the test group."""
    out = dict(mu_method)
    state = out.get("_stdcp_inject_state")
    if state is None:
        state = mu_method.get("_stdcp_inject_state")
        if state is None:
            state = {"Z_group_list": None, "local_rf": None, "local_rf_scale": None}
            mu_method["_stdcp_inject_state"] = state
        out["_stdcp_inject_state"] = state
    state["Z_group_list"] = Z_test_list
    state["local_rf"] = local_rf
    state["local_rf_scale"] = local_rf_scale
    return out


# ---------------------------------------------------------------------------
# OLS methods (real-data bootstrap)
# ---------------------------------------------------------------------------

def create_mu_method_ols(tau=0, c=1.0):
    """OLS with optional within-group shrinkage."""

    def fit_global(U_matrix, Z_list, group_index_vector):
        if len(group_index_vector) == 0:
            return None
        X_blocks, y_blocks = [], []
        for g in group_index_vector:
            Zg = Z_list[g]
            if len(Zg) == 0:
                continue
            X_blocks.append(_stack_features(U_matrix[g, :], Zg))
            y_blocks.append(np.array([z["Y"] for z in Zg], dtype=float))
        if not X_blocks:
            return None
        ols = LinearRegression(fit_intercept=True)
        ols.fit(np.vstack(X_blocks), np.concatenate(y_blocks))
        # record number of groups used to fit the global model
        try:
            ols._n_comp_groups = len(group_index_vector)
        except Exception:
            ols._n_comp_groups = 0
        return ols

    return _make_method(fit_global, tau, c)


def create_mu_method_ols_offset(tau=0, c=1.0):
    """OLS with within-group shrinkage."""
    return create_mu_method_ols(tau=tau, c=c)


def create_mu_method_ols_global_only():
    """OLS, pure global (tau=0)."""
    return create_mu_method_ols(tau=0, c=0.5)


# ---------------------------------------------------------------------------
# Oracle-structure predictor: mu(x, u) = u_d^2
# ---------------------------------------------------------------------------

class _UdSquaredModel:
    """Stateless global mean: E[Y | X, U, B=0] ≈ u_d^2 under the joint-XY DGP.

    Feature matrix passed to predict is [X | U]; u_d is the last column.
    """

    def __init__(self, n_comp_groups: int = 0):
        self._n_comp_groups = int(n_comp_groups)

    def predict(self, XU):
        XU = np.asarray(XU, dtype=float)
        if XU.ndim == 1:
            XU = XU.reshape(1, -1)
        return XU[:, -1] ** 2


def create_mu_method_ud_squared(tau=0, c=0.5):
    """Global predictor mu(x,u) = u_d^2 with optional within-group mean shrinkage."""

    def fit_global(U_matrix, Z_list, group_index_vector):
        if len(group_index_vector) == 0:
            return None
        return _UdSquaredModel(n_comp_groups=len(group_index_vector))

    return _make_method(fit_global, tau, c)


def create_mu_method_ud_squared_offset(tau=0, c=0.5):
    """u_d^2 global mean with optional within-group shrinkage."""
    return create_mu_method_ud_squared(tau=tau, c=c)


def create_mu_method_ud_squared_global_only():
    """u_d^2, pure global (tau=0)."""
    return create_mu_method_ud_squared(tau=0, c=0.5)


# ---------------------------------------------------------------------------
# Near-Bayes global predictor for the joint-XY latent-intercept DGP
# ---------------------------------------------------------------------------

class _BayesJointXYModel:
    """E[Y | X, U] under the known DGP (B_j marginalized; mean-0).

    Marginal (X,Y)|U is Gaussian with mean mu(U)=U^2 and covariance
    Sigma(U) with Sigma_yy inflated by gamma^2. The conditional mean is

        E[Y|X,U] = u_d^2 + Sigma_{yx} Sigma_{xx}^{-1} (X - mu_X(U)).

    gamma cancels from the mean (only affects residual variance), so the
    Bayes predictor does not need gamma. Feature matrix is [X | U].
    """

    def __init__(self, rho: float = 0.5, n_comp_groups: int = 0):
        self.rho = float(rho)
        self._n_comp_groups = int(n_comp_groups)

    def predict(self, XU):
        XU = np.asarray(XU, dtype=float)
        if XU.ndim == 1:
            XU = XU.reshape(1, -1)
        n, p = XU.shape
        # p_X + d = p and p_X = d - 1  =>  d = (p + 1) // 2
        d = (p + 1) // 2
        p_X = d - 1
        X = XU[:, :p_X]
        U = XU[:, p_X:]
        rho = self.rho
        out = np.empty(n, dtype=float)
        eye = np.eye(d)
        for i in range(n):
            u = U[i]
            mu = u ** 2
            S = (1.0 - rho) * np.diag(u) + rho * np.ones((d, d))
            S = 0.5 * (S + S.T) + 1e-8 * eye
            Sxx = S[:p_X, :p_X]
            Syx = S[p_X, :p_X]  # row: cov(Y, X)
            out[i] = mu[-1] + float(Syx @ np.linalg.solve(Sxx, X[i] - mu[:p_X]))
        return out


def create_mu_method_bayes_joint_xy(rho: float = 0.5, tau: float = 0, c: float = 0.5):
    """Near-Bayes global mu = E[Y|X,U] for the joint-XY DGP (known rho)."""

    def fit_global(U_matrix, Z_list, group_index_vector):
        if len(group_index_vector) == 0:
            return None
        return _BayesJointXYModel(rho=rho, n_comp_groups=len(group_index_vector))

    return _make_method(fit_global, tau, c)


def create_mu_method_bayes_joint_xy_offset(rho: float = 0.5, tau: float = 0, c: float = 0.5):
    return create_mu_method_bayes_joint_xy(rho=rho, tau=tau, c=c)


def create_mu_method_bayes_joint_xy_global_only(rho: float = 0.5):
    return create_mu_method_bayes_joint_xy(rho=rho, tau=0, c=0.5)


def create_mu_method_oracle_yxub(rho: float = 0.5):
    """
    Full oracle center E[Y | X, U, B] = m(u,x) + B for the joint-XY DGP.

    No within-group merging: the latent intercept is treated as known.
    Call ``register_oracle_group_B`` (or set ``_b_by_u``) before predicting so
    each group's U maps to its B.
    """
    b_by_u: dict[tuple, float] = {}
    base_model = _BayesJointXYModel(rho=rho, n_comp_groups=0)

    def fit_global(U_matrix, Z_list, group_index_vector):
        # Stateless closed form; only record |Strain| for API compatibility.
        n_comp = len(group_index_vector)
        return _BayesJointXYModel(rho=rho, n_comp_groups=n_comp)

    def predict_global(model_global, x_vector, u_vector):
        u = np.asarray(u_vector, dtype=float).ravel()
        key = tuple(u.tolist())
        if key not in b_by_u:
            raise KeyError(
                "oracle_yxub: B not registered for this U; "
                "call register_oracle_group_B before GHCP."
            )
        mu_xu = float(_predict_batch(model_global or base_model, np.asarray(x_vector, dtype=float).reshape(1, -1), u)[0])
        return mu_xu + float(b_by_u[key])

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return predict_global(model_global, x_vector, u_group_vector)

    return {
        "fit_global": fit_global,
        "predict_global": predict_global,
        "fit_group_adjustment": fit_group_adjustment,
        "predict_group_mu": predict_group_mu,
        "c": 1.0,
        "tau": 0,
        "merger": "none",
        "oracle_yxub": True,
        "_b_by_u": b_by_u,
    }


def register_oracle_group_B(mu_method, U_matrix, Z_list, group_index_vector=None):
    """Map each group's U -> B for ``create_mu_method_oracle_yxub``."""
    reg = mu_method.get("_b_by_u")
    if reg is None:
        raise ValueError("mu_method has no _b_by_u registry")
    if group_index_vector is None:
        group_index_vector = range(len(Z_list))
    for j in group_index_vector:
        Zj = Z_list[int(j)]
        if not Zj:
            continue
        if "B" not in Zj[0]:
            raise KeyError("observations must carry key 'B' for oracle_yxub")
        key = tuple(np.asarray(U_matrix[int(j)], dtype=float).ravel().tolist())
        reg[key] = float(Zj[0]["B"])
    return mu_method


def create_mu_method_ols_residual_correction(
    ridge_alpha: float = 1.0,
    local_adjustment_clip: float | None = None,
):
    """
    OLS global mu with per-group residual correction on training indices.

    Fit g_j on R = Y - mu_global(X) with penalized constant or linear correction;
    predict mu_tilde(x) = mu_global(x) + clip(g_j(x)).
    """
    base = create_mu_method_ols(tau=0, c=0.5)
    predict_global = base["predict_global"]

    def fit_group_adjustment(
        model_global,
        u_group_vector,
        Z_group_list,
        training_index_vector,
    ):
        idx = list(training_index_vector)
        if len(idx) <= 0 or model_global is None:
            return {"mode": "none", "tau": 0}

        x_local = np.array([Z_group_list[i]["X"] for i in idx], dtype=float)
        y_local = np.array([Z_group_list[i]["Y"] for i in idx], dtype=float)
        mu_global_train = np.array([
            predict_global(model_global, Z_group_list[i]["X"], u_group_vector)
            for i in idx
        ], dtype=float)
        residuals = y_local - mu_global_train
        return _fit_within_group_correction(
            x_local, residuals, ridge_alpha=ridge_alpha
        )

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        mu_g = predict_global(model_global, x_vector, u_group_vector)
        if not isinstance(group_adjustment, dict):
            return mu_g

        mode = group_adjustment.get("mode", "none")
        if mode == "none" or int(group_adjustment.get("tau", 0)) <= 0:
            return mu_g

        g_x = _predict_within_group_correction(group_adjustment, x_vector)
        if not np.isfinite(g_x):
            return mu_g
        clip = float(local_adjustment_clip if local_adjustment_clip is not None else 0.5)
        g_x = float(np.clip(g_x, -clip, clip))
        return mu_g + g_x

    base["fit_group_adjustment"] = fit_group_adjustment
    base["predict_group_mu"] = predict_group_mu
    return base


# Ding et al. (2021) NeurIPS folktables ACSIncome GBM hyperparameters (sklearn defaults
# except n_estimators=5, max_depth=5). Folktables ACSIncome is binary (>50k); here we
# reuse only the tree budget and fit XGBRegressor (squared-error) for point prediction.
DING_GBM_N_ESTIMATORS = 5
DING_GBM_MAX_DEPTH = 5
DING_GBM_RANDOM_STATE = 0
MAX_LOCAL_LOG_ADJUSTMENT = 0.5


def train_ding_2021_xgb_regressor(X, y, *, u_dim: int = 1, random_state: int = DING_GBM_RANDOM_STATE):
    """Pre-fit California XGBoost with Ding et al. (2021) GBM tree budget."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost: pip install xgboost") from exc

    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    if u_dim > 0:
        X_mat = np.hstack([X_mat, np.zeros((len(X_mat), int(u_dim)), dtype=float)])

    model = XGBRegressor(
        n_estimators=DING_GBM_N_ESTIMATORS,
        max_depth=DING_GBM_MAX_DEPTH,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_mat, y_vec)
    return model


MAX_LOCAL_LOG_ADJUSTMENT = 0.5


def _fit_g_constant_on_residuals(residuals: np.ndarray, kappa_j: float) -> float:
    """H = {constants}: g(x) = delta with squared penalty kappa_j * delta^2."""
    residuals = np.asarray(residuals, dtype=float).ravel()
    m = len(residuals)
    if m <= 0:
        return 0.0
    return float(np.sum(residuals) / (m + float(kappa_j)))


def _fit_g_linear_on_residuals(
    x_local: np.ndarray,
    residuals: np.ndarray,
    *,
    kappa_j: float | None = None,
):
    """
    H = {affine in x}: fit g with R ~ [1, X].

    Unpenalized OLS when m > p + 1; otherwise None (caller uses constant class).
    """
    x_local = np.asarray(x_local, dtype=float)
    residuals = np.asarray(residuals, dtype=float).ravel()
    m, p = x_local.shape
    if m <= p + 1:
        return None
    x_aug = np.column_stack([np.ones(m), x_local])
    try:
        coef, *_ = np.linalg.lstsq(x_aug, residuals, rcond=None)
        return coef
    except Exception:
        return None


def _predict_g_linear(coef, x_vector) -> float:
    x_aug = np.concatenate([[1.0], np.asarray(x_vector, dtype=float).ravel()])
    return float(x_aug @ coef)


def _fit_within_group_correction(
    x_local: np.ndarray,
    residuals: np.ndarray,
    *,
    ridge_alpha: float,
) -> dict:
    """
    Fit g_j in argmin_g (1/m) sum l(R - g(X)) + kappa_j pen(g) on training residuals.

    Returns an adjustment dict for predict_group_mu (mu_tilde = mu_glob + g_j(x)).
    """
    m = len(residuals)
    if m <= 0:
        return {"mode": "none", "tau": 0}

    p = x_local.shape[1]
    kappa_j = float(ridge_alpha)

    coef = _fit_g_linear_on_residuals(x_local, residuals, kappa_j=kappa_j)
    if coef is not None:
        return {"mode": "linear", "coef": coef, "tau": m, "kappa_j": kappa_j}

    delta = _fit_g_constant_on_residuals(residuals, kappa_j)
    return {"mode": "constant", "delta": delta, "tau": m, "kappa_j": kappa_j}


def _predict_within_group_correction(adjustment: dict, x_vector) -> float:
    mode = adjustment.get("mode", "none")
    if mode == "linear":
        return _predict_g_linear(adjustment["coef"], x_vector)
    if mode == "constant":
        return float(adjustment.get("delta", 0.0))
    return 0.0


def create_mu_method_pretrained_global(model, tau: int = 0, c: float = 0.5):
    """Use a fixed pre-trained global model (no re-fit on calibration data)."""

    def fit_global(U_matrix, Z_list, group_index_vector):
        if model is None:
            return None
        try:
            model._n_comp_groups = len(group_index_vector)
        except Exception:
            pass
        return model

    return _make_method(fit_global, tau=tau, c=c)


def create_mu_method_pretrained_offset(
    model,
    tau: int = 0,
    c: float = 0.5,
    ridge_alpha: float = 1.0,
    local_adjustment_clip: float = MAX_LOCAL_LOG_ADJUSTMENT,
):
    """
    Pre-trained global mu with per-group residual correction.

    For training indices i in [m_j], R_{j,i} = Y_{j,i} - mu_glob(U_j, X_{j,i}).
    Fit g_j by penalized regression of R on X (linear H when m_j > p+1, else
    constant intercept correction). Predict mu_tilde(U_j, x) = mu_glob(U_j, x) + g_j(x).
    """

    def fit_global(U_matrix, Z_list, group_index_vector):
        if model is None:
            return None
        try:
            model._n_comp_groups = len(group_index_vector)
        except Exception:
            pass
        return model

    base = _make_method(fit_global, tau=tau, c=c)
    predict_global = base["predict_global"]

    def fit_group_adjustment(
        model_global,
        u_group_vector,
        Z_group_list,
        training_index_vector,
    ):
        idx = list(training_index_vector)
        if len(idx) <= 0 or model_global is None:
            return {"mode": "none", "tau": 0}

        x_local = np.array([Z_group_list[i]["X"] for i in idx], dtype=float)
        y_local = np.array([Z_group_list[i]["Y"] for i in idx], dtype=float)
        mu_global_train = np.array([
            predict_global(model_global, Z_group_list[i]["X"], u_group_vector)
            for i in idx
        ], dtype=float)
        residuals = y_local - mu_global_train
        return _fit_within_group_correction(
            x_local, residuals, ridge_alpha=ridge_alpha
        )

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        mu_g = predict_global(model_global, x_vector, u_group_vector)
        if not isinstance(group_adjustment, dict):
            return mu_g

        mode = group_adjustment.get("mode", "none")
        if mode == "none" or int(group_adjustment.get("tau", 0)) <= 0:
            return mu_g

        g_x = _predict_within_group_correction(group_adjustment, x_vector)
        if not np.isfinite(g_x):
            return mu_g
        clip = float(local_adjustment_clip)
        g_x = float(np.clip(g_x, -clip, clip))
        return mu_g + g_x

    base["fit_group_adjustment"] = fit_group_adjustment
    base["predict_group_mu"] = predict_group_mu
    return base
