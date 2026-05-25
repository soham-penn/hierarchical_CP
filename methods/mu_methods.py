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

where

  w_g = |S_comp|^c / (|S_comp|^c + tau)

and
  tau      : number of within-group training obs (also controls shrinkage strength)
  c        : exponent; default 0.5 (sqrt rule)
  |S_comp| : number of groups used to fit the global model

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
        return {"group_mean": group_mean, "tau": len(idx)}

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        """Shrink global prediction toward within-group mean (legacy donor-HCP API)."""
        if not isinstance(group_adjustment, dict):
            return predict_global(model_global, x_vector, u_group_vector)
        group_mean = float(group_adjustment.get("group_mean", 0.0))
        tau_eff = int(group_adjustment.get("tau", 0))
        if tau_eff <= 0:
            return predict_global(model_global, x_vector, u_group_vector)
        N_comp = 0
        if model_global is not None:
            N_comp = int(getattr(model_global, "_n_comp_groups", 0))
        mu_g = predict_global(model_global, x_vector, u_group_vector)
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
        # Legacy
        "fit_group_adjustment": fit_group_adjustment,
        "predict_group_mu": predict_group_mu,
    }


# ---------------------------------------------------------------------------
# Random Forest method factory
# ---------------------------------------------------------------------------

def create_mu_method_random_forest(ntree=50, mtry=None, nodesize=5,
                                    random_state=123, tau=0, c=0.5):
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
                                           random_state=123, tau=0, c=0.5):
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


# ---------------------------------------------------------------------------
# OLS methods (real-data bootstrap)
# ---------------------------------------------------------------------------

def create_mu_method_ols(tau=0, c=0.5):
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


def create_mu_method_ols_offset(tau=0, c=0.5):
    """OLS with within-group shrinkage."""
    return create_mu_method_ols(tau=tau, c=c)


def create_mu_method_ols_global_only():
    """OLS, pure global (tau=0)."""
    return create_mu_method_ols(tau=0, c=0.5)
