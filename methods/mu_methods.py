"""
Mu-Estimation Method Objects (Random Forest + Offset + OLS)

This module defines methods for estimating the conditional mean function μ(X, U)
using Random Forest models or OLS with group-specific offsets.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_mu_method_random_forest_offset(ntree=50, mtry=None, nodesize=5, random_state=123):
    """
    Create a mu-estimation method using Random Forest with group-specific offsets.

    Fits a global RF model on (X, U) then estimates group offsets as mean residuals
    on specified within-group training indices.

    Parameters
    ----------
    ntree : int
        Number of trees.
    mtry : int or None
        Number of features considered at each split (if None uses sqrt(p_total)).
    nodesize : int
        Minimum samples per leaf.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Method object with fit/predict functions.
    """

    def _stack_group_features(U_group, Z_group):
        """
        Build feature matrix for one group: [X | repeated U].
        """
        X_mat = np.asarray([z["X"] for z in Z_group], dtype=float)
        U_rep = np.repeat(np.asarray(U_group, dtype=float).reshape(1, -1), X_mat.shape[0], axis=0)
        return np.hstack([X_mat, U_rep])

    def fit_global(U_matrix, Z_list, group_index_vector):
        """
        Fit global Random Forest on pooled data from selected groups (0-indexed).
        """
        if len(group_index_vector) == 0:
            return None

        # infer dimensions
        first_g = group_index_vector[0]
        p_X = len(Z_list[first_g][0]["X"])
        d_U = int(np.asarray(U_matrix).shape[1])
        p_total = p_X + d_U

        local_mtry = mtry
        if local_mtry is None:
            local_mtry = max(1, int(np.sqrt(p_total)))

        X_blocks = []
        y_blocks = []

        for g in group_index_vector:
            U_g = U_matrix[g, :]
            Z_g = Z_list[g]
            if len(Z_g) == 0:
                continue

            Xg = _stack_group_features(U_g, Z_g)
            yg = np.asarray([z["Y"] for z in Z_g], dtype=float)

            X_blocks.append(Xg)
            y_blocks.append(yg)

        if len(X_blocks) == 0:
            return None

        X_train = np.vstack(X_blocks)
        y_train = np.concatenate(y_blocks)

        rf = RandomForestRegressor(
            n_estimators=ntree,
            max_features=local_mtry,
            min_samples_leaf=nodesize,
            random_state=random_state,
            n_jobs=-1,  # parallelize within RF (optional; remove if you prefer)
        )
        rf.fit(X_train, y_train)
        return rf

    def predict_global(model_global, x_vector, u_vector):
        """
        Predict using the global model on a single (x,u).
        """
        if model_global is None:
            return 0.0

        x = np.asarray(x_vector, dtype=float).ravel()
        u = np.asarray(u_vector, dtype=float).ravel()
        feats = np.concatenate([x, u]).reshape(1, -1)
        return float(model_global.predict(feats)[0])

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        """
        Fit group-specific offset = mean(Y - mu_global) over training indices (0-indexed).
        """
        if len(training_index_vector) == 0:
            return 0.0

        idx = list(training_index_vector)
        y_train = np.asarray([Z_group_list[i]["Y"] for i in idx], dtype=float)

        # vectorized prediction on the selected indices
        X_sel = np.asarray([Z_group_list[i]["X"] for i in idx], dtype=float)
        u = np.asarray(u_group_vector, dtype=float).ravel()
        U_rep = np.repeat(u.reshape(1, -1), X_sel.shape[0], axis=0)
        feats = np.hstack([X_sel, U_rep])

        mu_global = 0.0 if model_global is None else model_global.predict(feats)
        mu_global = np.asarray(mu_global, dtype=float).ravel()

        return float(np.mean(y_train - mu_global))

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        """
        Predict using global model + group adjustment.
        """
        return predict_global(model_global, x_vector, u_group_vector) + float(group_adjustment)

    return {
        "fit_global": fit_global,
        "predict_global": predict_global,
        "fit_group_adjustment": fit_group_adjustment,
        "predict_group_mu": predict_group_mu,
    }


def create_mu_method_random_forest_global_only(ntree=50, mtry=None, nodesize=5, random_state=123):
    """
    Create a mu-estimation method using only a global Random Forest (no group offsets).
    """
    base = create_mu_method_random_forest_offset(
        ntree=ntree, mtry=mtry, nodesize=nodesize, random_state=random_state
    )

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return base["predict_global"](model_global, x_vector, u_group_vector)

    base["fit_group_adjustment"] = fit_group_adjustment
    base["predict_group_mu"] = predict_group_mu
    return base


def create_mu_method_ols_offset():
    """
    Create a mu-estimation method using OLS with group-specific offsets.
    Uses NumPy arrays (no pandas).
    """

    def _stack_group_features(U_group, Z_group):
        X_mat = np.asarray([z["X"] for z in Z_group], dtype=float)
        U_rep = np.repeat(np.asarray(U_group, dtype=float).reshape(1, -1), X_mat.shape[0], axis=0)
        return np.hstack([X_mat, U_rep])

    def fit_global(U_matrix, Z_list, group_index_vector):
        """
        Fit a global OLS model using data from specified groups (0-indexed).
        """
        if len(group_index_vector) == 0:
            return None

        X_blocks = []
        y_blocks = []

        for g in group_index_vector:
            U_g = U_matrix[g, :]
            Z_g = Z_list[g]
            if len(Z_g) == 0:
                continue

            Xg = _stack_group_features(U_g, Z_g)
            yg = np.asarray([z["Y"] for z in Z_g], dtype=float)

            X_blocks.append(Xg)
            y_blocks.append(yg)

        if len(X_blocks) == 0:
            return None

        X_train = np.vstack(X_blocks)
        y_train = np.concatenate(y_blocks)

        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_train, y_train)
        return ols

    def predict_global(model_global, x_vector, u_vector):
        """
        Predict using the global OLS model.
        """
        if model_global is None:
            return 0.0

        x = np.asarray(x_vector, dtype=float).ravel()
        u = np.asarray(u_vector, dtype=float).ravel()
        feats = np.concatenate([x, u]).reshape(1, -1)
        return float(model_global.predict(feats)[0])

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        """
        Fit group-specific adjustment = mean(Y - mu_global) over training indices (0-indexed).
        """
        if model_global is None or len(training_index_vector) == 0:
            return 0.0

        idx = list(training_index_vector)
        y_train = np.asarray([Z_group_list[i]["Y"] for i in idx], dtype=float)

        X_sel = np.asarray([Z_group_list[i]["X"] for i in idx], dtype=float)
        u = np.asarray(u_group_vector, dtype=float).ravel()
        U_rep = np.repeat(u.reshape(1, -1), X_sel.shape[0], axis=0)
        feats = np.hstack([X_sel, U_rep])

        mu_global = np.asarray(model_global.predict(feats), dtype=float).ravel()
        return float(np.mean(y_train - mu_global))  # (fix #7) ensure Python float

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return predict_global(model_global, x_vector, u_group_vector) + float(group_adjustment)

    return {
        "fit_global": fit_global,
        "predict_global": predict_global,
        "fit_group_adjustment": fit_group_adjustment,
        "predict_group_mu": predict_group_mu,
    }


def create_mu_method_ols_global_only():
    """
    Create a mu-estimation method using OLS without group-specific adjustments.
    """
    base = create_mu_method_ols_offset()

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return base["predict_global"](model_global, x_vector, u_group_vector)

    base["fit_group_adjustment"] = fit_group_adjustment
    base["predict_group_mu"] = predict_group_mu
    return base