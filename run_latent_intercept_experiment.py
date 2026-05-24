#!/usr/bin/env python3
"""Paired fixed-target simulation with tau_B in {0, 6}."""

from __future__ import annotations

from multiprocessing import Pool
import multiprocessing as _mp

# Prefer 'fork' start method on macOS to avoid spawn-import issues
try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_repeated_subsampling_interval_radius,
    compute_subsampling_once_interval_radius,
)
from methods.donor_hcp import (
    _ALPHA_SPLIT_TIEBREAK_SEED,
    _select_s_tilde_with_tie_randomization,
    get_donor_style_train_cal_split,
)
from methods.mu_methods import create_mu_method_ols_global_only, create_mu_method_ols_offset
from methods.sample_hcp import compute_sample_hcp_randomized_interval
from scores import absolute_residual_score, weighted_quantile
from DGP.code.experiments import _compute_std_cp_interval


K_HISTORICAL = 20
N_HISTORICAL = 21
N_TARGET = 36
TARGET_INDEX = 35

DIMENSION = 5
RHO = 0.5
U_MIN = 1.0
U_MAX = 5.0

O_VALUES = [0, 5, 10, 15, 20, 25, 30, 35]
ALPHA_VALUES = [0.10, 0.05]
TAU_B_VALUES = [0.0, 5.0]

N_EXPERIMENTS = 50
N_TARGET_GROUPS_PER_EXPERIMENT = 100
N_REPEATED_SUBSAMPLING = 50
ALPHA_SELECTION = 0.5
N_WORKERS = 6
BASE_SEED = 12345

# If True, sample group sizes N_i = 1 + Poisson(POISSON_LAMBDA)
POISSON_MODE = False
POISSON_LAMBDA = 20

N_RANDOMIZATION_DATASETS = 50
N_RANDOMIZATION_RERUNS = 100

SCRIPT_DIR = Path(__file__).resolve().parent
# Write outputs to the top-level NEW_RESULTS and NEW_PLOTS directories
OUT_DIR = SCRIPT_DIR / "NEW_RESULTS"
PLOT_DIR = SCRIPT_DIR / "NEW_PLOTS"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


O_COLORS = {
    0: "#0072B2",
    5: "#00A1D5",
    10: "#00B894",
    15: "#F39C12",
    20: "#E74C3C",
    25: "#8E44AD",
    30: "#C0392B",
    35: "#2C3E50",
}

BASELINE_GRAYS = {
    "HCP": "#333333",
    "Pooling": "#666666",
    "Subsampling": "#999999",
    "Repeated": "#bbbbbb",
}

WITHIN_LOCAL_MODE = "mean"  # 'mean' or 'ols'


def _mode_tag() -> str:
    return "poissonNmean21" if POISSON_MODE else "fixedN21"


def _alpha_tag(alpha: float) -> str:
    return f"alpha{int(round(alpha * 100)):02d}"


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(round(width * done / total))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _mu_function(u_vec: np.ndarray) -> np.ndarray:
    return np.asarray(u_vec, dtype=float).ravel() ** 2


def _sigma_function(u_vec: np.ndarray, rho: float) -> np.ndarray:
    u = np.asarray(u_vec, dtype=float).ravel()
    d = len(u)
    sigma = (1.0 - rho) * np.diag(u) + rho * np.ones((d, d), dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-8 * np.eye(d)
    return sigma


def generate_group_data(u_vec: np.ndarray, n_obs: int, b_j: float, rng: np.random.Generator) -> List[dict]:
    mu = _mu_function(u_vec)
    mu[-1] += b_j
    sigma = _sigma_function(u_vec, RHO)
    z = rng.multivariate_normal(mean=mu, cov=sigma, size=n_obs)
    return [{"X": z[i, :-1].astype(float), "Y": float(z[i, -1])} for i in range(n_obs)]


def generate_historical_groups(tau_b: float, rng: np.random.Generator, use_poisson: bool = False) -> Tuple[np.ndarray, List[List[dict]]]:
    u_hist = rng.uniform(low=U_MIN, high=U_MAX, size=(K_HISTORICAL, DIMENSION))
    b_hist = rng.normal(loc=0.0, scale=tau_b, size=K_HISTORICAL)
    if not use_poisson:
        z_hist = [generate_group_data(u_hist[j], N_HISTORICAL, b_hist[j], rng) for j in range(K_HISTORICAL)]
    else:
        n_vec = 1 + rng.poisson(lam=POISSON_LAMBDA, size=K_HISTORICAL)
        z_hist = [generate_group_data(u_hist[j], int(n_vec[j]), b_hist[j], rng) for j in range(K_HISTORICAL)]
    return u_hist, z_hist


def generate_target_groups(tau_b: float, n_groups: int, rng: np.random.Generator, use_poisson: bool = False) -> Tuple[np.ndarray, List[List[dict]]]:
    u_tgt = rng.uniform(low=U_MIN, high=U_MAX, size=(n_groups, DIMENSION))
    b_tgt = rng.normal(loc=0.0, scale=tau_b, size=n_groups)
    if not use_poisson:
        z_tgt = [generate_group_data(u_tgt[t], N_TARGET, b_tgt[t], rng) for t in range(n_groups)]
    else:
        z_tgt = []
        min_required_n = max(TARGET_INDEX + 1, max(O_VALUES) + 1)
        for t in range(n_groups):
            n_test = 1 + rng.poisson(lam=POISSON_LAMBDA)
            # Ensure enough points for all requested o values and target index.
            if n_test < min_required_n:
                n_test = min_required_n
            z_tgt.append(generate_group_data(u_tgt[t], int(n_test), b_tgt[t], rng))
    return u_tgt, z_tgt


def _covered(interval: Tuple[float, float], y: float) -> int:
    return int(interval[0] <= y <= interval[1])


def _width(interval: Tuple[float, float]) -> float:
    lo, hi = interval
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.nan


def _is_infinite(interval: Tuple[float, float]) -> int:
    lo, hi = interval
    return int(not (np.isfinite(lo) and np.isfinite(hi)))


def _baseline_interval(mu_hat: float, radius: float) -> Tuple[float, float]:
    if np.isfinite(radius):
        return (mu_hat - radius, mu_hat + radius)
    return (-np.inf, np.inf)


def _m_j(o_observed: int, n_prime: int, use_within: bool) -> int:
    if not use_within:
        return 0
    return int(min(np.floor(o_observed / 2), np.floor(n_prime / 2)))


def _local_mean_y(z_group: List[dict], m_j: int) -> float | None:
    if m_j <= 0:
        return None
    return float(np.mean([z_group[i]["Y"] for i in range(m_j)]))


def _local_ols_model(z_group: List[dict], m_j: int):
    if m_j < 2:
        return None
    x_local = np.array([z_group[i]["X"] for i in range(m_j)], dtype=float)
    y_local = np.array([z_group[i]["Y"] for i in range(m_j)], dtype=float)
    if m_j <= x_local.shape[1] + 1:
        return None
    x_aug = np.column_stack([np.ones(m_j), x_local])
    try:
        beta, *_ = np.linalg.lstsq(x_aug, y_local, rcond=None)
        return beta
    except Exception:
        return None


def _predict_local_ols(beta, x_vec) -> float | None:
    if beta is None:
        return None
    x_aug = np.concatenate([[1.0], np.asarray(x_vec, dtype=float).ravel()])
    return float(x_aug @ beta)


def _merge_mu(mu_global: float, mu_local: float | None, m_j: int, s_size: int, is_valid: bool) -> float:
    if m_j <= 0 or mu_local is None or not is_valid:
        return float(mu_global)
    lam = float(m_j) / float(s_size + m_j)
    return float((1.0 - lam) * mu_global + lam * mu_local)


def _compute_donor_hcp_interval_with_local(
    U_calibration: np.ndarray,
    Z_calibration: List[List[dict]],
    U_test: np.ndarray,
    Z_test: List[dict],
    o_observed: int,
    alpha: float,
    alpha_selection: float,
    mu_method,
    local_mode: str,  # 'none', 'mean', 'ols'
    test_index_target: int,
    random_seed: int | None = None,
) -> Dict:
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)], dtype=int)
    test_idx = K

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    if len(Z_test) < max(o_observed + 1, test_index_target + 1):
        raise ValueError("Z_test must have at least max(o+1, target+1) observations.")

    if K == 0:
        return {"interval": (-np.inf, np.inf), "mu_hat": 0.0, "number_selected_groups": 0, "donor_group_index": None}

    use_within = local_mode != "none"

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if len(S_tilde) == 0:
        global_model = mu_method["fit_global"](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K)),
        )
        m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_within=use_within)
        if local_mode == "mean":
            local_est_test = _local_mean_y(Z_test, m_test)
        elif local_mode == "ols":
            local_est_test = _local_ols_model(Z_test, m_test)
        else:
            local_est_test = None
        s_size = 1
        scores = []
        if o_observed >= (m_test + 1):
            for i in range(m_test, o_observed):
                z = Z_test[i]
                mu_g = mu_method["predict_global"](model_global=global_model, x_vector=z["X"], u_vector=U_test[0, :])
                if local_mode == "mean":
                    mu_loc = local_est_test
                    is_valid = mu_loc is not None
                elif local_mode == "ols":
                    mu_loc = _predict_local_ols(local_est_test, z["X"]) if local_est_test is not None else None
                    is_valid = mu_loc is not None
                else:
                    mu_loc = None
                    is_valid = False
                mu_tilde = _merge_mu(mu_g, mu_loc, m_test, s_size, is_valid)
                scores.append(abs(float(z["Y"]) - float(mu_tilde)))
            scores.append(np.inf)
            weights = np.ones(len(scores), dtype=float) / len(scores)
            q = weighted_quantile(scores, weights, alpha)
        else:
            q = np.inf
        x_target = Z_test[test_index_target]["X"]
        mu_g_target = mu_method["predict_global"](model_global=global_model, x_vector=x_target, u_vector=U_test[0, :])
        if local_mode == "mean":
            mu_loc_target = local_est_test
            is_valid_target = mu_loc_target is not None
        elif local_mode == "ols":
            mu_loc_target = _predict_local_ols(local_est_test, x_target) if local_est_test is not None else None
            is_valid_target = mu_loc_target is not None
        else:
            mu_loc_target = None
            is_valid_target = False
        mu_center = _merge_mu(mu_g_target, mu_loc_target, m_test, s_size, is_valid_target)
        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)
        return {"interval": interval, "mu_hat": float(mu_center), "number_selected_groups": 1, "donor_group_index": None}

    donor_rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = int(donor_rng.choice(S_tilde))
    N_donor = int(N[donor])

    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S = np.sort(np.concatenate([S_cal, [test_idx]]))
    s_size = len(S)

    S_comp = np.setdiff1d(list(range(K + 1)), S)
    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method["fit_global"](U_matrix=U_all, Z_list=Z_all, group_index_vector=list(S_comp))

    scores = []
    weights = []

    for j in S_cal:
        N_j_prime = int(N[j])
        m_j = _m_j(o_observed=o_observed, n_prime=N_j_prime, use_within=use_within)
        if N_j_prime <= m_j:
            continue
        if local_mode == "mean":
            local_est_j = _local_mean_y(Z_calibration[j], m_j)
        elif local_mode == "ols":
            local_est_j = _local_ols_model(Z_calibration[j], m_j)
        else:
            local_est_j = None
        idx_tail = list(range(m_j, N_j_prime))
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu_g = mu_method["predict_global"](model_global=global_model, x_vector=z["X"], u_vector=U_calibration[j, :])
            if local_mode == "mean":
                mu_loc = local_est_j
                is_valid = mu_loc is not None
            elif local_mode == "ols":
                mu_loc = _predict_local_ols(local_est_j, z["X"]) if local_est_j is not None else None
                is_valid = mu_loc is not None
            else:
                mu_loc = None
                is_valid = False
            mu_tilde = _merge_mu(mu_g, mu_loc, m_j, s_size, is_valid)
            scores.append(abs(float(z["Y"]) - float(mu_tilde)))
        w_j = 1.0 / (s_size * len(idx_tail))
        weights.extend([w_j] * len(idx_tail))

    m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_within=use_within)
    if local_mode == "mean":
        local_est_test = _local_mean_y(Z_test, m_test)
    elif local_mode == "ols":
        local_est_test = _local_ols_model(Z_test, m_test)
    else:
        local_est_test = None
    idx_tail_test = list(range(m_test, o_observed)) if o_observed > m_test else []
    test_scores = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu_g = mu_method["predict_global"](model_global=global_model, x_vector=z["X"], u_vector=U_test[0, :])
        if local_mode == "mean":
            mu_loc = local_est_test
            is_valid = mu_loc is not None
        elif local_mode == "ols":
            mu_loc = _predict_local_ols(local_est_test, z["X"]) if local_est_test is not None else None
            is_valid = mu_loc is not None
        else:
            mu_loc = None
            is_valid = False
        mu_tilde = _merge_mu(mu_g, mu_loc, m_test, s_size, is_valid)
        test_scores.append(abs(float(z["Y"]) - float(mu_tilde)))

    n_tail_finite = len(test_scores)
    n_inf = max(0, N_donor - o_observed)
    n_total_test = n_tail_finite + n_inf
    if n_total_test > 0:
        w_test = 1.0 / (s_size * n_total_test)
        if n_tail_finite > 0:
            scores.extend(test_scores)
            weights.extend([w_test] * n_tail_finite)
        if n_inf > 0:
            scores.extend([np.inf] * n_inf)
            weights.extend([w_test] * n_inf)

    if len(scores) == 0 or all(w <= 0 for w in weights):
        q = np.inf
    else:
        q = weighted_quantile(scores, weights, alpha)

    x_target = Z_test[test_index_target]["X"]
    mu_g_target = mu_method["predict_global"](model_global=global_model, x_vector=x_target, u_vector=U_test[0, :])
    if local_mode == "mean":
        mu_loc_target = local_est_test
        is_valid_target = mu_loc_target is not None
    elif local_mode == "ols":
        mu_loc_target = _predict_local_ols(local_est_test, x_target) if local_est_test is not None else None
        is_valid_target = mu_loc_target is not None
    else:
        mu_loc_target = None
        is_valid_target = False
    mu_center = _merge_mu(mu_g_target, mu_loc_target, m_test, s_size, is_valid_target)
    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)
    return {"interval": interval, "mu_hat": float(mu_center), "number_selected_groups": int(s_size), "donor_group_index": int(donor)}


def _row(
    tau_b: float,
    alpha: float,
    experiment: int,
    target_group: int,
    o: int,
    method: str,
    interval: Tuple[float, float],
    y_true: float,
    baseline_computed_at_o: int | None = None,
    target_index: int = TARGET_INDEX,
    m_j_used: int | None = None,
    y_target: float | None = None,
) -> Dict:
    return {
        "tau_B": tau_b,
        "alpha": alpha,
        "experiment": experiment,
        "target_group": target_group,
        "o": o,
        "method": method,
        "coverage": _covered(interval, y_true),
        "width": _width(interval),
        "lower": interval[0],
        "upper": interval[1],
        "infinite": _is_infinite(interval),
        "baseline_computed_at_o": baseline_computed_at_o,
        "target_index": target_index,
        "m_j_used": m_j_used,
        "y_target": y_target,
    }


def run_one_experiment(exp_id: int, tau_b: float, alpha: float) -> pd.DataFrame:
    use_poisson = POISSON_MODE
    seed = BASE_SEED + int(tau_b * 1_000_000) + int(alpha * 10_000) + exp_id
    rng = np.random.default_rng(seed)

    u_hist, z_hist = generate_historical_groups(tau_b, rng, use_poisson=use_poisson)
    u_target, z_target = generate_target_groups(tau_b, N_TARGET_GROUPS_PER_EXPERIMENT, rng, use_poisson=use_poisson)

    sample_sizes = [N_HISTORICAL] * K_HISTORICAL
    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=ALPHA_SELECTION,
    )

    mu_baseline = create_mu_method_ols_global_only()
    model_baseline = mu_baseline["fit_global"](
        U_matrix=u_hist,
        Z_list=z_hist,
        group_index_vector=train_idx,
    )

    scores_list = []
    for j in calib_idx:
        z_j = z_hist[j]
        x_j = np.array([z["X"] for z in z_j], dtype=float)
        y_j = np.array([z["Y"] for z in z_j], dtype=float)
        mu_j = np.array(
            [
                mu_baseline["predict_global"](
                    model_global=model_baseline,
                    x_vector=x_j[i],
                    u_vector=u_hist[j, :],
                )
                for i in range(len(x_j))
            ],
            dtype=float,
        )
        scores_list.append(absolute_residual_score(y_j, mu_j))

    t_hcp = compute_hcp_interval_radius(scores_list, alpha)
    t_pool = compute_pooling_interval_radius(scores_list, alpha)
    t_sub = compute_subsampling_once_interval_radius(scores_list, alpha)
    t_rep = compute_repeated_subsampling_interval_radius(scores_list, alpha, N_REPEATED_SUBSAMPLING)

    mu_hcp_within = create_mu_method_ols_offset()

    rows: List[Dict] = []
    for target_group in range(N_TARGET_GROUPS_PER_EXPERIMENT):
        z_test = z_target[target_group]
        u_test = u_target[target_group].reshape(1, -1)
        x_target = z_test[TARGET_INDEX]["X"]
        y_true = z_test[TARGET_INDEX]["Y"]

        mu_hat = mu_baseline["predict_global"](
            model_global=model_baseline,
            x_vector=x_target,
            u_vector=u_test[0, :],
        )
        base_intervals = {
            "HCP": _baseline_interval(mu_hat, t_hcp),
            "Pooling": _baseline_interval(mu_hat, t_pool),
            "Subsampling": _baseline_interval(mu_hat, t_sub),
            "Repeated": _baseline_interval(mu_hat, t_rep),
        }

        for o in O_VALUES:
            for method_name, interval in base_intervals.items():
                rows.append(
                    _row(
                        tau_b=tau_b,
                        alpha=alpha,
                        experiment=exp_id,
                        target_group=target_group,
                        o=o,
                        method=method_name,
                        interval=interval,
                        y_true=y_true,
                        baseline_computed_at_o=0,
                        y_target=y_true,
                    )
                )

            tau_within = o // 2
            dhcp_seed = seed + (target_group + 1) * 1_009 + (o + 1) * 131 + 17
            if o == 0:
                interval_dhcp_within = base_intervals["HCP"]
            else:
                interval_dhcp_within = _compute_donor_hcp_interval_with_local(
                    U_calibration=u_hist,
                    Z_calibration=z_hist,
                    U_test=u_test,
                    Z_test=z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=ALPHA_SELECTION,
                    mu_method=mu_hcp_within,
                    local_mode=WITHIN_LOCAL_MODE,
                    test_index_target=TARGET_INDEX,
                    random_seed=dhcp_seed,
                )["interval"]
            rows.append(
                _row(
                    tau_b,
                    alpha,
                    exp_id,
                    target_group,
                    o,
                    "Donor-HCP-within",
                    interval_dhcp_within,
                    y_true,
                    m_j_used=tau_within,
                    y_target=y_true,
                )
            )

            if o == 0:
                interval_dhcp_no_within = base_intervals["HCP"]
            else:
                interval_dhcp_no_within = _compute_donor_hcp_interval_with_local(
                    U_calibration=u_hist,
                    Z_calibration=z_hist,
                    U_test=u_test,
                    Z_test=z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=ALPHA_SELECTION,
                    mu_method=mu_hcp_within,
                    local_mode="none",
                    test_index_target=TARGET_INDEX,
                    random_seed=dhcp_seed + 1,
                )["interval"]
            rows.append(
                _row(
                    tau_b,
                    alpha,
                    exp_id,
                    target_group,
                    o,
                    "Donor-HCP-no-within",
                    interval_dhcp_no_within,
                    y_true,
                    m_j_used=0,
                    y_target=y_true,
                )
            )

            interval_shcp_within = compute_sample_hcp_randomized_interval(
                U_calibration=u_hist,
                Z_calibration=z_hist,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp_within,
                test_index_target=TARGET_INDEX,
            )["interval"]
            rows.append(
                _row(
                    tau_b,
                    alpha,
                    exp_id,
                    target_group,
                    o,
                    "S-HCP-within",
                    interval_shcp_within,
                    y_true,
                    m_j_used=tau_within,
                    y_target=y_true,
                )
            )

            x_hist = np.array([z_test[i]["X"] for i in range(o)], dtype=float)
            y_hist = np.array([z_test[i]["Y"] for i in range(o)], dtype=float)
            interval_stdcp = _compute_std_cp_interval(x_hist=x_hist, y_hist=y_hist, x_target=x_target, alpha=alpha)
            rows.append(_row(tau_b, alpha, exp_id, target_group, o, "Std-CP", interval_stdcp, y_true, y_target=y_true))

    return pd.DataFrame(rows)


def run_experiment_wrapper(args: Tuple[int, float, float]) -> pd.DataFrame:
    return run_one_experiment(*args)


def run_randomization_stability_one_dataset(
    dataset_id: int, tau_b: float, alpha: float, o: int
) -> Tuple[List[Dict], Dict]:
    use_poisson = POISSON_MODE
    seed = BASE_SEED + 50_000_000 + int(tau_b * 1_000_000) + int(alpha * 10_000) + dataset_id
    rng = np.random.default_rng(seed)
    u_hist, z_hist = generate_historical_groups(tau_b, rng, use_poisson=use_poisson)
    u_target, z_target = generate_target_groups(tau_b, 1, rng, use_poisson=use_poisson)
    u_test = u_target[0].reshape(1, -1)
    z_test = z_target[0]
    mu_hcp_within = create_mu_method_ols_offset()

    sample_sizes = [N_HISTORICAL] * K_HISTORICAL
    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=ALPHA_SELECTION,
    )
    mu_baseline = create_mu_method_ols_global_only()
    model_baseline = mu_baseline["fit_global"](
        U_matrix=u_hist,
        Z_list=z_hist,
        group_index_vector=train_idx,
    )
    scores_list = []
    for j in calib_idx:
        z_j = z_hist[j]
        x_j = np.array([z["X"] for z in z_j], dtype=float)
        y_j = np.array([z["Y"] for z in z_j], dtype=float)
        mu_j = np.array(
            [
                mu_baseline["predict_global"](
                    model_global=model_baseline,
                    x_vector=x_j[i],
                    u_vector=u_hist[j, :],
                )
                for i in range(len(x_j))
            ],
            dtype=float,
        )
        scores_list.append(absolute_residual_score(y_j, mu_j))
    t_hcp = compute_hcp_interval_radius(scores_list, alpha)
    x_target = z_test[TARGET_INDEX]["X"]
    mu_hat = mu_baseline["predict_global"](
        model_global=model_baseline,
        x_vector=x_target,
        u_vector=u_test[0, :],
    )
    o0_hcp_interval = _baseline_interval(mu_hat, t_hcp)

    rerun_rows: List[Dict] = []
    widths = []
    uppers = []
    for r in range(N_RANDOMIZATION_RERUNS):
        if o == 0:
            interval = o0_hcp_interval
        else:
            interval = _compute_donor_hcp_interval_with_local(
                U_calibration=u_hist,
                Z_calibration=z_hist,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp_within,
                local_mode=WITHIN_LOCAL_MODE,
                test_index_target=TARGET_INDEX,
                random_seed=seed + 1000 + r,
            )["interval"]
        w = _width(interval)
        rerun_rows.append(
            {
                "tau_B": tau_b,
                "alpha": alpha,
                "o": o,
                "dataset_id": dataset_id,
                "rerun_id": r,
                "lower": interval[0],
                "upper": interval[1],
                "width": w,
                "finite": int(np.isfinite(w)),
            }
        )
        if np.isfinite(w):
            widths.append(w)
            uppers.append(interval[1])

    if len(widths) == 0:
        dataset_metrics = {
            "tau_B": tau_b,
            "alpha": alpha,
            "o": o,
            "dataset_id": dataset_id,
            "mean_width": np.nan,
            "sd_upper": np.nan,
            "sd_width": np.nan,
            "relative_instability": np.nan,
            "n_finite": 0,
            "randomization_varied_only": True,
        }
    else:
        mean_width = float(np.mean(widths))
        sd_upper = float(np.std(uppers, ddof=1)) if len(uppers) > 1 else 0.0
        sd_width = float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0
        rel_instability = (sd_upper / mean_width) if mean_width > 1e-12 else np.nan
        dataset_metrics = {
            "tau_B": tau_b,
            "alpha": alpha,
            "o": o,
            "dataset_id": dataset_id,
            "mean_width": mean_width,
            "sd_upper": sd_upper,
            "sd_width": sd_width,
            "relative_instability": rel_instability,
            "n_finite": len(widths),
            "randomization_varied_only": True,
        }
    return rerun_rows, dataset_metrics


def run_stability_wrapper(args: Tuple[int, float, float, int]) -> Tuple[List[Dict], Dict]:
    return run_randomization_stability_one_dataset(*args)


def _boxplot_style(
    ax,
    data: np.ndarray,
    pos: float,
    color: str,
    width: float = 0.58,
    edge_color: str | None = None,
    hatch: str | None = None,
    alpha_val: float = 0.75,
    linestyle: str = "-",
):
    if edge_color is None:
        edge_color = color
    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        capprops=dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        flierprops=dict(marker=".", color=edge_color, markersize=3, alpha=0.4),
        boxprops=dict(facecolor=color, alpha=alpha_val, edgecolor=edge_color, linewidth=1.0, linestyle=linestyle),
    )
    if hatch:
        for patch in bp["boxes"]:
            patch.set_hatch(hatch)
    return bp


def _exp_metric(df: pd.DataFrame, method: str, o: int, metric: str) -> np.ndarray:
    d = df[(df["method"] == method) & (df["o"] == o)]
    vals = d.groupby("experiment", as_index=False)[metric].mean()[metric].to_numpy()
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def _exp_metric_o_independent(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    d = df[df["method"] == method]
    vals = d.groupby("experiment", as_index=False)[metric].mean()[metric].to_numpy()
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def _plot_two_panel_boxes(
    df: pd.DataFrame,
    tau_b: float,
    alpha: float,
    path: Path,
    o_vals: Iterable[int],
    compare_methods: List[Tuple[str, Dict]],
    include_hcp_baseline: bool,
    title_cov: str,
    title_width: str,
):
    d = df[(df["tau_B"] == tau_b) & (df["alpha"] == alpha)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, metric, title in [(axes[0], "coverage", title_cov), (axes[1], "width", title_width)]:
        pos = 0.0
        tick_pos = []
        tick_lab = []
        for o in o_vals:
            for method, style in compare_methods:
                vals = _exp_metric(d, method, o, metric)
                if len(vals):
                    _boxplot_style(
                        ax,
                        vals,
                        pos,
                        color=style["color"](o) if callable(style["color"]) else style["color"],
                        edge_color=style.get("edge_color"),
                        hatch=style.get("hatch"),
                        alpha_val=style.get("alpha", 0.75),
                        linestyle=style.get("linestyle", "-"),
                    )
                pos += style.get("step", 0.8)
            tick_pos.append(pos - 0.8)
            tick_lab.append(f"o={o}")
            pos += 0.5
        if include_hcp_baseline:
            vals_h = _exp_metric_o_independent(d, "HCP", metric)
            if len(vals_h):
                _boxplot_style(ax, vals_h, pos, color="#111111", edge_color="#111111", hatch="///", alpha_val=0.28, linestyle="--")
                tick_pos.append(pos)
                tick_lab.append("HCP")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=18, ha="right", fontsize=14)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Method", fontsize=16)
        ax.set_ylabel("Coverage" if metric == "coverage" else "Width", fontsize=18)
        ax.tick_params(axis="y", labelsize=13)
        if metric == "coverage":
            ax.axhline(1 - alpha, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
            ax.set_ylim(0.55, 1.05)
        ax.set_title(title, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_requested_sets(df_all: pd.DataFrame, stability_summary: pd.DataFrame) -> None:
    mode_tag = _mode_tag()
    for alpha_plot in ALPHA_VALUES:
        for tau_b in TAU_B_VALUES:
            tag = f"{mode_tag}_tauB{int(tau_b)}_{_alpha_tag(alpha_plot)}"

            _plot_two_panel_boxes(
                df_all,
                tau_b,
                alpha_plot,
                PLOT_DIR / f"{tag}_1_dhcp_with_vs_hcp_upto20.pdf",
                [o for o in O_VALUES if o <= 20],
                compare_methods=[("Donor-HCP-within", {"color": lambda o: O_COLORS[o], "step": 0.8})],
                include_hcp_baseline=True,
                title_cov="Coverage: Donor-HCP(within) vs HCP",
                title_width="Width: Donor-HCP(within) vs HCP",
            )

            _plot_two_panel_boxes(
                df_all,
                tau_b,
                alpha_plot,
                PLOT_DIR / f"{tag}_2_dhcp_with_vs_hcp_upto35.pdf",
                O_VALUES,
                compare_methods=[("Donor-HCP-within", {"color": lambda o: O_COLORS[o], "step": 0.8})],
                include_hcp_baseline=True,
                title_cov="Coverage: Donor-HCP(within) vs HCP",
                title_width="Width: Donor-HCP(within) vs HCP",
            )

            d = df_all[(df_all["tau_B"] == tau_b) & (df_all["alpha"] == alpha_plot)].copy()
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            for ax, metric in [(axes[0], "coverage"), (axes[1], "width")]:
                pos = 0.0
                ticks, labs = [], []
                for o in O_VALUES:
                    vals = _exp_metric(d, "Donor-HCP-within", o, metric)
                    if len(vals):
                        _boxplot_style(ax, vals, pos, O_COLORS[o])
                        ticks.append(pos)
                        labs.append(f"o={o}")
                    pos += 1.0
                pos += 1.0
                for method in ["HCP", "Pooling", "Subsampling", "Repeated"]:
                    vals = _exp_metric_o_independent(d, method, metric)
                    if len(vals):
                        _boxplot_style(ax, vals, pos, BASELINE_GRAYS[method], edge_color="#222", hatch="//", alpha_val=0.45, linestyle="--")
                        ticks.append(pos)
                        labs.append(method)
                    pos += 1.5
                ax.set_xticks(ticks)
                ax.set_xticklabels(labs, rotation=18, ha="right", fontsize=14)
                ax.grid(axis="y", alpha=0.25)
                ax.set_xlabel("Method", fontsize=16)
                ax.set_ylabel("Coverage" if metric == "coverage" else "Width", fontsize=18)
                if metric == "coverage":
                    ax.axhline(1 - alpha_plot, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
                    ax.set_ylim(0.55, 1.05)
                    ax.set_title("Coverage: Donor-HCP(within) vs baselines", fontsize=18, fontweight="bold")
                else:
                    ax.set_title("Width: Donor-HCP(within) vs baselines", fontsize=18, fontweight="bold")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"{tag}_3_dhcp_with_vs_all_baselines.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            _plot_two_panel_boxes(
                df_all,
                tau_b,
                alpha_plot,
                PLOT_DIR / f"{tag}_4_dhcp_with_vs_shcp_with_hcp_upto20.pdf",
                [o for o in O_VALUES if o <= 20],
                compare_methods=[
                    ("Donor-HCP-within", {"color": lambda o: O_COLORS[o], "step": 0.7}),
                    ("S-HCP-within", {"color": "#7F8C8D", "hatch": "xx", "alpha": 0.6, "step": 0.8}),
                ],
                include_hcp_baseline=True,
                title_cov="Coverage: Donor-HCP(within) vs S-HCP(within)",
                title_width="Width: Donor-HCP(within) vs S-HCP(within)",
            )

            _plot_two_panel_boxes(
                df_all,
                tau_b,
                alpha_plot,
                PLOT_DIR / f"{tag}_5_dhcp_with_vs_no_within_hcp_upto35.pdf",
                O_VALUES,
                compare_methods=[
                    ("Donor-HCP-within", {"color": lambda o: O_COLORS[o], "step": 0.7}),
                    ("Donor-HCP-no-within", {"color": "#f4a582", "hatch": "///", "alpha": 0.6, "edge_color": "#333", "step": 0.8}),
                ],
                include_hcp_baseline=True,
                title_cov="Coverage: Donor-HCP with vs without within-training",
                title_width="Width: Donor-HCP with vs without within-training",
            )

            stab = stability_summary[
                (stability_summary["tau_B"] == tau_b) & (stability_summary["alpha"] == alpha_plot)
            ].sort_values("o")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stab["o"], stab["relative_instability_mean"], marker="o", linewidth=2.0, color="#0072B2")
            ax.set_xlabel("Observed history size o", fontsize=16)
            ax.set_ylabel("Relative instability: SD(upper) / mean width", fontsize=16)
            ax.grid(alpha=0.25)
            ax.set_title("Randomization stability (Donor-HCP within)", fontsize=18, fontweight="bold")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"{tag}_6_randomization_stability.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    mode_tag = _mode_tag()
    print(f"Running latent-intercept paired experiment with {N_WORKERS} cores.")
    print(f"Run mode: {mode_tag}")
    print("All methods are evaluated on the same realized dataset within each outer replicate.")
    print(f"DGP: U ~ Unif([{U_MIN},{U_MAX}]^{DIMENSION}), target index={TARGET_INDEX}")

    all_results = []
    for tau_b in TAU_B_VALUES:
        for alpha in ALPHA_VALUES:
            args = [(exp_id, tau_b, alpha) for exp_id in range(N_EXPERIMENTS)]
            print(f"[Main] Running experiments for tau_B={tau_b}, alpha={alpha} ({len(args)} jobs)")
            with Pool(N_WORKERS) as pool:
                chunks = []
                for i, chunk in enumerate(pool.imap_unordered(run_experiment_wrapper, args), start=1):
                    chunks.append(chunk)
                    if i % 5 == 0 or i == len(args):
                        bar = _progress_bar(i, len(args))
                        print(
                            f"[Main] {bar} {i}/{len(args)} outer experiments "
                            f"for tau_B={tau_b}, alpha={alpha}"
                        )
            all_results.append(pd.concat(chunks, ignore_index=True))

    df_all = pd.concat(all_results, ignore_index=True)

    target_uniques = (
        df_all.groupby(["tau_B", "alpha", "experiment", "target_group"], as_index=False)["y_target"]
        .nunique()
        .rename(columns={"y_target": "y_target_nunique"})
    )
    if (target_uniques["y_target_nunique"] != 1).any():
        raise RuntimeError("Paired-data invariant failed: target value changed within an experiment/target dataset.")

    d_with = df_all[df_all["method"] == "Donor-HCP-within"][["o", "m_j_used"]].drop_duplicates()
    if not (d_with["m_j_used"].to_numpy() == (d_with["o"].to_numpy() // 2)).all():
        raise RuntimeError("Within-training invariant failed: expected m_j = floor(o/2) for Donor-HCP-within.")

    d_no = df_all[df_all["method"] == "Donor-HCP-no-within"]["m_j_used"].dropna()
    if not (d_no.to_numpy() == 0).all():
        raise RuntimeError("No-within invariant failed: expected m_j = 0 for Donor-HCP-no-within.")

    raw_path = OUT_DIR / f"{mode_tag}_raw_results_complete.csv"
    df_all.to_csv(raw_path, index=False)

    for alpha in ALPHA_VALUES:
        raw_alpha_path = OUT_DIR / f"{mode_tag}_{_alpha_tag(alpha)}_raw_results.csv"
        df_all[df_all["alpha"] == alpha].to_csv(raw_alpha_path, index=False)

    summary = (
        df_all.groupby(["tau_B", "alpha", "o", "method"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            width_mean=("width", "mean"),
            width_median=("width", "median"),
            width_std=("width", "std"),
            proportion_infinite=("infinite", "mean"),
            n_infinite=("infinite", "sum"),
        )
    )
    # Record base seed for reproducibility
    summary["base_seed"] = BASE_SEED
    summary_path = OUT_DIR / f"{mode_tag}_summary_by_tau_alpha_o_method.csv"
    summary.to_csv(summary_path, index=False)

    exp_summary = (
        df_all.groupby(["tau_B", "alpha", "experiment", "o", "method"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            width_mean=("width", "mean"),
            width_median=("width", "median"),
            width_std=("width", "std"),
            n_infinite=("infinite", "sum"),
        )
    )
    exp_summary["base_seed"] = BASE_SEED
    exp_summary_path = OUT_DIR / f"{mode_tag}_experiment_level_summary.csv"
    exp_summary.to_csv(exp_summary_path, index=False)

    print("Running randomization-stability experiment (varying donor-HCP randomization seed only).")
    rerun_rows_all: List[Dict] = []
    dataset_rows_all: List[Dict] = []
    for tau_b in TAU_B_VALUES:
        for alpha in ALPHA_VALUES:
            for o in O_VALUES:
                args = [(dataset_id, tau_b, alpha, o) for dataset_id in range(N_RANDOMIZATION_DATASETS)]
                print(f"[Stability] Running tau_B={tau_b}, alpha={alpha}, o={o} ({len(args)} datasets)")
                with Pool(N_WORKERS) as pool:
                    out_count = 0
                    for reruns, ds_row in pool.imap_unordered(run_stability_wrapper, args):
                        rerun_rows_all.extend(reruns)
                        dataset_rows_all.append(ds_row)
                        out_count += 1
                        if out_count % 10 == 0 or out_count == len(args):
                            bar = _progress_bar(out_count, len(args))
                            print(
                                f"[Stability] {bar} {out_count}/{len(args)} datasets "
                                f"for tau_B={tau_b}, alpha={alpha}, o={o}"
                            )

    df_stability_reruns = pd.DataFrame(rerun_rows_all)
    df_stability_dataset = pd.DataFrame(dataset_rows_all)
    stability_reruns_path = OUT_DIR / f"{mode_tag}_randomization_stability_reruns.csv"
    stability_dataset_path = OUT_DIR / f"{mode_tag}_randomization_stability_dataset_metrics.csv"
    df_stability_reruns.to_csv(stability_reruns_path, index=False)
    df_stability_dataset.to_csv(stability_dataset_path, index=False)

    stability_summary = (
        df_stability_dataset.groupby(["tau_B", "alpha", "o"], as_index=False)
        .agg(
            mean_width_mean=("mean_width", "mean"),
            sd_upper_mean=("sd_upper", "mean"),
            sd_width_mean=("sd_width", "mean"),
            relative_instability_mean=("relative_instability", "mean"),
            n_finite_total=("n_finite", "sum"),
        )
    )
    stability_summary["base_seed"] = BASE_SEED
    stability_summary_path = OUT_DIR / f"{mode_tag}_randomization_stability_summary.csv"
    stability_summary.to_csv(stability_summary_path, index=False)

    plot_requested_sets(df_all, stability_summary)

    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved experiment-level summary: {exp_summary_path}")
    print(f"Saved stability reruns: {stability_reruns_path}")
    print(f"Saved stability metrics: {stability_dataset_path}")
    print(f"Saved stability summary: {stability_summary_path}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()