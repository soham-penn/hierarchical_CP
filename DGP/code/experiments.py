"""
Experiment Runner

This module contains functions for running experiments comparing different
hierarchical conformal prediction methods.
"""

import numpy as np
import pandas as pd
from .data_generation import generate_calibration_data, generate_test_group
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius
)
from methods.donor_hcp import (
    compute_donor_hcp_randomized_interval,
    compute_donor_hcp_derandomized_interval,
    get_hcp_train_cal_split,
)
from methods.sample_hcp import (
    compute_sample_hcp_randomized_interval,
    compute_sample_hcp_derandomized_interval,
)
from scores import absolute_residual_score


def _safe_width_median(width_array):
    """Return median finite width; if none are finite, return +inf."""
    arr = np.asarray(width_array, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.inf
    return float(np.median(finite))


def _split_conformal_radius(abs_residuals, alpha):
    """Finite-sample split-conformal radius from calibration residuals."""
    scores = np.asarray(abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(1, k), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_std_cp_interval(x_hist, y_hist, x_target, alpha):
    """Standard split CP interval using only within-group history."""
    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)

    n_hist = len(y_hist)
    n_train = n_hist // 2
    n_cal = n_hist - n_train
    if n_train < 1 or n_cal < 1:
        return (-np.inf, np.inf)

    perm = np.random.permutation(n_hist)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    x_train = x_hist[train_idx]
    y_train = y_hist[train_idx]
    x_cal = x_hist[cal_idx]
    y_cal = y_hist[cal_idx]

    x_train_aug = np.column_stack([np.ones(len(x_train)), x_train])
    x_cal_aug = np.column_stack([np.ones(len(x_cal)), x_cal])

    beta, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    y_cal_pred = x_cal_aug @ beta
    radius = _split_conformal_radius(np.abs(y_cal - y_cal_pred), alpha)

    x_target = np.asarray(x_target, dtype=float).reshape(1, -1)
    x_target_aug = np.column_stack([np.ones(1), x_target])
    y_hat = float((x_target_aug @ beta)[0])

    if np.isfinite(radius):
        return (y_hat - radius, y_hat + radius)
    return (-np.inf, np.inf)


def run_one_experiment(number_groups_k, lambda_Poisson, dgp_specification,
                       o_values, target_index, alpha, number_subsampling_repetitions,
                       alpha_selection, number_test_groups,
                       mu_method_baseline, mu_method_hcp,
                       mu_method_hcp_no_within=None):
    """
    Run one experiment comparing all methods across multiple o values.

    All o values share the same calibration data, the same fitted model, and
    the same test groups.  The prediction target is always Z_test[target_index],
    so baseline methods (which ignore test-group history) produce identical
    results for every o value — making the comparison across o values fair.

    Parameters
    ----------
    o_values : list of int
        History sizes to evaluate.  Each must satisfy o <= target_index.
    target_index : int
        0-based index of the prediction target in each test group.
        Test groups are generated to have at least target_index+1 observations.
    """
    # ------------------------------------------------------------------
    # 1. Generate calibration data (once for all o values)
    # ------------------------------------------------------------------
    cal = generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification
    )
    U_cal = cal['U_calibration']
    Z_cal = cal['Z_calibration']

    # ------------------------------------------------------------------
    # 2. Train/calib split (HCP uses independent split without donor removal)
    # ------------------------------------------------------------------
    sample_sizes = [len(zg) for zg in Z_cal]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_selection,
    )

    # ------------------------------------------------------------------
    # 3. Fit baseline model (once)
    # ------------------------------------------------------------------
    model_baseline = mu_method_baseline['fit_global'](
        U_matrix=U_cal,
        Z_list=Z_cal,
        group_index_vector=train_idx
    )

    # ------------------------------------------------------------------
    # 4. Compute baseline conformity scores on calib_idx (once)
    # ------------------------------------------------------------------
    scores_list = []
    for j in calib_idx:
        Zj = Z_cal[j]
        Uj = U_cal[j, :]
        yj = np.array([z['Y'] for z in Zj])
        Xj = np.array([z['X'] for z in Zj])
        muj = np.array([
            mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=Xj[i],
                u_vector=Uj
            ) for i in range(len(Xj))
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    # Baseline radii — fixed for the whole experiment, same across all o
    T_hcp  = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub  = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep  = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, number_subsampling_repetitions
    )

    # ------------------------------------------------------------------
    # 5. Initialise result arrays
    #    donor-HCP/sample-HCP variants: one array per o value.
    #    Baselines: one shared array (identical across o values).
    # ------------------------------------------------------------------
    cov_dr  = {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}
    cov_dr_no_within = {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}
    cov_dd  = {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}
    cov_sr  = {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}
    cov_sd  = {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}
    cov_stdcp = {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}
    wid_dr  = {o: np.full(number_test_groups, np.nan) for o in o_values}
    wid_dr_no_within = {o: np.full(number_test_groups, np.nan) for o in o_values}
    wid_dd  = {o: np.full(number_test_groups, np.nan) for o in o_values}
    wid_sr  = {o: np.full(number_test_groups, np.nan) for o in o_values}
    wid_sd  = {o: np.full(number_test_groups, np.nan) for o in o_values}
    wid_stdcp = {o: np.full(number_test_groups, np.nan) for o in o_values}
    inf_dr  = {o: 0 for o in o_values}
    inf_dr_no_within = {o: 0 for o in o_values}
    inf_dd  = {o: 0 for o in o_values}
    inf_sr  = {o: 0 for o in o_values}
    inf_sd  = {o: 0 for o in o_values}
    inf_stdcp = {o: 0 for o in o_values}

    cov_hcp  = np.zeros(number_test_groups, dtype=bool)
    cov_pool = np.zeros(number_test_groups, dtype=bool)
    cov_sub  = np.zeros(number_test_groups, dtype=bool)
    cov_rep  = np.zeros(number_test_groups, dtype=bool)
    wid_hcp  = np.full(number_test_groups, np.nan)
    wid_pool = np.full(number_test_groups, np.nan)
    wid_sub  = np.full(number_test_groups, np.nan)
    wid_rep  = np.full(number_test_groups, np.nan)
    inf_hcp = inf_pool = inf_sub = inf_rep = 0

    # ------------------------------------------------------------------
    # 6. Evaluate on test groups
    # ------------------------------------------------------------------
    for t in range(number_test_groups):
        # Generate test group ensuring N >= target_index + 1
        test = generate_test_group(
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_observed=target_index
        )
        U_test = test['U_test']
        Z_test = test['Z_test']
        N_test = test['N_test']

        if N_test < target_index + 1:
            continue

        true_target = Z_test[target_index]['Y']
        X_target    = Z_test[target_index]['X']

        # Baseline center prediction — fixed target, same for all o
        mu_baseline_hat = mu_method_baseline['predict_global'](
            model_global=model_baseline,
            x_vector=X_target,
            u_vector=U_test[0, :]
        )

        # ---- Baseline methods (computed once per test group) ----
        for T, cov_arr, wid_arr in [
            (T_hcp,  cov_hcp,  wid_hcp),
            (T_pool, cov_pool, wid_pool),
            (T_sub,  cov_sub,  wid_sub),
            (T_rep,  cov_rep,  wid_rep),
        ]:
            lo = mu_baseline_hat - T if np.isfinite(T) else -np.inf
            hi = mu_baseline_hat + T if np.isfinite(T) else  np.inf
            cov_arr[t] = (lo <= true_target <= hi)
            if np.isfinite(lo) and np.isfinite(hi):
                wid_arr[t] = hi - lo
        if not np.isfinite(T_hcp):  inf_hcp  += 1
        if not np.isfinite(T_pool): inf_pool += 1
        if not np.isfinite(T_sub):  inf_sub  += 1
        if not np.isfinite(T_rep):  inf_rep  += 1

        # ---- donor-HCP and sample-HCP variants for each o ----
        for o in o_values:
            res_dr = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_selection,
                mu_method=mu_method_hcp,
                test_index_target=target_index,
            )
            lo, hi = res_dr['interval']
            cov_dr[o][t] = (lo <= true_target <= hi)
            if np.isfinite(lo) and np.isfinite(hi):
                wid_dr[o][t] = hi - lo
            else:
                inf_dr[o] += 1

            if mu_method_hcp_no_within is not None:
                res_dr_no = compute_donor_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp_no_within,
                    test_index_target=target_index,
                    tau_override=0,
                )
                lo, hi = res_dr_no['interval']
                cov_dr_no_within[o][t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_dr_no_within[o][t] = hi - lo
                else:
                    inf_dr_no_within[o] += 1

            res_dd = compute_donor_hcp_derandomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                test_index_target=target_index,
                alpha_selection=alpha_selection,
                mu_method=mu_method_hcp,
            )
            lo, hi = res_dd['interval']
            cov_dd[o][t] = (lo <= true_target <= hi)
            if np.isfinite(lo) and np.isfinite(hi):
                wid_dd[o][t] = hi - lo
            else:
                inf_dd[o] += 1

            res_sr = compute_sample_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                test_index_target=target_index,
                alpha_selection=alpha_selection,
                mu_method=mu_method_hcp
            )
            lo, hi = res_sr['interval']
            cov_sr[o][t] = (lo <= true_target <= hi)
            if np.isfinite(lo) and np.isfinite(hi):
                wid_sr[o][t] = hi - lo
            else:
                inf_sr[o] += 1

            res_sd = compute_sample_hcp_derandomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_selection,
                mu_method=mu_method_hcp,
                test_index_target=target_index,
            )
            lo, hi = res_sd['interval']
            cov_sd[o][t] = (lo <= true_target <= hi)
            if np.isfinite(lo) and np.isfinite(hi):
                wid_sd[o][t] = hi - lo
            else:
                inf_sd[o] += 1

            # Standard split CP inside test group using first o points as history.
            x_hist = np.array([Z_test[i]['X'] for i in range(o)])
            y_hist = np.array([Z_test[i]['Y'] for i in range(o)])
            lo, hi = _compute_std_cp_interval(
                x_hist=x_hist,
                y_hist=y_hist,
                x_target=X_target,
                alpha=alpha,
            )
            cov_stdcp[o][t] = (lo <= true_target <= hi)
            if np.isfinite(lo) and np.isfinite(hi):
                wid_stdcp[o][t] = hi - lo
            else:
                inf_stdcp[o] += 1

    # ------------------------------------------------------------------
    # 7. Aggregate: one row per o value; baselines replicated across o
    # ------------------------------------------------------------------
    rows = []
    for o in o_values:
        rows.append({
            'o_observed':          o,
            'coverage_donor_hcp_randomized': np.mean(cov_dr[o]),
            'coverage_donor_hcp_no_within': np.mean(cov_dr_no_within[o]) if mu_method_hcp_no_within is not None else np.nan,
            'coverage_donor_hcp_derandomized': np.mean(cov_dd[o]),
            'coverage_sample_hcp_randomized': np.mean(cov_sr[o]),
            'coverage_sample_hcp_derandomized': np.mean(cov_sd[o]),
            'coverage_stdcp':     np.mean(cov_stdcp[o]),
            'coverage_hcp':        np.mean(cov_hcp),
            'coverage_pool':       np.mean(cov_pool),
            'coverage_sub':        np.mean(cov_sub),
            'coverage_rep':        np.mean(cov_rep),
            'width_donor_hcp_randomized': _safe_width_median(wid_dr[o]),
            'width_donor_hcp_no_within': _safe_width_median(wid_dr_no_within[o]) if mu_method_hcp_no_within is not None else np.nan,
            'width_donor_hcp_derandomized': _safe_width_median(wid_dd[o]),
            'width_sample_hcp_randomized': _safe_width_median(wid_sr[o]),
            'width_sample_hcp_derandomized': _safe_width_median(wid_sd[o]),
            'width_stdcp':        _safe_width_median(wid_stdcp[o]),
            'width_hcp':           _safe_width_median(wid_hcp),
            'width_pool':          _safe_width_median(wid_pool),
            'width_sub':           _safe_width_median(wid_sub),
            'width_rep':           _safe_width_median(wid_rep),
            'infinite_donor_hcp_randomized': inf_dr[o],
            'infinite_donor_hcp_no_within': inf_dr_no_within[o] if mu_method_hcp_no_within is not None else np.nan,
            'infinite_donor_hcp_derandomized': inf_dd[o],
            'infinite_sample_hcp_randomized': inf_sr[o],
            'infinite_sample_hcp_derandomized': inf_sd[o],
            'infinite_stdcp':     inf_stdcp[o],
            'infinite_hcp':        inf_hcp,
            'infinite_pool':       inf_pool,
            'infinite_sub':        inf_sub,
            'infinite_rep':        inf_rep,
        })
    return pd.DataFrame(rows)


def run_experiments_outer(number_experiments, number_groups_k, lambda_Poisson,
                          dgp_specification, o_values, target_index, alpha=0.1,
                          number_subsampling_repetitions=50,
                          alpha_selection=0.1, number_test_groups=100,
                          mu_method_baseline=None, mu_method_hcp=None,
                          mu_method_hcp_no_within=None,
                          show_progress=True):
    """
    Run multiple experiments and return combined results.

    Parameters
    ----------
    o_values : list of int
        History sizes to evaluate within each experiment.
    target_index : int
        0-based index of the fixed prediction target in each test group.
    """
    if show_progress:
        print(f"Running {number_experiments} experiments sequentially...")

    results_list = []
    for e in range(number_experiments):
        if show_progress and (e + 1) % 5 == 0:
            print(f"  Completed {e + 1}/{number_experiments} experiments")

        res = run_one_experiment(
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_values=o_values,
            target_index=target_index,
            alpha=alpha,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_method_baseline,
            mu_method_hcp=mu_method_hcp,
            mu_method_hcp_no_within=mu_method_hcp_no_within,
        )
        res['experiment'] = e + 1
        results_list.append(res)

    combined = pd.concat(results_list, ignore_index=True)
    cols = ['experiment'] + [c for c in combined.columns if c != 'experiment']
    return combined[cols]


# ======================================================================
# CONDITIONAL COVERAGE EXPERIMENT
# ======================================================================

def run_one_experiment_conditional(
        number_groups_k, lambda_Poisson, dgp_specification,
        o_values, target_index, u_grid, alpha,
        number_subsampling_repetitions, alpha_selection,
        number_test_groups, mu_method_baseline, mu_method_hcp):
    """
    One U-conditional-coverage experiment.

    For each value ``u_g`` in ``u_grid`` we generate
    ``number_test_groups`` independent test groups whose group covariate
    is fixed at  U = [u_g, 0.5, …, 0.5].  Within each test group, X
    and Y are drawn from P(X, Y | U) as usual.

    This tests the group-level (U-conditional) coverage guarantee:
    for a new group with a specific U value, does the method cover
    Y_{target_index} with probability ≥ 1 − α?

    Returns
    -------
    pd.DataFrame   One row per (o_observed, u_grid_val).
    """
    d = dgp_specification['dimension']

    # ------------------------------------------------------------------
    # 1. Calibration data (once for the whole experiment)
    # ------------------------------------------------------------------
    cal   = generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
    )
    U_cal = cal['U_calibration']
    Z_cal = cal['Z_calibration']

    # ------------------------------------------------------------------
    # 2. Train / calib split (HCP uses independent split without donor removal)
    # ------------------------------------------------------------------
    sample_sizes = [len(zg) for zg in Z_cal]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_selection,
    )

    # ------------------------------------------------------------------
    # 3. Fit baseline model (once)
    # ------------------------------------------------------------------
    model_baseline = mu_method_baseline['fit_global'](
        U_matrix=U_cal,
        Z_list=Z_cal,
        group_index_vector=train_idx,
    )

    # ------------------------------------------------------------------
    # 4. Calibration scores → baseline radii (once)
    # ------------------------------------------------------------------
    scores_list = []
    for j in calib_idx:
        Zj  = Z_cal[j]
        Uj  = U_cal[j, :]
        yj  = np.array([z['Y'] for z in Zj])
        Xj  = np.array([z['X'] for z in Zj])
        muj = np.array([
            mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=Xj[i],
                u_vector=Uj,
            ) for i in range(len(Xj))
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    T_hcp  = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub  = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep  = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, number_subsampling_repetitions,
    )

    # ------------------------------------------------------------------
    # 5. Allocate result arrays   shape: (n_o, n_grid, n_test)
    # ------------------------------------------------------------------
    G  = len(u_grid)
    O  = len(o_values)
    NT = number_test_groups

    cov_dr  = np.zeros((O, G, NT))
    cov_dd  = np.zeros((O, G, NT))
    cov_sr  = np.zeros((O, G, NT))
    cov_sd  = np.zeros((O, G, NT))
    wid_dr  = np.full((O, G, NT), np.nan)
    wid_dd  = np.full((O, G, NT), np.nan)
    wid_sr  = np.full((O, G, NT), np.nan)
    wid_sd  = np.full((O, G, NT), np.nan)
    inf_dr  = np.zeros((O, G), dtype=int)
    inf_dd  = np.zeros((O, G), dtype=int)
    inf_sr  = np.zeros((O, G), dtype=int)
    inf_sd  = np.zeros((O, G), dtype=int)

    cov_hcp  = np.zeros((G, NT))
    cov_pool = np.zeros((G, NT))
    cov_sub  = np.zeros((G, NT))
    cov_rep  = np.zeros((G, NT))
    wid_hcp  = np.full((G, NT), np.nan)
    wid_pool = np.full((G, NT), np.nan)
    wid_sub  = np.full((G, NT), np.nan)
    wid_rep  = np.full((G, NT), np.nan)

    # Baseline radii are the same for all (u_g, t); count inf once
    inf_hcp  = (G * NT) if not np.isfinite(T_hcp)  else 0
    inf_pool = (G * NT) if not np.isfinite(T_pool) else 0
    inf_sub  = (G * NT) if not np.isfinite(T_sub)  else 0
    inf_rep  = (G * NT) if not np.isfinite(T_rep)  else 0

    # ------------------------------------------------------------------
    # 6. Main loop: for each U-grid point, generate NT test groups
    # ------------------------------------------------------------------
    for g, u_g in enumerate(u_grid):
        # Fix U = [u_g, 0.5, …, 0.5]  (entire U vector is determined)
        U_fixed = np.full(d, 0.5)
        U_fixed[0] = float(u_g)

        for t in range(NT):
            test = generate_test_group(
                lambda_Poisson=lambda_Poisson,
                dgp_specification=dgp_specification,
                o_observed=target_index,
                fixed_U=U_fixed,
            )
            U_test = test['U_test']
            Z_test = test['Z_test']
            N_test = test['N_test']

            if N_test < target_index + 1:
                continue

            true_target = Z_test[target_index]['Y']
            X_target    = Z_test[target_index]['X']
            U_t         = U_test[0, :]

            # ---- Baseline methods ----
            mu_hat = mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=X_target,
                u_vector=U_t,
            )
            for T_rad, c_arr, w_arr in [
                (T_hcp,  cov_hcp[g],  wid_hcp[g]),
                (T_pool, cov_pool[g], wid_pool[g]),
                (T_sub,  cov_sub[g],  wid_sub[g]),
                (T_rep,  cov_rep[g],  wid_rep[g]),
            ]:
                lo = mu_hat - T_rad if np.isfinite(T_rad) else -np.inf
                hi = mu_hat + T_rad if np.isfinite(T_rad) else  np.inf
                c_arr[t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    w_arr[t] = hi - lo

            # ---- donor-HCP and sample-HCP variants for each o value ----
            for oi, o in enumerate(o_values):
                res_dr = compute_donor_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                    test_index_target=target_index,
                )
                lo, hi = res_dr['interval']
                cov_dr[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_dr[oi, g, t] = hi - lo
                else:
                    inf_dr[oi, g] += 1

                res_dd = compute_donor_hcp_derandomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    test_index_target=target_index,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                )
                lo, hi = res_dd['interval']
                cov_dd[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_dd[oi, g, t] = hi - lo
                else:
                    inf_dd[oi, g] += 1

                res_sr = compute_sample_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    test_index_target=target_index,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                )
                lo, hi = res_sr['interval']
                cov_sr[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_sr[oi, g, t] = hi - lo
                else:
                    inf_sr[oi, g] += 1

                res_sd = compute_sample_hcp_derandomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                    test_index_target=target_index,
                )
                lo, hi = res_sd['interval']
                cov_sd[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_sd[oi, g, t] = hi - lo
                else:
                    inf_sd[oi, g] += 1

    # ------------------------------------------------------------------
    # 7. Aggregate: one row per (o, u_g)
    # ------------------------------------------------------------------
    rows = []
    for oi, o in enumerate(o_values):
        for g, u_g in enumerate(u_grid):
            rows.append({
                'o_observed':          o,
                'u_grid_val':          float(u_g),
                'coverage_donor_hcp_randomized': np.mean(cov_dr[oi, g]),
                'coverage_donor_hcp_derandomized': np.mean(cov_dd[oi, g]),
                'coverage_sample_hcp_randomized': np.mean(cov_sr[oi, g]),
                'coverage_sample_hcp_derandomized': np.mean(cov_sd[oi, g]),
                'coverage_hcp':        np.mean(cov_hcp[g]),
                'coverage_pool':       np.mean(cov_pool[g]),
                'coverage_sub':        np.mean(cov_sub[g]),
                'coverage_rep':        np.mean(cov_rep[g]),
                'width_donor_hcp_randomized': np.nanmedian(wid_dr[oi, g]),
                'width_donor_hcp_derandomized': np.nanmedian(wid_dd[oi, g]),
                'width_sample_hcp_randomized': np.nanmedian(wid_sr[oi, g]),
                'width_sample_hcp_derandomized': np.nanmedian(wid_sd[oi, g]),
                'width_hcp':           np.nanmedian(wid_hcp[g]),
                'width_pool':          np.nanmedian(wid_pool[g]),
                'width_sub':           np.nanmedian(wid_sub[g]),
                'width_rep':           np.nanmedian(wid_rep[g]),
                'infinite_donor_hcp_randomized': int(inf_dr[oi, g]),
                'infinite_donor_hcp_derandomized': int(inf_dd[oi, g]),
                'infinite_sample_hcp_randomized': int(inf_sr[oi, g]),
                'infinite_sample_hcp_derandomized': int(inf_sd[oi, g]),
                'infinite_hcp':        inf_hcp,
                'infinite_pool':       inf_pool,
                'infinite_sub':        inf_sub,
                'infinite_rep':        inf_rep,
            })
    return pd.DataFrame(rows)


def run_experiments_conditional(
        number_experiments, number_groups_k, lambda_Poisson,
        dgp_specification, o_values, target_index, u_grid,
        alpha=0.1, number_subsampling_repetitions=50,
        alpha_selection=0.1, number_test_groups=50,
        mu_method_baseline=None, mu_method_hcp=None,
        show_progress=True):
    """
    Run ``number_experiments`` U-conditional-coverage experiments and combine.

    Each experiment generates an independent calibration dataset and
    evaluates coverage for test groups whose group covariate U[0] is
    fixed to each value in ``u_grid`` (with U[1:] = 0.5).

    Returns
    -------
    pd.DataFrame
        One row per (experiment, o_observed, u_grid_val).
    """
    if show_progress:
        print(f"Running {number_experiments} U-conditional experiments …")

    results_list = []
    for e in range(number_experiments):
        if show_progress and (e + 1) % 5 == 0:
            print(f"  Completed {e + 1}/{number_experiments} experiments")
        res = run_one_experiment_conditional(
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_values=o_values,
            target_index=target_index,
            u_grid=u_grid,
            alpha=alpha,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_method_baseline,
            mu_method_hcp=mu_method_hcp,
        )
        res['experiment'] = e + 1
        results_list.append(res)

    combined = pd.concat(results_list, ignore_index=True)
    cols = ['experiment'] + [c for c in combined.columns if c != 'experiment']
    return combined[cols]
