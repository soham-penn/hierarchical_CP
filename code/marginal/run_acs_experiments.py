"""
ACS coverage experiments (fixed ACS snapshot, no row bootstrap).

marginal:
  Each replicate: draw n_calib_per_stratum PUMAs from each of strata 1–4 (calibration),
  then average coverage over every other eligible target PUMA (index 20). No bootstrap.

conditional:
  Same as marginal but calibration PUMAs are fixed once (group_selection_seed).

marginal_one_target:
  Each replicate: resample calibration + one random target PUMA; one coverage indicator.
  (Older Option B; not the same as conditional/marginal above.)

uniform_one_target:
  Each replicate: uniformly draw n_puma_groups calibration PUMAs from all eligible PUMAs,
  then uniformly draw one target PUMA from the remaining eligible PUMAs. No bootstrap.

marginal_uniform:
  Each replicate: uniformly draw n_puma_groups calibration PUMAs from all eligible PUMAs
  (no BA+ strata), then average coverage over every other eligible target PUMA (index 20).

Note: repeated_experiments_stratified_acs.py uses bootstrap; these true-marginal scripts do not.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import math
import json
import pickle
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPO_ROOT / "real_data"
ACS_DIR = REAL_DATA_DIR / "acs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REAL_DATA_DIR))

from code.paths import PLOTS_MARGINAL, RESULTS_MARGINAL
from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs

from methods.mu_methods import (
    create_mu_method_ols_global_only,
    create_mu_method_ols_offset,
    create_mu_method_ols_residual_correction,
    create_mu_method_pretrained_global,
    create_mu_method_pretrained_offset,
    create_mu_method_random_forest_global_only,
    create_mu_method_random_forest_offset,
    create_mu_method_random_forest_residual_correction,
    train_ding_2021_xgb_regressor,
)

RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
PREDICTOR_CHOICES = ("ols", "rf", "ding_xgb")
REPLICATE_CACHE_VERSION = 2
CACHE_CONFIG_KEYS = (
    "alpha",
    "alpha_selection",
    "seed",
    "quantile_mode",
    "quantile_base_seed",
    "outcome_scale",
    "within_group",
    "within_group_mode",
    "predictor",
    "o_values",
    "n_repeated",
    "permute_rows",
    "design",
    "score_type",
    "stdcp_score_type",
    "stdcp_center",
)
SCORE_TYPE_CHOICES = ("absolute", "studentized")

from methods.donor_hcp import get_hcp_train_cal_split
from methods.donor_hcp import compute_donor_hcp_randomized_interval
from methods.sample_hcp import (
    compute_sample_hcp_randomized_interval,
)
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius,
)
from scores import absolute_residual_score, conformal_threshold, make_quantile_seed
from methods.nonconformity import (
    absolute_or_studentized_baseline_scores,
    fit_score_aux,
    get_score_type,
    interval_from_threshold,
)

# Method lists
BASELINE_METHODS = ['HCP', 'Pooling', 'Subsampling', 'Repeated']
HCP_METHODS = ['Donor-HCP', 'S-HCP']
STD_CP_METHODS = ['Std-CP']
METHODS = BASELINE_METHODS + HCP_METHODS + STD_CP_METHODS

# Fixed prediction target (default: 21st individual, index 20). History size o uses indices 0..o-1.
TARGET_INDEX = 20
MIN_TARGET_PUMA_SIZE = TARGET_INDEX + 1  # 21


def _set_target_index(target_index: int) -> None:
    """Override module target index (CLI --target_index)."""
    global TARGET_INDEX, MIN_TARGET_PUMA_SIZE
    TARGET_INDEX = int(target_index)
    MIN_TARGET_PUMA_SIZE = TARGET_INDEX + 1


def _result_dir_name(
    *, permuted: bool, alpha_str: str, target_index: int, predictor: str = "ols",
    within_group: bool = True, outcome_scale: str = "log1p",
    within_group_mode: str = "mean", drop_top_income_fraction: float = 0.0,
    yoep_min_year: int = 2012, min_income: float | None = 10000.0,
    stdcp_center: str = "local", score_type: str = "absolute",
    stdcp_score_type: str | None = None,
) -> str:
    prefix = "true_marginal_permuted" if permuted else "true_marginal"
    if predictor != "ols":
        prefix = f"{prefix}_{predictor}"
    if not within_group:
        prefix = f"{prefix}_no_within"
    if outcome_scale == "income":
        if predictor == "ols":
            prefix = f"{prefix}_ols"
        else:
            prefix = f"{prefix}_income"
        if drop_top_income_fraction > 0:
            prefix = f"{prefix}_trim{int(round(drop_top_income_fraction * 100))}"
        elif min_income is None or min_income <= 0:
            prefix = f"{prefix}_notrim"
        if within_group and within_group_mode == "correction":
            prefix = f"{prefix}_corr"
    if target_index != 20:
        prefix = f"{prefix}_t{target_index}"
    if int(yoep_min_year) != 2012:
        prefix = f"{prefix}_yoep{int(yoep_min_year)}"
    if str(stdcp_center).lower() == "global":
        prefix = f"{prefix}_stdcpglobal"
    st = str(score_type).lower()
    sst = str(stdcp_score_type if stdcp_score_type is not None else score_type).lower()
    if st == "studentized":
        prefix = f"{prefix}_studentized"
    elif sst == "studentized":
        # GHCP absolute + Std-CP studentized
        prefix = f"{prefix}_stdcpstud"
    return f"{prefix}_{alpha_str}"


OUTCOME_SCALE_CHOICES = ("log1p", "income")
WITHIN_GROUP_MODE_CHOICES = ("mean", "correction")
LOCAL_ADJUSTMENT_CLIP = {"log1p": 0.5, "income": 50_000.0}
PLOTS_ACS_ROOT = PLOTS_MARGINAL / "acs"


def _acs_paper_suite_subdir(
    *,
    predictor: str,
    outcome_scale: str,
    within_group: bool,
    within_group_mode: str = "mean",
    drop_top_income_fraction: float = 0.0,
    yoep_min_year: int = 2012,
    min_income: float | None = 10000.0,
    design: str = "marginal",
) -> str | None:
    """Subfolder under paper-results/acs for income-scale paper artifacts."""
    if outcome_scale != "income":
        return None
    if design == "marginal_uniform":
        return "unstratified"
    if predictor == "ols":
        if (
            int(yoep_min_year) == 2012
            and drop_top_income_fraction <= 0
            and within_group
            and str(within_group_mode).lower() == "mean"
        ):
            return "ols/legacy_mean"
        if int(yoep_min_year) != 2012:
            no_income_filters = drop_top_income_fraction <= 0 and (
                min_income is None or min_income <= 0
            )
            if no_income_filters:
                if within_group and str(within_group_mode).lower() == "mean":
                    return "ols_2000_mean"
                return "ols_yoep2000_notrim"
            return "ols_yoep2000"
        return "ols"
    if predictor == "rf":
        no_income_filters = drop_top_income_fraction <= 0 and (
            min_income is None or min_income <= 0
        )
        if (
            int(yoep_min_year) == 2012
            and no_income_filters
            and within_group
            and str(within_group_mode).lower() == "mean"
        ):
            return "rf_2012_mean"
        if (
            int(yoep_min_year) == 2012
            and no_income_filters
            and within_group
            and str(within_group_mode).lower() == "correction"
        ):
            return "rf_2012_residual"
        if int(yoep_min_year) != 2012:
            if no_income_filters:
                # Paper ACS suite lives at paper-results/acs/results (not acs/rf/).
                return "."
        if (
            within_group
            and drop_top_income_fraction > 0
            and str(within_group_mode).lower() == "mean"
        ):
            return "rf/trim2_mean"
        return "rf" if within_group else "rf/no_within"
    if predictor == "ding_xgb":
        return "xgboost" if within_group else "xgboost/no_within"
    return None


def _acs_output_dir(
    *,
    permuted: bool,
    alpha_str: str,
    target_index: int,
    predictor: str,
    within_group: bool,
    outcome_scale: str,
    within_group_mode: str,
    drop_top_income_fraction: float,
    yoep_min_year: int = 2012,
    min_income: float | None = 10000.0,
    design: str = "marginal",
    stdcp_center: str = "local",
    score_type: str = "absolute",
    stdcp_score_type: str | None = None,
) -> Path:
    """Write ACS artifacts under paper-results/acs/{suite}/results/.

    Paper RF YOEP≥2000 no-trim runs use suite "." → paper-results/acs/results/.
    """
    run_tag = _result_dir_name(
        permuted=permuted,
        alpha_str=alpha_str,
        target_index=target_index,
        predictor=predictor,
        within_group=within_group,
        outcome_scale=outcome_scale,
        within_group_mode=within_group_mode,
        drop_top_income_fraction=drop_top_income_fraction,
        yoep_min_year=yoep_min_year,
        min_income=min_income,
        stdcp_center=stdcp_center,
        score_type=score_type,
        stdcp_score_type=stdcp_score_type,
    )
    # Fully studentized ACS runs land in paper-results/acs/studentized/
    if str(score_type).lower() == "studentized" and outcome_scale == "income":
        return PLOTS_ACS_ROOT / "studentized" / "results" / run_tag
    suite = _acs_paper_suite_subdir(
        predictor=predictor,
        outcome_scale=outcome_scale,
        within_group=within_group,
        within_group_mode=within_group_mode,
        drop_top_income_fraction=drop_top_income_fraction,
        yoep_min_year=yoep_min_year,
        min_income=min_income,
        design=design,
    )
    if suite is not None:
        if suite == "ols/legacy_mean":
            run_tag = f"true_marginal_permuted_income_{alpha_str}"
        return PLOTS_ACS_ROOT / suite / "results" / run_tag
    return RESULTS_MARGINAL / "acs" / run_tag


def _outcome_transform(outcome_scale: str):
    if outcome_scale == "income":
        return lambda x: np.asarray(x, dtype=float)
    if outcome_scale == "log1p":
        return np.log1p
    raise ValueError(f"Unknown outcome_scale {outcome_scale!r}; expected {OUTCOME_SCALE_CHOICES}")

# ==================== Helper functions ====================

def _interval_from_radius(mu_hat, T):
    return (mu_hat - T, mu_hat + T)

def _covered(interval, true_y):
    return 1.0 if interval[0] <= true_y <= interval[1] else 0.0

def _width(interval):
    return interval[1] - interval[0]

def _width_income_from_interval(interval, outcome_scale: str = "log1p"):
    """Map a finite prediction interval to width on the income scale."""
    lower, upper = interval
    if outcome_scale == "income":
        return float(upper - lower)
    income_lower = np.expm1(lower)
    income_upper = np.expm1(upper)
    return float(income_upper - income_lower)


def _width_income_from_log1p_interval(interval):
    """Backward-compatible alias for log1p outcome scale."""
    return _width_income_from_interval(interval, outcome_scale="log1p")


def _split_conformal_radius(abs_residuals, alpha, quantile_mode="randomized",
                            random_seed=None, rng=None):
    """Finite-sample split-conformal radius from calibration residuals."""
    scores = np.asarray(abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    if quantile_mode == "randomized":
        # Append +∞ so randomized split CP targets exact 1-α coverage
        # (same augmentation as DGP Std-CP / donor-HCP test-only split).
        scores = np.append(scores, np.inf)
        weights = np.ones(len(scores), dtype=float) / len(scores)
        return conformal_threshold(
            scores=scores,
            weights=weights,
            alpha=alpha,
            quantile_mode="randomized",
            random_seed=random_seed,
            rng=rng,
            return_info=False,
        )
    k = int(np.ceil((n + 1) * (1 - alpha)))
    # Finite-sample split CP: if the required order statistic exceeds n, the
    # only valid radius is +∞ (do not silently fall back to the max).
    if k > n:
        return np.inf
    k = max(1, k)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_std_cp_interval(x_hist, y_hist, x_target, alpha, rng,
                             quantile_mode="randomized", quantile_random_seed=None,
                             ntree=None, nodesize=None, rf_random_state=None,
                             score_type="absolute",
                             return_intermediates=False):
    """
    Standard split CP within one target PUMA using only its first o history rows.

    Fit a Random Forest on a random half of indices 0..o-1, calibrate on the
    other half, and predict x_target. Other PUMAs are not used.
    At o=0 (or fewer than 2 history rows) the interval is infinite.

    score_type:
      absolute     — |Y-μ|
      studentized  — |Y-μ|/σ with local RF-σ on training absolute residuals

    If return_intermediates, returns (interval, intermediates) with everything
    needed to rebuild absolute or studentized Std-CP without refitting μ.
    """
    from sklearn.ensemble import RandomForestRegressor

    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)

    n_hist = len(y_hist)
    n_train = n_hist // 2
    n_cal = n_hist - n_train
    if n_train < 1 or n_cal < 1:
        empty = (-np.inf, np.inf)
        return (empty, None) if return_intermediates else empty

    perm = rng.permutation(n_hist)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    x_train = x_hist[train_idx]
    y_train = y_hist[train_idx]
    x_cal = x_hist[cal_idx]
    y_cal = y_hist[cal_idx]

    n_estimators = int(RF_NTREE if ntree is None else ntree)
    min_leaf = int(RF_NODESIZE if nodesize is None else nodesize)
    # With tiny train sizes (e.g. o=10 => n_train=5), nodesize may equal n_train;
    # RF then predicts the train mean (no splits), which is the intended baseline.
    min_leaf = max(1, min(min_leaf, len(y_train)))
    rs = int(RF_RANDOM_STATE if rf_random_state is None else rf_random_state)
    st = str(score_type).lower().replace("-", "_")
    if st in ("standardized", "studentised", "studentized", "std", "sigma"):
        st = "studentized"
    else:
        st = "absolute"

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        random_state=rs,
        n_jobs=1,
    )
    rf.fit(x_train, y_train)
    y_cal_pred = rf.predict(x_cal)
    x_target_arr = np.asarray(x_target, dtype=float).reshape(1, -1)
    y_hat = float(rf.predict(x_target_arr)[0])
    abs_tr = np.abs(y_train - rf.predict(x_train))

    sigma_cal = None
    sigma_target = None
    if st == "studentized":
        rf_s = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            random_state=rs + 17,
            n_jobs=1,
        )
        rf_s.fit(x_train, abs_tr)
        sigma_cal = np.clip(rf_s.predict(x_cal), 1e-6, 1e12)
        scores = np.abs(y_cal - y_cal_pred) / sigma_cal
        radius = _split_conformal_radius(
            scores,
            alpha,
            quantile_mode=quantile_mode,
            random_seed=quantile_random_seed,
            rng=rng,
        )
        if not np.isfinite(radius):
            interval = (-np.inf, np.inf)
        else:
            sigma_target = float(np.clip(rf_s.predict(x_target_arr)[0], 1e-6, 1e12))
            interval = (y_hat - radius * sigma_target, y_hat + radius * sigma_target)
    else:
        radius = _split_conformal_radius(
            np.abs(y_cal - y_cal_pred),
            alpha,
            quantile_mode=quantile_mode,
            random_seed=quantile_random_seed,
            rng=rng,
        )
        interval = _interval_from_radius(y_hat, radius)

    if not return_intermediates:
        return interval

    inter = {
        "center_mode": "local",
        "score_type_used": st,
        "train_idx": np.asarray(train_idx, dtype=int).copy(),
        "cal_idx": np.asarray(cal_idx, dtype=int).copy(),
        "x_hist": np.asarray(x_hist, dtype=float).copy(),
        "y_hist": np.asarray(y_hist, dtype=float).copy(),
        "x_target": np.asarray(x_target_arr, dtype=float).ravel().copy(),
        "mu_cal": np.asarray(y_cal_pred, dtype=float).copy(),
        "mu_target": float(y_hat),
        "abs_train": np.asarray(abs_tr, dtype=float).copy(),
        "sigma_cal": None if sigma_cal is None else np.asarray(sigma_cal, dtype=float).copy(),
        "sigma_target": sigma_target,
    }
    return interval, inter


def _compute_std_cp_interval_global_center(
    x_hist, y_hist, x_target, alpha, *,
    mu_hat_hist, mu_hat_target,
    quantile_mode="randomized", quantile_random_seed=None, rng=None,
    score_aux=None,
):
    """
    Inductive Std-CP with a frozen global predictor as the center.

    Uses all o history residuals for the conformal radius (no local refit /
    no half-split). With score_aux studentized, residuals are |Y-μ|/σ and the
    interval is μ ± q·σ(x_target). At o=0 the interval is infinite.
    """
    from methods.nonconformity import predict_scale

    y_hist = np.asarray(y_hist, dtype=float).ravel()
    mu_hat_hist = np.asarray(mu_hat_hist, dtype=float).ravel()
    if len(y_hist) < 1 or len(mu_hat_hist) != len(y_hist):
        return (-np.inf, np.inf)

    st = (score_aux or {}).get("score_type", "absolute")
    x_hist = np.asarray(x_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)
    if st == "studentized" and score_aux is not None:
        # U is unused in ACS (zeros); pass zeros matching feature convention.
        u0 = np.zeros(1, dtype=float)
        sig_hist = np.array(
            [predict_scale(score_aux, x_hist[i], u0) for i in range(len(y_hist))],
            dtype=float,
        )
        scores = np.abs(y_hist - mu_hat_hist) / np.clip(sig_hist, 1e-6, 1e12)
    else:
        scores = np.abs(y_hist - mu_hat_hist)

    radius = _split_conformal_radius(
        scores,
        alpha,
        quantile_mode=quantile_mode,
        random_seed=quantile_random_seed,
        rng=rng,
    )
    if not np.isfinite(radius):
        return (-np.inf, np.inf)
    if st == "studentized" and score_aux is not None:
        u0 = np.zeros(1, dtype=float)
        s_t = float(predict_scale(score_aux, x_target, u0))
        return (float(mu_hat_target) - radius * s_t, float(mu_hat_target) + radius * s_t)
    return _interval_from_radius(float(mu_hat_target), radius)


def compute_global_pooled_income_interval(df: pd.DataFrame, alpha: float):
    """One global prediction set: empirical income quantiles over all ACS rows."""
    incomes = df["income"].astype(float).values
    q_lo, q_hi = np.quantile(incomes, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(q_lo), float(q_hi)


def alpha_to_tag(alpha: float) -> str:
    """Stable filename tag, e.g. 0.1 -> alpha10 and 0.075 -> alpha07p5."""
    pct = f"{100.0 * float(alpha):.6g}".replace(".", "p")
    if "p" not in pct and len(pct) < 2:
        pct = pct.zfill(2)
    return f"alpha{pct}"


def _interval_to_income_bounds(interval, outcome_scale: str = "log1p"):
    lo, hi = interval
    if outcome_scale == "income":
        return (
            float(lo) if np.isfinite(lo) else -np.inf,
            float(hi) if np.isfinite(hi) else np.inf,
        )
    return (np.expm1(lo) if np.isfinite(lo) else -np.inf,
            np.expm1(hi) if np.isfinite(hi) else np.inf)


def _result_record(interval, true_y, outcome_scale: str = "log1p"):
    """Standard per-method result dict with outcome and income endpoints."""
    lo_inc, hi_inc = _interval_to_income_bounds(interval, outcome_scale=outcome_scale)
    return {
        "coverage": _covered(interval, true_y),
        "width": _width(interval),
        "width_income": _width_income_from_interval(interval, outcome_scale=outcome_scale),
        "lower": interval[0],
        "upper": interval[1],
        "lower_income": lo_inc,
        "upper_income": hi_inc,
    }


def _without_within_group_training(mu_method):
    """Return a mu_method wrapper that ignores within-group history."""
    base_predict_global = mu_method["predict_global"]
    out = dict(mu_method)

    def fit_group_adjustment(model_global, u_group_vector,
                             Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return base_predict_global(model_global, x_vector, u_group_vector)

    out["fit_group_adjustment"] = fit_group_adjustment
    out["predict_group_mu"] = predict_group_mu
    return out


def _attach_score_type(mu_method: dict, config: dict) -> dict:
    """Stamp score_type (+ RF hyperparams used by fit_score_aux) onto a mu_method."""
    out = dict(mu_method)
    st = str(config.get("score_type", "absolute")).lower()
    if st not in SCORE_TYPE_CHOICES:
        st = "absolute"
    out["score_type"] = st
    out.setdefault("rf_ntree", RF_NTREE)
    out.setdefault("rf_nodesize", RF_NODESIZE)
    out.setdefault("rf_random_state", RF_RANDOM_STATE)
    return out


def _make_mu_baseline(config):
    """Global predictor for HCP baselines (OLS, RF, or Ding pre-trained XGB)."""
    predictor = str(config.get("predictor", "ols")).lower()
    if predictor == "ding_xgb":
        mu = create_mu_method_pretrained_global(config["pretrained_model"], tau=0)
    elif predictor == "rf":
        mu = create_mu_method_random_forest_global_only(
            ntree=RF_NTREE,
            nodesize=RF_NODESIZE,
            random_state=RF_RANDOM_STATE,
        )
    elif predictor == "ols":
        mu = create_mu_method_ols_global_only()
    else:
        raise ValueError(f"Unknown predictor {predictor!r}; expected one of {PREDICTOR_CHOICES}")
    return _attach_score_type(mu, config)


def _make_mu_hcp(config):
    """Mu method for HCP-style methods, optionally disabling within-group training."""
    predictor = str(config.get("predictor", "ols")).lower()
    outcome_scale = str(config.get("outcome_scale", "log1p"))
    within_group_mode = str(config.get("within_group_mode", "mean")).lower()
    local_clip = LOCAL_ADJUSTMENT_CLIP.get(outcome_scale, LOCAL_ADJUSTMENT_CLIP["log1p"])
    if predictor == "ding_xgb":
        mu_hcp = create_mu_method_pretrained_offset(
            config["pretrained_model"],
            local_adjustment_clip=local_clip,
        )
    elif predictor == "rf" and within_group_mode == "correction":
        mu_hcp = create_mu_method_random_forest_residual_correction(
            ntree=RF_NTREE,
            nodesize=RF_NODESIZE,
            random_state=RF_RANDOM_STATE,
            local_adjustment_clip=local_clip,
        )
    elif predictor == "rf":
        mu_hcp = create_mu_method_random_forest_offset(
            ntree=RF_NTREE,
            nodesize=RF_NODESIZE,
            random_state=RF_RANDOM_STATE,
            c=1.0,  # Eq. (4): w_g = |Strain| / (|Strain| + τ)
        )
    elif predictor == "ols" and within_group_mode == "correction":
        mu_hcp = create_mu_method_ols_residual_correction(
            local_adjustment_clip=local_clip,
        )
    elif predictor == "ols":
        mu_hcp = create_mu_method_ols_offset()
    else:
        raise ValueError(f"Unknown predictor {predictor!r}; expected one of {PREDICTOR_CHOICES}")
    if not bool(config.get("within_group", True)):
        mu_hcp = _without_within_group_training(mu_hcp)
    return _attach_score_type(mu_hcp, config)


def _fit_baseline_score_aux(mu_baseline, model_baseline, U_calibration_full,
                            Z_calibration_full, train_idx, alpha):
    """Auxiliary models for baseline HCP scores (RF-σ when studentized)."""
    return fit_score_aux(
        U_all=U_calibration_full,
        Z_all=Z_calibration_full,
        group_index_vector=train_idx,
        mu_method=mu_baseline,
        global_model=model_baseline,
        alpha=alpha,
    )


def _baseline_group_scores(Zj, Uj, muj, aux):
    """Absolute or studentized residual scores for one calibration group."""
    yj = np.asarray([z["Y"] for z in Zj], dtype=float)
    muj = np.asarray(muj, dtype=float)
    if get_score_type({"score_type": (aux or {}).get("score_type", "absolute")}) != "studentized":
        return absolute_residual_score(yj, muj)
    from methods.nonconformity import predict_scale
    sigma = np.array([predict_scale(aux, z["X"], Uj) for z in Zj], dtype=float)
    return absolute_or_studentized_baseline_scores(yj, muj, sigma)


def _baseline_interval(mu_hat, T, x_target, u_target, aux):
    """Map baseline conformal radius to an interval (respects studentized σ)."""
    return interval_from_threshold(T, mu_hat, x_target, u_target, aux)


def _tau_override(config):
    """Use tau=0 when within-group training is disabled."""
    return None if bool(config.get("within_group", True)) else 0


def compute_group_share_baplus(df, group_col):
    """Compute share of BA+ for each group for stratification."""
    out = {}
    for grp in df[group_col].unique():
        grp_df = df[df[group_col] == grp]
        out[grp] = float((grp_df['educ_level'] == 'BAplus').mean())
    return out


def _build_share_baplus_strata(eligible_groups, group_share_baplus, n_strata=5):
    """Build quantile strata from the PUMA-level share of BA+."""
    shares = pd.Series({g: group_share_baplus[g] for g in eligible_groups}, dtype=float)
    bins = pd.qcut(shares, q=int(n_strata), duplicates='drop')

    strata = {}
    for b in bins.cat.categories:
        members = shares.index[bins == b].tolist()
        if members:
            strata[str(b)] = members

    if len(strata) == 0:
        strata = {"all": list(eligible_groups)}

    return strata


def stratified_sample_groups(eligible_groups, strata, selection_seed, n_groups_to_select):
    """Use stratified sampling to select non-target groups."""
    stratum_names = sorted(list(strata.keys()))
    n_strata = len(stratum_names)
    if n_strata == 0:
        raise ValueError("No strata available for stratified sampling")

    if n_groups_to_select > len(eligible_groups):
        raise ValueError(
            f"Cannot select {n_groups_to_select} groups from {len(eligible_groups)} eligible groups"
        )

    rng = np.random.default_rng(selection_seed)

    # Balanced allocation across strata
    base = n_groups_to_select // n_strata
    rem = n_groups_to_select % n_strata

    capacity = {k: len(strata[k]) for k in stratum_names}
    alloc = {k: min(base, capacity[k]) for k in stratum_names}
    assigned = sum(alloc.values())

    # Distribute leftover picks
    while assigned < n_groups_to_select:
        order = sorted(
            stratum_names,
            key=lambda k: (capacity[k] - alloc[k], rng.random()),
            reverse=True,
        )
        progressed = False
        for k in order:
            if alloc[k] < capacity[k]:
                alloc[k] += 1
                assigned += 1
                progressed = True
                if assigned >= n_groups_to_select:
                    break
        if not progressed:
            break

    selected = []
    for k in stratum_names:
        take = int(alloc[k])
        if take <= 0:
            continue
        chosen = rng.choice(np.asarray(strata[k]), size=take, replace=False).tolist()
        selected.extend(chosen)

    if len(selected) != n_groups_to_select:
        raise ValueError(
            f"Balanced allocation produced {len(selected)} groups, expected {n_groups_to_select}"
        )

    return sorted(selected)


def sample_calibration_fixed_per_stratum(
    strata,
    group_counts,
    selection_seed,
    o_values,
    n_calib_per_stratum=4,
    n_calib_strata=4,
    min_calib_puma_size=None,
):
    """
    Draw a fixed calibration set: n_calib_per_stratum PUMAs from each of strata 1..n_calib_strata.
    Returns sorted calib_groups (same draw for all replicates when selection_seed is fixed).
    """
    stratum_names = sorted(list(strata.keys()))
    if len(stratum_names) < n_calib_strata:
        raise ValueError(
            f"Need at least {n_calib_strata} strata, got {len(stratum_names)}"
        )

    if min_calib_puma_size is None:
        min_calib_puma_size = max(int(o) for o in o_values) + 1
    min_calib_puma_size = int(min_calib_puma_size)
    calib_strata = stratum_names[:n_calib_strata]
    rng = np.random.default_rng(selection_seed)

    calib_groups = []
    for k in calib_strata:
        members = [
            int(g)
            for g in strata[k]
            if int(group_counts.get(g, 0)) >= min_calib_puma_size
        ]
        if len(members) < n_calib_per_stratum:
            raise ValueError(
                f"Stratum {k}: need {n_calib_per_stratum} PUMAs with size >= {min_calib_puma_size}, "
                f"got {len(members)}"
            )
        chosen = rng.choice(np.asarray(members), size=n_calib_per_stratum, replace=False)
        calib_groups.extend(int(g) for g in chosen)

    return sorted(calib_groups)


def sample_calibration_uniform(
    eligible_groups,
    group_counts,
    selection_seed,
    n_calib_groups=21,
    o_values=None,
    min_calib_puma_size=None,
):
    """
    Draw n_calib_groups calibration PUMAs uniformly without replacement (no stratification).

    PUMAs must have size >= min_calib_puma_size (default max(o)+1).
    """
    if min_calib_puma_size is None:
        if o_values is None:
            min_calib_puma_size = MIN_TARGET_PUMA_SIZE
        else:
            min_calib_puma_size = max(int(o) for o in o_values) + 1
    min_calib_puma_size = int(min_calib_puma_size)

    pool = [
        int(g)
        for g in eligible_groups
        if int(group_counts.get(g, 0)) >= min_calib_puma_size
    ]
    n_calib_groups = int(n_calib_groups)
    if len(pool) < n_calib_groups:
        raise ValueError(
            f"Need {n_calib_groups} calibration PUMAs with size >= {min_calib_puma_size}, "
            f"got {len(pool)}"
        )

    rng = np.random.default_rng(selection_seed)
    chosen = rng.choice(np.asarray(pool, dtype=int), size=n_calib_groups, replace=False)
    return sorted(int(g) for g in chosen)


def _load_group_data_static(df, X, group_col, groups, truncate_n=None, rng=None):
    """Load observed rows per PUMA.

    If rng is provided, rows are permuted within each PUMA without replacement.
    This keeps the ACS snapshot fixed while making the target stream order random
    across true-marginal replicates.
    """
    group_data = {}
    for grp in groups:
        grp_idx = np.where((df[group_col] == grp).values)[0]
        if rng is not None:
            grp_idx = rng.permutation(grp_idx)
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,
            'income': df.iloc[grp_idx]['income'].values.astype(float),
        }
        if truncate_n is not None:
            n_cap = int(truncate_n)
            for key in ('X', 'Y', 'income'):
                group_data[grp][key] = group_data[grp][key][:n_cap]
    return group_data


def _serialize_group_data(group_data: dict) -> dict:
    """Convert group_data arrays to plain numpy for pickle caching."""
    return {
        int(grp): {
            "X": np.asarray(data["X"], dtype=float),
            "Y": np.asarray(data["Y"], dtype=float),
            "income": np.asarray(data["income"], dtype=float),
        }
        for grp, data in group_data.items()
    }


def _deserialize_group_data(serialized: dict) -> dict:
    return {
        int(grp): {
            "X": np.asarray(data["X"], dtype=float),
            "Y": np.asarray(data["Y"], dtype=float),
            "income": np.asarray(data["income"], dtype=float),
        }
        for grp, data in serialized.items()
    }


def _build_z_calibration(group_data: dict, calib_groups: list) -> list:
    return [
        [
            {"X": group_data[grp]["X"][i], "Y": group_data[grp]["Y"][i]}
            for i in range(len(group_data[grp]["Y"]))
        ]
        for grp in calib_groups
    ]


def _cache_config_subset(config: dict) -> dict:
    return {k: config[k] for k in CACHE_CONFIG_KEYS if k in config}


def _replicate_cache_path(cache_dir: Path, replicate_idx: int) -> Path:
    return cache_dir / f"rep_{int(replicate_idx):05d}.pkl"


def save_replicate_cache(cache_dir: Path, replicate_idx: int, cache: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _replicate_cache_path(cache_dir, replicate_idx).open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_replicate_cache(cache_dir: Path, replicate_idx: int) -> dict:
    with _replicate_cache_path(cache_dir, replicate_idx).open("rb") as f:
        return pickle.load(f)


def write_cache_manifest(
    cache_dir: Path,
    config: dict,
    *,
    B: int,
    global_pooled: dict | None,
    cohort: dict | None = None,
) -> None:
    manifest = {
        "version": REPLICATE_CACHE_VERSION,
        "B": int(B),
        "config": _cache_config_subset(config),
        "global_pooled": global_pooled,
        "cohort": cohort or {},
        "methods": METHODS,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _compute_hcp_over_targets(
    *,
    group_data: dict,
    calib_groups: list,
    test_groups: list,
    o_values: list,
    config: dict,
    replicate_idx: int,
) -> dict:
    """Evaluate Donor-HCP and S-HCP averaged over target PUMAs."""
    alpha = config["alpha"]
    alpha_sel = config.get("alpha_selection", 0.5)
    quantile_mode = config.get("quantile_mode", "deterministic")
    quantile_base_seed = config.get("quantile_base_seed", config["seed"])
    outcome_scale = str(config.get("outcome_scale", "log1p"))

    Z_calibration_full = _build_z_calibration(group_data, calib_groups)
    U_calibration_full = np.zeros((len(calib_groups), 1))
    U_test = np.zeros((1, 1))
    mu_hcp = _make_mu_hcp(config)

    hcp_result = {m: {} for m in HCP_METHODS}
    for o in o_values:
        eligible_test = [
            grp for grp in test_groups
            if len(group_data[grp]["Y"]) > max(o, TARGET_INDEX)
        ]
        if len(eligible_test) == 0:
            nan_rec = {
                "coverage": np.nan,
                "width": np.nan,
                "width_income": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "lower_income": np.nan,
                "upper_income": np.nan,
            }
            for method in HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            continue

        hcp_cov = {m: [] for m in HCP_METHODS}
        hcp_wid = {m: [] for m in HCP_METHODS}
        hcp_wid_inc = {m: [] for m in HCP_METHODS}

        for test_group in eligible_test:
            target_index = TARGET_INDEX
            true_y = group_data[test_group]["Y"][target_index]
            Z_test_full = [
                {"X": group_data[test_group]["X"][i], "Y": group_data[test_group]["Y"][i]}
                for i in range(len(group_data[test_group]["Y"]))
            ]

            try:
                dhcp_seed = (
                    config["seed"]
                    + (replicate_idx + 1) * 1009
                    + (o + 1) * 131
                    + 17
                )
                res_dhcp = compute_donor_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test,
                    Z_test=Z_test_full,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                    test_index_target=target_index,
                    tau_override=_tau_override(config),
                    random_seed=dhcp_seed,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "donor_hcp"
                    ),
                )
                int_dhcp = res_dhcp["interval"]
            except Exception:
                int_dhcp = (-np.inf, np.inf)

            hcp_cov["Donor-HCP"].append(_covered(int_dhcp, true_y))
            hcp_wid["Donor-HCP"].append(_width(int_dhcp))
            hcp_wid_inc["Donor-HCP"].append(
                _width_income_from_interval(int_dhcp, outcome_scale=outcome_scale)
            )

            try:
                res_shcp = compute_sample_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test,
                    Z_test=Z_test_full,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                    test_index_target=target_index,
                    tau_override=_tau_override(config),
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "sample_hcp"
                    ),
                )
                int_shcp = res_shcp["interval"]
            except Exception:
                int_shcp = (-np.inf, np.inf)

            hcp_cov["S-HCP"].append(_covered(int_shcp, true_y))
            hcp_wid["S-HCP"].append(_width(int_shcp))
            hcp_wid_inc["S-HCP"].append(
                _width_income_from_interval(int_shcp, outcome_scale=outcome_scale)
            )

        for method in HCP_METHODS:
            hcp_result[method][o] = {
                "coverage": float(np.mean(hcp_cov[method])),
                "width": float(np.nanmean(hcp_wid[method])),
                "width_income": float(np.nanmean(hcp_wid_inc[method])),
                "lower": np.nan,
                "upper": np.nan,
                "lower_income": np.nan,
                "upper_income": np.nan,
            }
    return hcp_result


def recompute_replicate_from_cache(cache: dict, *, within_group_mode: str) -> dict:
    """Re-evaluate Donor-HCP / S-HCP from a saved replicate cache."""
    config = dict(cache["config"])
    config["within_group_mode"] = within_group_mode
    group_data = _deserialize_group_data(cache["group_data"])
    hcp_result = _compute_hcp_over_targets(
        group_data=group_data,
        calib_groups=list(cache["calib_groups"]),
        test_groups=list(cache["test_groups"]),
        o_values=list(config["o_values"]),
        config=config,
        replicate_idx=int(cache["replicate_idx"]),
    )
    return {
        "baseline": cache["baseline"],
        "hcp": hcp_result,
        "stdcp": cache["stdcp"],
        "income_target": cache["income_target"],
    }


def _recompute_one_cache_worker(args: tuple) -> tuple[int, dict]:
    replicate_idx, cache_dir, within_group_mode = args
    cache = load_replicate_cache(cache_dir, replicate_idx)
    return replicate_idx, recompute_replicate_from_cache(
        cache, within_group_mode=within_group_mode
    )


def run_recompute_from_cache(
    cache_dir: Path,
    *,
    within_group_mode: str,
    B: int,
    o_values: list,
    n_workers: int = 1,
) -> dict:
    """Aggregate replicate-level results after swapping within-group mode."""
    n_o = len(o_values)
    n_m = len(METHODS)
    m_map = {m: i for i, m in enumerate(METHODS)}

    cov = np.full((B, n_m, n_o), np.nan)
    wid = np.full((B, n_m, n_o), np.nan)
    wid_income = np.full((B, n_m, n_o), np.nan)
    lower_log = np.full((B, n_m, n_o), np.nan)
    upper_log = np.full((B, n_m, n_o), np.nan)
    lower_income = np.full((B, n_m, n_o), np.nan)
    upper_income = np.full((B, n_m, n_o), np.nan)
    income_targets = np.full(B, np.nan)

    def _write_result(rep_idx: int, result: dict | None) -> None:
        if result is None:
            return
        income_targets[rep_idx] = result.get("income_target", np.nan)

        def _store(m_i, o_i, rec):
            cov[rep_idx, m_i, o_i] = rec["coverage"]
            wid[rep_idx, m_i, o_i] = rec["width"]
            wid_income[rep_idx, m_i, o_i] = rec["width_income"]
            lower_log[rep_idx, m_i, o_i] = rec["lower"]
            upper_log[rep_idx, m_i, o_i] = rec["upper"]
            lower_income[rep_idx, m_i, o_i] = rec["lower_income"]
            upper_income[rep_idx, m_i, o_i] = rec["upper_income"]

        for method in BASELINE_METHODS:
            if 0 in result["baseline"][method]:
                _store(m_map[method], 0, result["baseline"][method][0])

        for method in HCP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                if o in result["hcp"][method]:
                    _store(m_i, o_i, result["hcp"][method][o])

        for method in STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                if o in result["stdcp"][method]:
                    _store(m_i, o_i, result["stdcp"][method][o])

    worker_args = [(b, cache_dir, within_group_mode) for b in range(B)]
    if n_workers <= 1:
        for b in range(B):
            if (b + 1) % 100 == 0 or (b + 1) <= 10:
                print(f"    Recompute replicate {b + 1}/{B}")
            rep_idx, result = _recompute_one_cache_worker(worker_args[b])
            _write_result(rep_idx, result)
    else:
        completed = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for rep_idx, result in pool.imap_unordered(_recompute_one_cache_worker, worker_args):
                completed += 1
                if completed % 100 == 0 or completed <= 10 or completed == B:
                    print(f"    Recompute replicate {completed}/{B}")
                _write_result(rep_idx, result)

    return {
        "methods": METHODS,
        "o_values": o_values,
        "coverage": cov,
        "width": wid,
        "width_income": wid_income,
        "lower_log": lower_log,
        "upper_log": upper_log,
        "lower_income": lower_income,
        "upper_income": upper_income,
        "income_targets": income_targets,
    }


def _test_groups_from_calib(eligible_groups, calib_groups, group_counts,
                            min_target_size=MIN_TARGET_PUMA_SIZE):
    calib_set = set(calib_groups)
    return sorted(
        int(g) for g in eligible_groups
        if g not in calib_set and int(group_counts.get(g, 0)) >= min_target_size
    )


def strata_where_all_pumas_at_least(strata, group_counts, min_size):
    """Return sorted stratum keys whose every PUMA has at least min_size observations."""
    out = []
    for k in sorted(strata.keys()):
        if all(int(group_counts[g]) >= min_size for g in strata[k]):
            out.append(k)
    return out


def pick_target_stratum_key(strata, group_counts, target_stratum_index, min_target_size):
    """
  Pick target stratum: must have all PUMAs with size >= min_target_size.

    target_stratum_index: 1-based index among sorted strata (default 5), or -1 for
    the highest qualifying stratum.
    """
    qualifying = strata_where_all_pumas_at_least(strata, group_counts, min_target_size)
    if not qualifying:
        raise ValueError(
            f"No stratum has all PUMAs with size >= {min_target_size}. "
            f"Qualifying check failed on strata: {list(strata.keys())}"
        )
    names = sorted(strata.keys())
    if target_stratum_index == -1:
        return qualifying[-1]
    idx = int(target_stratum_index) - 1
    if idx < 0 or idx >= len(names):
        raise ValueError(f"target_stratum_index must be in 1..{len(names)}")
    key = names[idx]
    if key not in qualifying:
        raise ValueError(
            f"Stratum {target_stratum_index} ({key}) is not all >= {min_target_size}. "
            f"Qualifying strata: {qualifying}"
        )
    return key


def sample_calibration_and_target_symmetric(
    strata,
    group_counts,
    eligible_groups,
    selection_seed,
    o_values,
    n_calib_per_stratum=5,
    n_calib_strata=4,
    min_target_size=MIN_TARGET_PUMA_SIZE,
    min_calib_puma_size=None,
):
    """
    Option B: stratified calibration + uniform target from remaining eligible PUMAs.

    - Draw 5 PUMAs without replacement from each of strata 1..n_calib_strata (size > max o).
    - Draw 1 target PUMA uniformly from eligible PUMAs not in the calibration set
      (size >= min_target_size so TARGET_INDEX exists).
    - Target and calibration sets are disjoint by construction.

    Returns (calib_groups, test_group) or (None, None) if pools are too small.
    """
    stratum_names = sorted(list(strata.keys()))
    if len(stratum_names) < n_calib_strata:
        raise ValueError(
            f"Need at least {n_calib_strata} strata, got {len(stratum_names)}"
        )

    if min_calib_puma_size is None:
        min_calib_puma_size = max(int(o) for o in o_values) + 1
    min_calib_puma_size = int(min_calib_puma_size)
    calib_strata = stratum_names[:n_calib_strata]
    rng = np.random.default_rng(selection_seed)

    calib_groups = []
    for k in calib_strata:
        members = [
            int(g)
            for g in strata[k]
            if int(group_counts.get(g, 0)) >= min_calib_puma_size
        ]
        if len(members) < n_calib_per_stratum:
            return None, None
        chosen = rng.choice(np.asarray(members), size=n_calib_per_stratum, replace=False)
        calib_groups.extend(int(g) for g in chosen)

    calib_set = set(calib_groups)
    target_pool = [
        int(g)
        for g in eligible_groups
        if g not in calib_set and int(group_counts.get(g, 0)) >= min_target_size
    ]
    if len(target_pool) == 0:
        return None, None

    test_group = int(rng.choice(np.asarray(target_pool, dtype=int), size=1)[0])
    if test_group in calib_set:
        raise RuntimeError("Target PUMA overlapped calibration set (should not happen).")
    return sorted(calib_groups), test_group


def sample_calibration_and_target_uniform(
    eligible_groups,
    group_counts,
    selection_seed,
    n_calib_groups=20,
    min_target_size=MIN_TARGET_PUMA_SIZE,
):
    """
    Uniformly draw calibration and target PUMAs without stratification.

    - Draw n_calib_groups calibration PUMAs uniformly without replacement.
    - Draw one target PUMA uniformly from eligible PUMAs not in calibration.
    - Target PUMA must have at least min_target_size rows.
    """
    target_eligible = [
        int(g)
        for g in eligible_groups
        if int(group_counts.get(g, 0)) >= min_target_size
    ]
    if len(target_eligible) <= n_calib_groups:
        return None, None

    rng = np.random.default_rng(selection_seed)
    calib_groups = rng.choice(
        np.asarray(target_eligible, dtype=int),
        size=int(n_calib_groups),
        replace=False,
    ).tolist()
    calib_set = set(int(g) for g in calib_groups)
    target_pool = [int(g) for g in target_eligible if int(g) not in calib_set]
    if len(target_pool) == 0:
        return None, None

    test_group = int(rng.choice(np.asarray(target_pool, dtype=int), size=1)[0])
    if test_group in calib_set:
        raise RuntimeError("Target PUMA overlapped calibration set (should not happen).")

    return sorted(int(g) for g in calib_groups), test_group


def sample_calibration_mixed_target_large(
    all_groups,
    group_counts,
    selection_seed,
    n_calib_groups=20,
    min_target_size=MIN_TARGET_PUMA_SIZE,
):
    """
    Mixed-size calibration + large target (new ACS sampling plan).

    - Draw ``n_calib_groups`` reference PUMAs uniformly from *all* groups
      (any size; some may be < 20).
    - Draw one test PUMA from the unselected groups with size >= min_target_size.
    """
    all_groups = [int(g) for g in all_groups]
    if len(all_groups) < int(n_calib_groups) + 1:
        return None, None

    target_eligible = [
        int(g) for g in all_groups if int(group_counts.get(g, 0)) >= int(min_target_size)
    ]
    if len(target_eligible) == 0:
        return None, None

    rng = np.random.default_rng(selection_seed)
    calib_groups = rng.choice(
        np.asarray(all_groups, dtype=int),
        size=int(n_calib_groups),
        replace=False,
    ).tolist()
    calib_set = set(int(g) for g in calib_groups)
    target_pool = [int(g) for g in target_eligible if int(g) not in calib_set]
    if len(target_pool) == 0:
        return None, None

    test_group = int(rng.choice(np.asarray(target_pool, dtype=int), size=1)[0])
    return sorted(int(g) for g in calib_groups), test_group


def count_possible_symmetric_draws(
    strata,
    group_counts,
    eligible_groups,
    o_values,
    n_calib_per_stratum=5,
    n_calib_strata=4,
    min_target_size=MIN_TARGET_PUMA_SIZE,
    min_calib_puma_size=None,
):
    """Approx. number of distinct (calib, target) pairs under Option B sampling."""
    if min_calib_puma_size is None:
        min_calib_puma_size = max(int(o) for o in o_values) + 1
    min_calib_puma_size = int(min_calib_puma_size)
    names = sorted(strata.keys())
    calib_strata = names[:n_calib_strata]

    total = 1
    for k in calib_strata:
        n = sum(1 for g in strata[k] if int(group_counts.get(g, 0)) >= min_calib_puma_size)
        if n < n_calib_per_stratum:
            return 0
        total *= math.comb(n, n_calib_per_stratum)

    n_elig_target = sum(
        1 for g in eligible_groups if int(group_counts.get(g, 0)) >= min_target_size
    )
    n_calib = n_calib_per_stratum * n_calib_strata
    # Target pool size depends on which calib PUMAs were drawn; upper bound:
    total *= max(1, n_elig_target - n_calib)
    return total


# ==================== Single replicate ====================

def run_one_replicate(df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx):
    """
    Run one true marginal replicate (Option B):
    1. Five PUMAs from each of strata 1–4 (calibration), disjoint from target
    2. One target PUMA uniform from remaining eligible (size >= 21)
    3. Full PUMA rows; fixed target at index 20
    """
    alpha = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep = config.get('n_repeated', 50)
    quantile_mode = config.get('quantile_mode', 'deterministic')
    stdcp_quantile_mode = str(config.get('stdcp_quantile_mode', 'randomized'))
    quantile_base_seed = config.get('quantile_base_seed', config['seed'])
    outcome_scale = str(config.get('outcome_scale', 'log1p'))
    n_calib_per_stratum = config.get('n_calib_per_stratum', 5)
    n_puma_groups = config.get('n_puma_groups', 20)
    o_values_rep = config.get('o_values', [0, 5, 10, 15, 20])
    eligible_groups = config.get('eligible_groups', [])
    design = config.get('design', 'marginal_one_target')

    group_counts = df.groupby(group_col).size()

    selection_seed = config['seed'] + replicate_idx * 1009
    try:
        if design == 'uniform_one_target':
            calib_groups, test_group = sample_calibration_and_target_uniform(
                eligible_groups=eligible_groups,
                group_counts=group_counts,
                selection_seed=selection_seed,
                n_calib_groups=n_puma_groups,
                min_target_size=MIN_TARGET_PUMA_SIZE,
            )
        elif design == 'mixed_size_one_target':
            calib_groups, test_group = sample_calibration_mixed_target_large(
                all_groups=eligible_groups,
                group_counts=group_counts,
                selection_seed=selection_seed,
                n_calib_groups=n_puma_groups,
                min_target_size=MIN_TARGET_PUMA_SIZE,
            )
        else:
            calib_groups, test_group = sample_calibration_and_target_symmetric(
                strata=strata,
                group_counts=group_counts,
                eligible_groups=eligible_groups,
                selection_seed=selection_seed,
                o_values=o_values_rep,
                n_calib_per_stratum=n_calib_per_stratum,
                n_calib_strata=4,
                min_target_size=MIN_TARGET_PUMA_SIZE,
                min_calib_puma_size=config.get('min_calib_puma_size'),
            )
    except (ValueError, RuntimeError):
        return None

    if calib_groups is None or test_group is None:
        return None

    if test_group in set(calib_groups):
        raise RuntimeError(
            f"Replicate {replicate_idx}: target PUMA {test_group} in calibration set."
        )

    if int(group_counts[test_group]) < MIN_TARGET_PUMA_SIZE:
        return None

    # Full PUMA rows (variable group sizes); target is always index TARGET_INDEX.
    # Optional row permutation randomizes the stream order without bootstrapping.
    row_rng = (
        np.random.default_rng(config['seed'] + replicate_idx * 1009 + 811)
        if bool(config.get('permute_rows', False))
        else None
    )
    group_data = {}
    for grp in calib_groups + [test_group]:
        grp_mask = (df[group_col] == grp).values
        grp_idx = np.where(grp_mask)[0]
        if row_rng is not None:
            grp_idx = row_rng.permutation(grp_idx)
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,
            'income': df.iloc[grp_idx]['income'].values.astype(float),
        }

    # Build Z_calibration for all calibration groups
    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    # HCP train/cal split
    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel,
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    mu_baseline = _make_mu_baseline(config)
    mu_hcp = _make_mu_hcp(config)

    model_baseline = mu_baseline['fit_global'](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )
    score_aux = _fit_baseline_score_aux(
        mu_baseline, model_baseline, U_calibration_full, Z_calibration_full,
        train_idx, alpha,
    )

    scores_list = []
    for j in calib_idx:
        Zj = Z_calibration_full[j]
        Uj = U_calibration_full[j]
        muj = np.array([
            mu_baseline['predict_global'](model_baseline, z['X'], Uj)
            for z in Zj
        ])
        scores_list.append(_baseline_group_scores(Zj, Uj, muj, score_aux))

    T_hcp = compute_hcp_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "hcp"),
    )
    T_pool = compute_pooling_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "pool"),
    )
    T_sub = compute_subsampling_once_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "sub"),
    )
    T_rep = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, n_rep,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "rep"),
    )

    U_test = np.zeros((1, 1))
    score_type = str(config.get("score_type", "absolute")).lower()
    stdcp_score_type = str(config.get("stdcp_score_type", score_type)).lower()
    want_cache = bool(config.get("cache_dir"))

    baseline_result = {m: {} for m in BASELINE_METHODS}
    stdcp_result = {m: {} for m in STD_CP_METHODS}
    hcp_result = {m: {} for m in HCP_METHODS}
    by_o_intermediates = {} if want_cache else None

    # IMPORTANT: Baseline methods (HCP, Pooling, etc.) should only be computed ONCE
    # since they don't use test group observations - they only depend on calibration data
    # We'll compute them at o=0 and reuse for all o values

    if len(group_data[test_group]['Y']) <= TARGET_INDEX:
        # Not enough observations - mark baselines as NaN too
        nan_rec = {
            'coverage': np.nan, 'width': np.nan, 'width_income': np.nan,
            'lower': np.nan, 'upper': np.nan, 'lower_income': np.nan, 'upper_income': np.nan,
        }
        for method in BASELINE_METHODS:
            baseline_result[method][0] = nan_rec.copy()
        for o in o_values:
            for method in HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
        return {
            'baseline': baseline_result,
            'hcp': hcp_result,
            'stdcp': stdcp_result,
            'income_target': np.nan,
        }

    target_index = TARGET_INDEX
    x_target = group_data[test_group]['X'][target_index]
    true_y = group_data[test_group]['Y'][target_index]
    income_target = float(group_data[test_group]['income'][target_index])

    # Baseline methods at fixed target (index 20)
    mu_hat_baseline = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
    for method, T in [
        ('HCP', T_hcp),
        ('Pooling', T_pool),
        ('Subsampling', T_sub),
        ('Repeated', T_rep),
    ]:
        interval = _baseline_interval(mu_hat_baseline, T, x_target, U_test[0], score_aux)
        baseline_result[method][0] = _result_record(interval, true_y, outcome_scale=outcome_scale)

    Z_test_full = [
        {'X': group_data[test_group]['X'][i], 'Y': group_data[test_group]['Y'][i]}
        for i in range(len(group_data[test_group]['Y']))
    ]

    # HCP methods and Std-CP: compute for each o value
    for o in o_values:
        if len(group_data[test_group]['Y']) <= max(o, TARGET_INDEX):
            nan_rec = {
                'coverage': np.nan, 'width': np.nan, 'width_income': np.nan,
                'lower': np.nan, 'upper': np.nan, 'lower_income': np.nan, 'upper_income': np.nan,
            }
            for method in HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
            continue

        o_inter = {} if want_cache else None

        # Donor-HCP with within-group correction (always computed; not forced to HCP at o=0)
        try:
            dhcp_seed = (
                config['seed']
                + (replicate_idx + 1) * 1009
                + (o + 1) * 131
                + 17
            )
            res_dhcp = compute_donor_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test,
                Z_test=Z_test_full,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                test_index_target=target_index,
                tau_override=_tau_override(config),
                random_seed=dhcp_seed,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, replicate_idx, target_index, o, "donor_hcp"
                ),
                return_intermediates=want_cache,
            )
            int_dhcp = res_dhcp['interval']
            if o_inter is not None:
                o_inter['donor_hcp'] = res_dhcp.get('intermediates')
                o_inter['dhcp_seed'] = int(dhcp_seed)
        except Exception:
            int_dhcp = (-np.inf, np.inf)
            if o_inter is not None:
                o_inter['donor_hcp'] = None

        hcp_result['Donor-HCP'][o] = _result_record(int_dhcp, true_y, outcome_scale=outcome_scale)

        # Sample-HCP (randomized) - pass full Z_test, not truncated
        try:
            res_shcp = compute_sample_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test, Z_test=Z_test_full,  # FULL Z_test
                o_observed=o, alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                test_index_target=target_index,  # FIXED target_index
                tau_override=_tau_override(config),
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, replicate_idx, target_index, o, "sample_hcp"
                ),
            )
            int_shcp = res_shcp['interval']
        except Exception:
            int_shcp = (-np.inf, np.inf)

        hcp_result['S-HCP'][o] = _result_record(int_shcp, true_y, outcome_scale=outcome_scale)

        # Standard split CP: first o rows in target PUMA only; predict index target_index.
        # GHCP uses score_type; Std-CP uses stdcp_score_type (may differ).
        stdcp_inter = None
        skip_stdcp = bool(config.get("skip_stdcp", False))
        stdcp_o_values_cfg = config.get("stdcp_o_values")
        if stdcp_o_values_cfg is not None:
            stdcp_o_allowed = {int(v) for v in stdcp_o_values_cfg}
        else:
            stdcp_o_allowed = None
        if skip_stdcp or (stdcp_o_allowed is not None and int(o) not in stdcp_o_allowed):
            int_stdcp = (-np.inf, np.inf)
        elif o > 0:
            stdcp_rng = np.random.default_rng(
                make_quantile_seed(quantile_base_seed, replicate_idx, target_index, o, "stdcp_split")
            )
            stdcp_center = str(config.get("stdcp_center", "local")).lower()
            if stdcp_center == "global":
                X_hist = np.asarray(group_data[test_group]['X'][:o], dtype=float)
                Y_hist = np.asarray(group_data[test_group]['Y'][:o], dtype=float)
                mu_hist = np.array([
                    mu_baseline['predict_global'](model_baseline, X_hist[i], U_test[0])
                    for i in range(len(Y_hist))
                ], dtype=float)
                int_stdcp = _compute_std_cp_interval_global_center(
                    x_hist=X_hist,
                    y_hist=Y_hist,
                    x_target=x_target,
                    alpha=alpha,
                    mu_hat_hist=mu_hist,
                    mu_hat_target=mu_hat_baseline,
                    quantile_mode=stdcp_quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "stdcp"
                    ),
                    rng=stdcp_rng,
                    score_aux=score_aux if stdcp_score_type == "studentized" else None,
                )
                if want_cache:
                    stdcp_inter = {
                        "center_mode": "global",
                        "score_type_used": stdcp_score_type,
                        "x_hist": X_hist.copy(),
                        "y_hist": Y_hist.copy(),
                        "x_target": np.asarray(x_target, dtype=float).ravel().copy(),
                        "y_true": float(true_y),
                        "income_true": float(income_target),
                        "mu_hist": mu_hist.copy(),
                        "mu_target": float(mu_hat_baseline),
                    }
            else:
                out_std = _compute_std_cp_interval(
                    x_hist=group_data[test_group]['X'][:o],
                    y_hist=group_data[test_group]['Y'][:o],
                    x_target=x_target,
                    alpha=alpha,
                    rng=stdcp_rng,
                    quantile_mode=stdcp_quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "stdcp"
                    ),
                    score_type=stdcp_score_type,
                    return_intermediates=want_cache,
                )
                if want_cache:
                    int_stdcp, stdcp_inter = out_std
                    if stdcp_inter is not None:
                        stdcp_inter["y_true"] = float(true_y)
                        stdcp_inter["income_true"] = float(income_target)
                else:
                    int_stdcp = out_std
        else:
            int_stdcp = (-np.inf, np.inf)

        stdcp_result['Std-CP'][o] = _result_record(int_stdcp, true_y, outcome_scale=outcome_scale)
        if o_inter is not None:
            o_inter['stdcp'] = stdcp_inter
            by_o_intermediates[int(o)] = o_inter

    result = {
        'baseline': baseline_result,
        'hcp': hcp_result,
        'stdcp': stdcp_result,
        'income_target': income_target,
    }
    if want_cache:
        result['_cache'] = {
            'version': REPLICATE_CACHE_VERSION,
            'replicate_idx': int(replicate_idx),
            'calib_groups': [int(g) for g in calib_groups],
            'test_group': int(test_group),
            'test_groups': [int(test_group)],
            'group_data': _serialize_group_data(group_data),
            'selection_seed': int(selection_seed),
            'baseline_split': {
                'train_idx': [int(i) for i in train_idx],
                'calib_idx': [int(i) for i in calib_idx],
            },
            'target_index': int(target_index),
            'true_y': float(true_y),
            'income_target': float(income_target),
            'by_o': by_o_intermediates,
            'baseline': baseline_result,
            'hcp': hcp_result,
            'stdcp': stdcp_result,
            'config': _cache_config_subset(config),
        }
    return result


def run_one_replicate_avg_over_targets(
    df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
    fixed_calib_groups=None,
):
    """
    Average coverage over all target PUMAs disjoint from calibration.

    - fixed_calib_groups is None (marginal): resample calib PUMAs each replicate.
    - fixed_calib_groups set (conditional): same calib PUMAs every replicate.
    - Uses observed ACS rows only (no bootstrap). Target index TARGET_INDEX.
    """
    alpha = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep = config.get('n_repeated', 50)
    quantile_mode = config.get('quantile_mode', 'deterministic')
    stdcp_quantile_mode = str(config.get('stdcp_quantile_mode', 'randomized'))
    quantile_base_seed = config.get('quantile_base_seed', config['seed'])
    outcome_scale = str(config.get('outcome_scale', 'log1p'))
    truncate_n = config.get('truncate_n', None)
    n_calib_per = config.get('n_calib_per_stratum', 4)
    o_values_rep = config.get('o_values', o_values)

    group_counts = df.groupby(group_col).size()

    if fixed_calib_groups is not None:
        calib_groups = list(fixed_calib_groups)
    else:
        selection_seed = config['seed'] + replicate_idx * 1009
        design = config.get('design', 'marginal')
        try:
            if design == 'marginal_uniform':
                calib_groups = sample_calibration_uniform(
                    eligible_groups=eligible_groups,
                    group_counts=group_counts,
                    selection_seed=selection_seed,
                    n_calib_groups=config.get('n_puma_groups', 21),
                    o_values=o_values_rep,
                    min_calib_puma_size=config.get('min_calib_puma_size'),
                )
            else:
                calib_groups = sample_calibration_fixed_per_stratum(
                    strata=strata,
                    group_counts=group_counts,
                    selection_seed=selection_seed,
                    o_values=o_values_rep,
                    n_calib_per_stratum=n_calib_per,
                    n_calib_strata=4,
                    min_calib_puma_size=config.get('min_calib_puma_size'),
                )
        except ValueError:
            return None

    test_groups = _test_groups_from_calib(
        eligible_groups, calib_groups, group_counts,
    )
    if len(test_groups) == 0:
        return None

    all_groups = list(calib_groups) + test_groups
    row_rng = (
        np.random.default_rng(config['seed'] + replicate_idx * 1009 + 811)
        if bool(config.get('permute_rows', False))
        else None
    )
    group_data = _load_group_data_static(
        df, X, group_col, all_groups, truncate_n=truncate_n, rng=row_rng,
    )

    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel,
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    mu_baseline = _make_mu_baseline(config)

    model_baseline = mu_baseline['fit_global'](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )
    score_aux = _fit_baseline_score_aux(
        mu_baseline, model_baseline, U_calibration_full, Z_calibration_full,
        train_idx, alpha,
    )
    score_type = str(config.get("score_type", "absolute")).lower()

    scores_list = []
    for j in calib_idx:
        Zj = Z_calibration_full[j]
        Uj = U_calibration_full[j]
        muj = np.array([
            mu_baseline['predict_global'](model_baseline, z['X'], Uj)
            for z in Zj
        ])
        scores_list.append(_baseline_group_scores(Zj, Uj, muj, score_aux))

    T_hcp = compute_hcp_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "hcp"),
    )
    T_pool = compute_pooling_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "pool"),
    )
    T_sub = compute_subsampling_once_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "sub"),
    )
    T_rep = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, n_rep,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "rep"),
    )

    U_test = np.zeros((1, 1))
    baseline_result = {m: {} for m in BASELINE_METHODS}
    stdcp_result = {m: {} for m in STD_CP_METHODS}
    income_vals = []

    for o in o_values:
        eligible_test = [
            grp for grp in test_groups
            if len(group_data[grp]['Y']) > max(o, TARGET_INDEX)
        ]
        if len(eligible_test) == 0:
            nan_rec = {
                'coverage': np.nan, 'width': np.nan, 'width_income': np.nan,
                'lower': np.nan, 'upper': np.nan, 'lower_income': np.nan, 'upper_income': np.nan,
            }
            for method in BASELINE_METHODS:
                baseline_result[method][0] = nan_rec.copy()
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
            continue

        base_cov = {m: [] for m in BASELINE_METHODS}
        base_wid = {m: [] for m in BASELINE_METHODS}
        base_wid_inc = {m: [] for m in BASELINE_METHODS}
        std_cov = {m: [] for m in STD_CP_METHODS}
        std_wid = {m: [] for m in STD_CP_METHODS}
        std_wid_inc = {m: [] for m in STD_CP_METHODS}
        o_income_vals = []

        for test_group in eligible_test:
            target_index = TARGET_INDEX
            x_target = group_data[test_group]['X'][target_index]
            true_y = group_data[test_group]['Y'][target_index]
            o_income_vals.append(float(group_data[test_group]['income'][target_index]))

            mu_hat_baseline = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
            for method, T in [
                ('HCP', T_hcp),
                ('Pooling', T_pool),
                ('Subsampling', T_sub),
                ('Repeated', T_rep),
            ]:
                interval = _baseline_interval(mu_hat_baseline, T, x_target, U_test[0], score_aux)
                base_cov[method].append(_covered(interval, true_y))
                base_wid[method].append(_width(interval))
                base_wid_inc[method].append(_width_income_from_interval(interval, outcome_scale=outcome_scale))

            if o > 0:
                stdcp_rng = np.random.default_rng(
                    make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "stdcp_split"
                    )
                )
                stdcp_center = str(config.get("stdcp_center", "local")).lower()
                stdcp_score_type = str(config.get("stdcp_score_type", score_type)).lower()
                if stdcp_center == "global":
                    X_hist = np.asarray(group_data[test_group]['X'][:o], dtype=float)
                    Y_hist = np.asarray(group_data[test_group]['Y'][:o], dtype=float)
                    mu_hist = np.array([
                        mu_baseline['predict_global'](model_baseline, X_hist[i], U_test[0])
                        for i in range(len(Y_hist))
                    ], dtype=float)
                    int_stdcp = _compute_std_cp_interval_global_center(
                        x_hist=X_hist,
                        y_hist=Y_hist,
                        x_target=x_target,
                        alpha=alpha,
                        mu_hat_hist=mu_hist,
                        mu_hat_target=mu_hat_baseline,
                        quantile_mode=stdcp_quantile_mode,
                        quantile_random_seed=make_quantile_seed(
                            quantile_base_seed, replicate_idx, target_index, o, "stdcp"
                        ),
                        rng=stdcp_rng,
                        score_aux=score_aux if stdcp_score_type == "studentized" else None,
                    )
                else:
                    int_stdcp = _compute_std_cp_interval(
                        x_hist=group_data[test_group]['X'][:o],
                        y_hist=group_data[test_group]['Y'][:o],
                        x_target=x_target,
                        alpha=alpha,
                        rng=stdcp_rng,
                        quantile_mode=stdcp_quantile_mode,
                        quantile_random_seed=make_quantile_seed(
                            quantile_base_seed, replicate_idx, target_index, o, "stdcp"
                        ),
                        score_type=stdcp_score_type,
                    )
            else:
                int_stdcp = (-np.inf, np.inf)

            std_cov['Std-CP'].append(_covered(int_stdcp, true_y))
            std_wid['Std-CP'].append(_width(int_stdcp))
            std_wid_inc['Std-CP'].append(_width_income_from_interval(int_stdcp, outcome_scale=outcome_scale))

        if o_income_vals:
            income_vals.extend(o_income_vals)

        for method in BASELINE_METHODS:
            baseline_result[method][0] = {
                'coverage': float(np.mean(base_cov[method])),
                'width': float(np.nanmean(base_wid[method])),
                'width_income': float(np.nanmean(base_wid_inc[method])),
                'lower': np.nan,
                'upper': np.nan,
                'lower_income': np.nan,
                'upper_income': np.nan,
            }

        for method in STD_CP_METHODS:
            stdcp_result[method][o] = {
                'coverage': float(np.mean(std_cov[method])),
                'width': float(np.nanmean(std_wid[method])),
                'width_income': float(np.nanmean(std_wid_inc[method])),
                'lower': np.nan,
                'upper': np.nan,
                'lower_income': np.nan,
                'upper_income': np.nan,
            }

    hcp_result = _compute_hcp_over_targets(
        group_data=group_data,
        calib_groups=calib_groups,
        test_groups=test_groups,
        o_values=o_values,
        config=config,
        replicate_idx=replicate_idx,
    )

    result = {
        'baseline': baseline_result,
        'hcp': hcp_result,
        'stdcp': stdcp_result,
        'income_target': float(np.nanmean(income_vals)) if income_vals else np.nan,
    }
    if config.get('cache_dir'):
        result['_cache'] = {
            'version': REPLICATE_CACHE_VERSION,
            'replicate_idx': int(replicate_idx),
            'calib_groups': [int(g) for g in calib_groups],
            'test_groups': [int(g) for g in test_groups],
            'group_data': _serialize_group_data(group_data),
            'baseline': baseline_result,
            'stdcp': stdcp_result,
            'income_target': result['income_target'],
            'config': _cache_config_subset(config),
        }
    return result


# ==================== Worker functions ====================

_WORKER_SHARED = None

def _init_worker(df, X, eligible_groups, strata, group_col, o_values, config,
                 calib_groups=None, test_groups=None):
    global _WORKER_SHARED
    _WORKER_SHARED = (
        df, X, list(eligible_groups), strata, group_col, o_values, config,
        calib_groups, test_groups,
    )


def _run_one_replicate_worker(replicate_idx):
    df, X, eligible_groups, strata, group_col, o_values, config, calib_groups, test_groups = (
        _WORKER_SHARED
    )
    design = config.get('design', 'marginal')
    if design == 'conditional':
        result = run_one_replicate_avg_over_targets(
            df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
            fixed_calib_groups=calib_groups,
        )
    elif design in ('marginal_one_target', 'uniform_one_target', 'mixed_size_one_target'):
        result = run_one_replicate(
            df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
        )
    elif design in ('marginal', 'marginal_uniform'):
        # resample calib each replicate, average over all test PUMAs
        result = run_one_replicate_avg_over_targets(
            df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
            fixed_calib_groups=None,
        )
    return replicate_idx, result


# ==================== Main experiment ====================

def run_true_marginal_experiments(df, X, eligible_groups, strata, group_col, o_values, config):
    """Run true marginal experiments."""
    B = config['B']
    n_o = len(o_values)
    n_workers = int(config.get('n_workers', 1))

    n_m = len(METHODS)
    m_map = {m: i for i, m in enumerate(METHODS)}

    cov = np.full((B, n_m, n_o), np.nan)
    wid = np.full((B, n_m, n_o), np.nan)
    wid_income = np.full((B, n_m, n_o), np.nan)
    lower_log = np.full((B, n_m, n_o), np.nan)
    upper_log = np.full((B, n_m, n_o), np.nan)
    lower_income = np.full((B, n_m, n_o), np.nan)
    upper_income = np.full((B, n_m, n_o), np.nan)
    income_targets = np.full(B, np.nan)

    print(f"  Running {B} true marginal replicates with {n_workers} workers")

    def _write_result(rep_idx, result):
        if result is None:
            return

        cache = result.pop('_cache', None)
        if cache is not None and config.get('cache_dir'):
            save_replicate_cache(Path(config['cache_dir']), rep_idx, cache)

        income_targets[rep_idx] = result.get('income_target', np.nan)

        def _store(m_i, o_i, rec):
            cov[rep_idx, m_i, o_i] = rec['coverage']
            wid[rep_idx, m_i, o_i] = rec['width']
            wid_income[rep_idx, m_i, o_i] = rec['width_income']
            lower_log[rep_idx, m_i, o_i] = rec['lower']
            upper_log[rep_idx, m_i, o_i] = rec['upper']
            lower_income[rep_idx, m_i, o_i] = rec['lower_income']
            upper_income[rep_idx, m_i, o_i] = rec['upper_income']

        for method in BASELINE_METHODS:
            if 0 in result['baseline'][method]:
                _store(m_map[method], 0, result['baseline'][method][0])

        for method in HCP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                if o in result['hcp'][method]:
                    _store(m_i, o_i, result['hcp'][method][o])

        for method in STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                if o in result['stdcp'][method]:
                    _store(m_i, o_i, result['stdcp'][method][o])

    calib_groups = config.get('fixed_calib_groups')
    test_groups = config.get('fixed_test_groups')

    if n_workers <= 1:
        _init_worker(
            df, X, eligible_groups, strata, group_col, o_values, config,
            calib_groups, test_groups,
        )
        for b in range(B):
            if (b + 1) % 100 == 0 or (b + 1) <= 10:
                print(f"    Replicate {b+1}/{B}")
            rep_idx, result = _run_one_replicate_worker(b)
            _write_result(rep_idx, result)
    else:
        completed = 0
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(
                df, X, eligible_groups, strata, group_col, o_values, config,
                calib_groups, test_groups,
            ),
        ) as pool:
            for rep_idx, result in pool.imap_unordered(_run_one_replicate_worker, range(B)):
                completed += 1
                if completed % 100 == 0 or completed <= 10 or completed == B:
                    print(f"    Replicate {completed}/{B}")
                _write_result(rep_idx, result)

    return {
        'methods': METHODS,
        'o_values': o_values,
        'coverage': cov,
        'width': wid,
        'width_income': wid_income,
        'lower_log': lower_log,
        'upper_log': upper_log,
        'lower_income': lower_income,
        'upper_income': upper_income,
        'income_targets': income_targets,
    }


# ==================== Save results ====================

def save_results_to_csv(results, output_path, global_pooled=None):
    """Save results to CSV in long format."""
    methods = results['methods']
    o_values = results['o_values']
    cov = results['coverage']
    wid = results['width']
    wid_income = results['width_income']
    lower_log = results['lower_log']
    upper_log = results['upper_log']
    lower_income = results['lower_income']
    upper_income = results['upper_income']
    income_targets = results['income_targets']

    B = cov.shape[0]

    rows = []
    for b in range(B):
        for m_i, method in enumerate(methods):
            for o_i, o in enumerate(o_values):
                rows.append({
                    'replicate': b,
                    'method': method,
                    'o': o,
                    'coverage': cov[b, m_i, o_i],
                    'width': wid[b, m_i, o_i],
                    'width_income': wid_income[b, m_i, o_i],
                    'lower': lower_log[b, m_i, o_i],
                    'upper': upper_log[b, m_i, o_i],
                    'lower_income': lower_income[b, m_i, o_i],
                    'upper_income': upper_income[b, m_i, o_i],
                    'income_target': income_targets[b],
                })

    if global_pooled is not None:
        L_p, U_p = global_pooled['lower'], global_pooled['upper']
        for b in range(B):
            inc = income_targets[b]
            covered = (
                float(L_p <= inc <= U_p)
                if np.isfinite(inc) and np.isfinite(L_p) and np.isfinite(U_p)
                else np.nan
            )
            for o in o_values:
                rows.append({
                    'replicate': b,
                    'method': 'Pooled-Income-Quantile',
                    'o': o,
                    'coverage': covered,
                    'width': np.nan,
                    'width_income': U_p - L_p,
                    'lower': np.nan,
                    'upper': np.nan,
                    'lower_income': L_p,
                    'upper_income': U_p,
                    'income_target': inc,
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  Saved detailed results to {output_path}")
    return df_out


def save_summaries(df_out: pd.DataFrame, summary_dir: Path):
    """Write coverage/width summary CSVs (long + pivot)."""
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (method, o), g in df_out.groupby(['method', 'o']):
        cov_vals = g['coverage'].dropna().values
        n = len(cov_vals)
        if n == 0:
            continue
        p = float(np.mean(cov_vals))
        se = float(np.sqrt(p * (1 - p) / n))
        w_inc = g['width_income'].replace([np.inf, -np.inf], np.nan).dropna().values
        rows.append({
            'method': method,
            'o': o,
            'coverage_mean': p,
            'coverage_std': float(np.std(cov_vals, ddof=1)) if n > 1 else 0.0,
            'coverage_se': se,
            'width_income_median': float(np.median(w_inc)) if len(w_inc) else np.nan,
            'width_income_mean': float(np.mean(w_inc)) if len(w_inc) else np.nan,
            'width_income_std': float(np.std(w_inc, ddof=1)) if len(w_inc) > 1 else np.nan,
            'n': n,
        })

    df_long = pd.DataFrame(rows)
    df_long.to_csv(summary_dir / 'acs_true_marg_summary_long.csv', index=False)

    if len(df_long):
        cov_pivot = df_long.pivot(index='o', columns='method', values='coverage_mean')
        cov_pivot.to_csv(summary_dir / 'acs_true_marg_coverage_summary.csv')
        wid_pivot = df_long.pivot(index='o', columns='method', values='width_income_median')
        wid_pivot.to_csv(summary_dir / 'acs_true_marg_width_summary.csv')

    print(f"  Saved summaries to {summary_dir}")
    return df_long


def save_endpoints_analysis(df_out, global_pooled, summary_dir, o_compare=20):
    """Report fixed global pooled endpoints vs Donor-HCP replicate intervals."""
    summary_dir = Path(summary_dir)
    L_p, U_p = global_pooled['lower'], global_pooled['upper']

    lines = [
        "ACS true marginal — interval endpoints",
        "=" * 60,
        "",
        "Global pooled-income quantile (all filtered ACS rows, one interval):",
        f"  lower_income = ${L_p:,.0f}",
        f"  upper_income = ${U_p:,.0f}",
        f"  width        = ${U_p - L_p:,.0f}",
        "",
    ]

    dhcp = df_out[(df_out['method'] == 'Donor-HCP') & (df_out['o'] == o_compare)].copy()
    fin = dhcp[np.isfinite(dhcp['lower_income']) & np.isfinite(dhcp['upper_income'])]
    if len(fin):
        med_lo = fin['lower_income'].median()
        med_hi = fin['upper_income'].median()
        lines += [
            f"Donor-HCP (with local correction) at o={o_compare} — replicate intervals (income $):",
            f"  median lower = ${med_lo:,.0f}",
            f"  median upper = ${med_hi:,.0f}",
            f"  median width = ${(med_hi - med_lo):,.0f}",
            f"  replicate lower: 5%=${fin['lower_income'].quantile(0.05):,.0f}, "
            f"95%=${fin['lower_income'].quantile(0.95):,.0f}",
            f"  replicate upper: 5%=${fin['upper_income'].quantile(0.05):,.0f}, "
            f"95%=${fin['upper_income'].quantile(0.95):,.0f}",
            "",
            "Geometry vs global pooled interval (per replicate):",
        ]
        nested = ((fin['lower_income'] >= L_p) & (fin['upper_income'] <= U_p)).mean()
        overlap = (
            (fin['upper_income'] >= L_p) & (fin['lower_income'] <= U_p)
        ).mean()
        strictly_lower = (fin['upper_income'] < L_p).mean()
        strictly_higher = (fin['lower_income'] > U_p).mean()
        lines += [
            f"  nested inside pooled:     {100 * nested:.1f}%",
            f"  overlaps pooled:          {100 * overlap:.1f}%",
            f"  entirely below pooled:    {100 * strictly_lower:.1f}%",
            f"  entirely above pooled:    {100 * strictly_higher:.1f}%",
            "",
            "Interpretation: Donor-HCP intervals vary by replicate (centered at mu_hat);",
            "they are not nested in each other. Compare overlap with the fixed pooled band.",
        ]

    text = "\n".join(lines)
    out_txt = summary_dir / 'acs_true_marg_endpoints.txt'
    out_txt.write_text(text)
    print(text)
    print(f"\n  Saved endpoint report: {out_txt}")

    rows = [{
        'method': 'Pooled-Income-Quantile',
        'o': 'global',
        'lower_income': L_p,
        'upper_income': U_p,
        'width_income': U_p - L_p,
    }]
    if len(fin):
        rows.append({
            'method': 'Donor-HCP',
            'o': o_compare,
            'lower_income': med_lo,
            'upper_income': med_hi,
            'width_income': med_hi - med_lo,
            'note': 'medians over replicates',
        })
    pd.DataFrame(rows).to_csv(summary_dir / 'acs_true_marg_endpoints_summary.csv', index=False)


# ==================== Main entry point ====================

if __name__ == '__main__':
    import argparse
    mp.set_start_method('fork', force=True)

    base_dir = ACS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1000, help='Number of replicates')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of workers')
    parser.add_argument('--alpha', type=float, default=0.2, help='Miscoverage level (0.2 => 80%% coverage)')
    parser.add_argument('--acs_state', type=str, default='CA')
    parser.add_argument('--acs_csv', type=str, default=None, help='Path to ACS PUMS CSV')
    parser.add_argument('--n_puma_groups', type=int, default=20, help='Number of non-target PUMAs')
    parser.add_argument('--min_puma_size', type=int, default=None,
                        help='Minimum PUMA size for eligibility/strata (default: 21)')
    parser.add_argument(
        '--min_calib_puma_size',
        type=int,
        default=None,
        help='Minimum PUMA size for calibration draws (default: --min_puma_size; targets still need --target_index+1)',
    )
    parser.add_argument(
        '--target_index',
        type=int,
        default=20,
        help='Fixed within-PUMA target row index (history uses indices 0..o-1; need size > max(o, target_index))',
    )
    parser.add_argument('--o_values', type=str, default='0,5,10,15,20', help='Comma-separated o values')
    parser.add_argument(
        '--design',
        type=str,
        default='marginal',
        choices=['marginal', 'marginal_uniform', 'conditional', 'marginal_one_target', 'uniform_one_target', 'mixed_size_one_target'],
        help=(
            'marginal: resample calib PUMAs each replicate, average over all test PUMAs. '
            'marginal_uniform: uniform calib PUMA draw (no strata), same averaging. '
            'conditional: fixed calib PUMAs, same averaging (no bootstrap). '
            'marginal_one_target: stratified calib + one random target PUMA per replicate. '
            'uniform_one_target: uniform calib PUMAs + one uniform target PUMA per replicate '
            '(both drawn from size-eligible pool). '
            'mixed_size_one_target: calib from all PUMAs (any size); target from unselected '
            'PUMAs with size >= target_index+1.'
        ),
    )
    parser.add_argument(
        '--n_calib_per_stratum',
        type=int,
        default=None,
        help='PUMAs per stratum for calibration (default 5 if marginal, 4 if conditional)',
    )
    parser.add_argument(
        '--group_selection_seed',
        type=int,
        default=42,
        help='Seed for fixed calibration PUMA draw (conditional design)',
    )
    parser.add_argument(
        '--truncate_n',
        type=int,
        default=None,
        help='If set, use only the first N rows per PUMA (DGP uses N=21)',
    )
    parser.add_argument(
        '--quantile-mode',
        choices=['deterministic', 'randomized'],
        default='deterministic',
        help='Conformal threshold selection for HCP/GHCP/baselines (not Std-CP).',
    )
    parser.add_argument(
        '--stdcp-quantile-mode',
        choices=['deterministic', 'randomized'],
        default='randomized',
        help='Conformal threshold selection for Std-CP only (default: randomized).',
    )
    parser.add_argument(
        '--quantile-base-seed',
        type=int,
        default=456,
        help='Base seed for reproducible randomized conformal quantiles.',
    )
    parser.add_argument(
        '--no_within_group',
        action='store_true',
        help='Disable within-group training/shrinkage in Donor-HCP and S-HCP.',
    )
    parser.add_argument(
        '--no_permute_rows',
        action='store_true',
        help=(
            'Disable within-PUMA row permutation (default: permute). '
            'Not recommended: fixed row order breaks exchangeability for target index 20.'
        ),
    )
    parser.add_argument(
        '--skip_stdcp',
        action='store_true',
        help=(
            'Skip local Std-CP (expensive RF fit per o). Std-CP rows are written as '
            'trivial/infinite placeholders; recompute later with the same seeds.'
        ),
    )
    parser.add_argument(
        '--stdcp_o_values',
        type=str,
        default=None,
        help=(
            'Comma-separated o values for which to fit Std-CP (default: all --o_values). '
            'Other o get placeholder intervals. Implies Std-CP is not fully skipped.'
        ),
    )
    parser.add_argument(
        '--predictor',
        type=str,
        choices=PREDICTOR_CHOICES,
        default='ols',
        help='Global mu learner: ols, rf, or ding_xgb (CA pre-trained XGB, Ding 2021 hyperparams).',
    )
    parser.add_argument(
        '--outcome_scale',
        type=str,
        choices=OUTCOME_SCALE_CHOICES,
        default='log1p',
        help='Outcome scale for Y and conformal prediction: log1p (default) or raw income.',
    )
    parser.add_argument(
        '--within_group_mode',
        type=str,
        choices=WITHIN_GROUP_MODE_CHOICES,
        default='mean',
        help='Within-group adjustment for OLS/RF: mean shrinkage (default) or residual correction.',
    )
    parser.add_argument(
        '--drop_top_income_pct',
        type=float,
        default=0.0,
        help='Drop top fraction of incomes as outliers before fitting (e.g. 0.02 = top 2%%).',
    )
    parser.add_argument(
        '--min_income',
        type=float,
        default=10000.0,
        help='Minimum income floor (default 10000). Set to 0 to disable.',
    )
    parser.add_argument(
        '--yoep_min_year',
        type=int,
        default=2012,
        help='Minimum year-of-entry cutoff for recent immigrants (default: 2012).',
    )
    parser.add_argument(
        '--no_age_filter',
        action='store_true',
        help='Disable age 25–54 filter (keep all ages).',
    )
    parser.add_argument(
        '--no_hours_filter',
        action='store_true',
        help='Disable usual-hours labor-force filter (NA hours filled with 0).',
    )
    parser.add_argument(
        '--exclude_entry_recency',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Drop entry_recency (YOEP-derived) from OLS/RF predictors (default: True).',
    )
    parser.add_argument(
        '--exclude_cow',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Drop class-of-worker (cow) dummies from OLS/RF predictors (default: True).',
    )
    parser.add_argument(
        '--exclude_hours',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Drop hours from OLS/RF predictors (default: False; cohort may still filter on hours).',
    )
    parser.add_argument(
        '--stdcp_center',
        type=str,
        choices=('local', 'global'),
        default='local',
        help=(
            'Std-CP center: local = fit RF on o/2 target-PUMA rows (default); '
            'global = freeze the same global RF used by HCP/GHCP and conformalize '
            'on all o local residuals (fairer apples-to-apples baseline).'
        ),
    )
    parser.add_argument(
        '--score_type',
        type=str,
        choices=SCORE_TYPE_CHOICES,
        default='absolute',
        help=(
            'GHCP/HCP/baseline nonconformity: absolute |Y-μ| (default) or studentized '
            '|Y-μ|/σ (RF-σ). Fully studentized ACS runs write under '
            'paper-results/acs/studentized/.'
        ),
    )
    parser.add_argument(
        '--stdcp_score_type',
        type=str,
        choices=SCORE_TYPE_CHOICES,
        default=None,
        help=(
            'Std-CP score independently of --score_type (default: same as --score_type). '
            'Use --score_type absolute --stdcp_score_type studentized for GHCP absolute + '
            'Std-CP studentized.'
        ),
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help=(
            'Directory for per-replicate v2 caches (group_data, selection, per-o GHCP '
            'score atoms, Std-CP split + μ/σ ingredients) for later score swaps.'
        ),
    )
    parser.add_argument(
        '--recompute_from_cache',
        type=str,
        default=None,
        help='Read replicate caches from this directory and write results without rerunning ACS.',
    )
    parser.add_argument(
        '--recompute_within_group_mode',
        type=str,
        choices=WITHIN_GROUP_MODE_CHOICES,
        default=None,
        help='Within-group mode when using --recompute_from_cache.',
    )

    args = parser.parse_args()
    args.permute_rows = not args.no_permute_rows
    args.min_income_effective = None if args.min_income <= 0 else float(args.min_income)
    _set_target_index(args.target_index)
    if args.min_puma_size is None:
        args.min_puma_size = 21
    if args.min_calib_puma_size is None:
        args.min_calib_puma_size = args.min_puma_size

    print("=" * 70)
    print("ACS TRUE MARGINAL COVERAGE EXPERIMENTS")
    print("=" * 70)

    acs_csv = Path(args.acs_csv) if args.acs_csv else (base_dir / 'data/acs_data_all50states.csv')
    if not acs_csv.exists():
        raise FileNotFoundError(
            f"ACS data not found: {acs_csv}\n"
            "Run: python real_data/acs/download_acs_ca_pums.py\n"
            "Or pass --acs_csv /path/to/acs_data_all50states.csv"
        )

    # Load data
    print("\nLoading ACS data...")
    age_min = None if args.no_age_filter else 25
    age_max = None if args.no_age_filter else 54
    min_hours = None if args.no_hours_filter else 40
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=[args.acs_state],
        age_min=age_min,
        age_max=age_max,
        yoep_min_year=args.yoep_min_year,
        min_hours=min_hours,
        min_income=args.min_income_effective,
        drop_top_income_fraction=(
            args.drop_top_income_pct if args.drop_top_income_pct > 0 else None
        ),
        y_transform=_outcome_transform(args.outcome_scale),
    )

    df = df.dropna(subset=['puma']).copy()
    df['puma'] = df['puma'].astype(int)
    X = build_design_matrix_acs(
        df,
        exclude_entry_recency=args.exclude_entry_recency,
        exclude_cow=args.exclude_cow,
        exclude_hours=args.exclude_hours,
    )

    # Get eligible groups
    counts = df.groupby('puma').size()
    counts_eligible = counts[counts >= args.min_puma_size]
    eligible_groups = counts_eligible.index.to_numpy().tolist()

    print(f"  State: {args.acs_state}")
    print(f"  Total observations: {len(df)}")
    print(f"  Total PUMAs: {len(counts)}")
    print(f"  Eligible PUMAs (size >= {args.min_puma_size}): {len(eligible_groups)}")
    print(f"  Min group size: {counts_eligible.min()}")
    print(f"  Max group size: {counts_eligible.max()}")
    print(f"  Median group size: {counts_eligible.median():.0f}")

    # Build strata
    print("\n  Computing BA+ share stratification...")
    group_share_baplus = compute_group_share_baplus(df, 'puma')
    strata = _build_share_baplus_strata(
        eligible_groups=eligible_groups,
        group_share_baplus=group_share_baplus,
        n_strata=5,
    )

    group_counts = counts
    target_eligible = [
        g for g in eligible_groups if int(group_counts[g]) >= MIN_TARGET_PUMA_SIZE
    ]

    print(f"  Number of strata: {len(strata)}")
    for i, k in enumerate(sorted(strata.keys()), start=1):
        n_big = sum(1 for g in strata[k] if int(group_counts[g]) >= MIN_TARGET_PUMA_SIZE)
        role = "calibration strata" if i <= 4 else "target pool only"
        print(f"    Stratum {i} {k}: {len(strata[k])} PUMAs, {n_big} with size>={MIN_TARGET_PUMA_SIZE} ({role})")
    print(f"  Target-eligible PUMAs (size >= {MIN_TARGET_PUMA_SIZE}): {len(target_eligible)}")

    o_values = sorted(set(int(v) for v in args.o_values.split(',')))
    stdcp_o_values = None
    if args.stdcp_o_values:
        stdcp_o_values = sorted(set(int(v) for v in args.stdcp_o_values.split(',')))
    print(f"\n  o values: {o_values}")

    design = args.design
    n_calib_per = args.n_calib_per_stratum
    if n_calib_per is None:
        n_calib_per = 4 if design in ('conditional', 'marginal') else 5

    fixed_calib_groups = None
    fixed_test_groups = None
    if design == 'conditional':
        fixed_calib_groups = sample_calibration_fixed_per_stratum(
            strata=strata,
            group_counts=group_counts,
            selection_seed=args.group_selection_seed,
            o_values=o_values,
            n_calib_per_stratum=n_calib_per,
            n_calib_strata=4,
            min_calib_puma_size=args.min_calib_puma_size,
        )
        calib_set = set(fixed_calib_groups)
        fixed_test_groups = sorted(
            g for g in eligible_groups
            if g not in calib_set and int(group_counts[g]) >= MIN_TARGET_PUMA_SIZE
        )
        if len(fixed_test_groups) == 0:
            raise ValueError('No test PUMAs left after fixing calibration set.')

    n_possible = count_possible_symmetric_draws(
        strata,
        group_counts,
        eligible_groups,
        o_values,
        n_calib_per_stratum=n_calib_per,
        min_calib_puma_size=args.min_calib_puma_size,
    )
    if design == 'conditional':
        print(
            f"\n  Conditional: fixed {n_calib_per} calib PUMAs/stratum × 4 strata "
            f"(seed={args.group_selection_seed}), no row bootstrap, "
            f"row permutation={args.permute_rows}, "
            f"average coverage over {len(fixed_test_groups)} test PUMAs (index {TARGET_INDEX})"
        )
        print(f"  Fixed calibration PUMAs: {fixed_calib_groups}")
    elif design == 'marginal_one_target':
        print(
            f"\n  Marginal one-target: {n_calib_per} calib PUMAs/stratum from strata 1–4 "
            f"(calib size >= {args.min_calib_puma_size}), then 1 target PUMA from the remainder; "
            f"row permutation={args.permute_rows}"
        )
        print(f"  Approx. upper bound on (calib, target) pairs: {n_possible:,}; B={args.B} replicates")
    elif design == 'uniform_one_target':
        print(
            f"\n  Uniform one-target: each replicate draws {args.n_puma_groups} calibration PUMAs "
            f"uniformly from all {len(target_eligible)} target-eligible PUMAs, then 1 target "
            f"PUMA uniformly from the remainder (index {TARGET_INDEX}); no row bootstrap; "
            f"row permutation={args.permute_rows}"
        )
    elif design == 'mixed_size_one_target':
        print(
            f"\n  Mixed-size one-target: each replicate draws {args.n_puma_groups} calibration "
            f"PUMAs uniformly from all {len(eligible_groups)} PUMAs (any size), then 1 test "
            f"PUMA from unselected PUMAs with size >= {MIN_TARGET_PUMA_SIZE} "
            f"(predict index {TARGET_INDEX}); row permutation={args.permute_rows}"
        )
    elif design == 'marginal_uniform':
        n_test_approx = max(0, len(target_eligible) - args.n_puma_groups)
        print(
            f"\n  Marginal uniform (no strata): each replicate draws {args.n_puma_groups} "
            f"calibration PUMAs uniformly from all {len(target_eligible)} eligible PUMAs "
            f"(size >= {args.min_calib_puma_size}), no row bootstrap, "
            f"row permutation={args.permute_rows}, "
            f"then averages coverage over remaining target PUMAs (~{n_test_approx} when all eligible "
            f"have size >= {MIN_TARGET_PUMA_SIZE}; index {TARGET_INDEX})"
        )
    else:
        print(
            f"\n  Marginal: each replicate draws {n_calib_per} calib PUMAs/stratum × 4 strata "
            f"(calib size >= {args.min_calib_puma_size}), no row bootstrap, "
            f"row permutation={args.permute_rows}, "
            f"then averages coverage over target PUMAs with size > max(o, {TARGET_INDEX}) "
            f"(index {TARGET_INDEX})"
        )
        n_test_approx = sum(
            1 for g in target_eligible
            if int(group_counts[g]) > max(max(o_values), TARGET_INDEX)
        )
        print(f"  Target PUMAs with size > max(o, target_index): {n_test_approx}")

    # Global pooled-income interval (one prediction set for all data)
    pooled_lo, pooled_hi = compute_global_pooled_income_interval(df, args.alpha)
    global_pooled = {
        'lower': pooled_lo,
        'upper': pooled_hi,
        'width': pooled_hi - pooled_lo,
    }
    print(f"\n  Global pooled-income interval (all {len(df)} rows):")
    print(f"    lower = ${pooled_lo:,.0f},  upper = ${pooled_hi:,.0f},  width = ${global_pooled['width']:,.0f}")

    pretrained_model = None
    if args.predictor == "ding_xgb":
        target_label = "income" if args.outcome_scale == "income" else "log1p income"
        print(
            "\n  Pre-training Ding et al. (2021) XGBoost on California ACS "
            f"(n_estimators={5}, max_depth={5}, target={target_label})..."
        )
        pretrained_model = train_ding_2021_xgb_regressor(X, df["y"].to_numpy(), u_dim=1)
        print(f"  Pre-trained on {len(df)} individuals.")

    # Config
    config = {
        'B': args.B,
        'seed': 456,
        'alpha': args.alpha,
        'alpha_selection': 0.5,
        'quantile_mode': args.quantile_mode,
        'stdcp_quantile_mode': args.stdcp_quantile_mode,
        'quantile_base_seed': args.quantile_base_seed,
        'n_repeated': 50,
        'n_puma_groups': args.n_puma_groups,
        'n_calib_per_stratum': n_calib_per,
        'o_values': o_values,
        'eligible_groups': eligible_groups,
        'n_workers': args.n_workers,
        'design': design,
        'fixed_calib_groups': fixed_calib_groups,
        'fixed_test_groups': fixed_test_groups,
        'truncate_n': args.truncate_n,
        'within_group': not args.no_within_group,
        'permute_rows': args.permute_rows,
        'skip_stdcp': bool(args.skip_stdcp),
        'stdcp_o_values': stdcp_o_values,
        'min_calib_puma_size': args.min_calib_puma_size,
        'predictor': args.predictor,
        'pretrained_model': pretrained_model,
        'outcome_scale': args.outcome_scale,
        'within_group_mode': args.within_group_mode,
        'drop_top_income_fraction': float(args.drop_top_income_pct),
        'yoep_min_year': int(args.yoep_min_year),
        'min_income': args.min_income_effective,
        'stdcp_center': str(args.stdcp_center),
        'score_type': str(args.score_type),
        'stdcp_score_type': str(
            args.stdcp_score_type if args.stdcp_score_type is not None else args.score_type
        ),
    }
    if args.cache_dir:
        config['cache_dir'] = str(Path(args.cache_dir).resolve())
        print(f"  Replicate cache: {config['cache_dir']}")

    within_group_mode_for_output = args.within_group_mode
    cohort = {}
    if args.recompute_from_cache:
        if args.recompute_within_group_mode is None:
            raise ValueError("--recompute_within_group_mode is required with --recompute_from_cache")
        cache_dir = Path(args.recompute_from_cache).resolve()
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        config.update(manifest.get("config", {}))
        config["within_group_mode"] = args.recompute_within_group_mode
        within_group_mode_for_output = args.recompute_within_group_mode
        cohort = manifest.get("cohort", {})
        global_pooled = manifest.get("global_pooled")
        if global_pooled is None:
            raise ValueError("Cache manifest is missing global_pooled")
        o_values = list(config["o_values"])
        B = int(manifest.get("B", config["B"]))
        print(f"\nRecomputing from cache: {cache_dir}")
        print(f"  Within-group mode: {within_group_mode_for_output}")
        print(f"  Replicates: {B}")
        print(f"  Workers: {args.n_workers}")
        print()
        results = run_recompute_from_cache(
            cache_dir,
            within_group_mode=within_group_mode_for_output,
            B=B,
            o_values=o_values,
            n_workers=args.n_workers,
        )
    else:
        print(f"\nRunning ACS coverage experiments (design={design}):")
        print(f"  Alpha: {args.alpha} (nominal coverage {1 - args.alpha:.0%})")
        print(f"  Outcome scale: {args.outcome_scale}")
        print(f"  YOEP cutoff: >= {args.yoep_min_year}")
        if args.no_age_filter:
            print("  Age filter: disabled")
        if args.no_hours_filter:
            print("  Hours filter: disabled")
        if args.exclude_entry_recency:
            print("  Predictors: entry_recency excluded (YOEP used only as cohort filter)")
        if args.exclude_cow:
            print("  Predictors: class-of-worker (cow) dummies excluded")
        else:
            print("  Predictors: class-of-worker (cow) dummies included")
        if args.exclude_hours:
            print("  Predictors: hours excluded (cohort hours filter may still apply)")
        elif not args.no_hours_filter:
            print("  Predictors: hours included")
        if args.drop_top_income_pct > 0:
            print(f"  Dropped top {args.drop_top_income_pct:.0%} income outliers (per state)")
        elif args.min_income_effective is None:
            print("  No income floor or top-income trim")
        elif args.min_income_effective > 0:
            print(f"  Minimum income: >= ${args.min_income_effective:,.0f}")
        if args.predictor == "ding_xgb":
            print("  Global predictor: ding_xgb (CA pre-trained; residual correction g_j on R=Y-mu_glob)")
        elif args.predictor == "rf":
            print(f"  Global predictor: rf (ntree={RF_NTREE}, nodesize={RF_NODESIZE})")
        else:
            print(f"  Global predictor: {args.predictor}")
        if not args.no_within_group and args.predictor in ("ols", "rf"):
            print(f"  {args.predictor.upper()} within-group mode: {args.within_group_mode}")
        print(f"  Donor-HCP: stock randomized interval; within-group training = {not args.no_within_group}")
        print(f"  Conformal quantile mode (HCP/GHCP/baselines): {args.quantile_mode}")
        print(f"  Std-CP quantile mode: {config['stdcp_quantile_mode']}")
        print(f"  GHCP/HCP score type: {config['score_type']}")
        print(f"  Std-CP score type: {config['stdcp_score_type']}")
        print(f"  Std-CP center: {args.stdcp_center}")
        if config.get("skip_stdcp"):
            print("  Std-CP: SKIPPED (placeholders only; recompute with same seeds)")
        elif config.get("stdcp_o_values") is not None:
            print(f"  Std-CP o values: {config['stdcp_o_values']} (others placeholder)")
        print(
            f"  Target index: {TARGET_INDEX}; calib ∩ target = ∅; no ACS row bootstrap; "
            f"row permutation = {args.permute_rows}"
        )
        if not args.permute_rows:
            print(
                "  WARNING: permute_rows=False — indices 0..o-1 are fixed ACS row order, "
                f"not exchangeable with target index {TARGET_INDEX}. Expect coverage to fall as o increases."
            )
        print(f"  Replicates: {args.B}")
        print(f"  Non-target PUMAs per replicate: {args.n_puma_groups}")
        print(f"  Workers: {args.n_workers}")
        print()

        results = run_true_marginal_experiments(
            df=df,
            X=X,
            eligible_groups=eligible_groups,
            strata=strata,
            group_col='puma',
            o_values=o_values,
            config=config,
        )
        if args.cache_dir:
            write_cache_manifest(
                Path(config['cache_dir']),
                config,
                B=config['B'],
                global_pooled=global_pooled,
                cohort={
                    'yoep_min_year': int(args.yoep_min_year),
                    'min_income': args.min_income_effective,
                    'drop_top_income_pct': float(args.drop_top_income_pct),
                    'acs_state': args.acs_state,
                },
            )

    # Save results (income-scale runs -> paper-results/acs/{suite}/results/)
    alpha_str = alpha_to_tag(config['alpha'])  # e.g., "alpha10" or "alpha07p5"
    predictor_for_output = config.get('predictor', args.predictor)
    outcome_scale_for_output = config.get('outcome_scale', args.outcome_scale)
    within_group_for_output = bool(config.get('within_group', not args.no_within_group))
    design_for_output = config.get('design', design)
    drop_top_for_output = float(
        config.get('drop_top_income_fraction', args.drop_top_income_pct)
    )
    permute_rows_for_output = bool(config.get('permute_rows', args.permute_rows))
    min_income_for_output = (
        cohort.get('min_income', args.min_income_effective)
        if args.recompute_from_cache else args.min_income_effective
    )
    if min_income_for_output is not None and float(min_income_for_output) <= 0:
        min_income_for_output = None
    yoep_for_output = int(
        cohort.get('yoep_min_year', args.yoep_min_year)
        if args.recompute_from_cache else args.yoep_min_year
    )
    output_dir = _acs_output_dir(
        permuted=permute_rows_for_output,
        alpha_str=alpha_str,
        target_index=TARGET_INDEX,
        predictor=predictor_for_output,
        within_group=within_group_for_output,
        outcome_scale=outcome_scale_for_output,
        within_group_mode=within_group_mode_for_output,
        drop_top_income_fraction=drop_top_for_output,
        yoep_min_year=yoep_for_output,
        min_income=min_income_for_output,
        design=design_for_output,
        stdcp_center=str(config.get('stdcp_center', getattr(args, 'stdcp_center', 'local'))),
        score_type=str(config.get('score_type', getattr(args, 'score_type', 'absolute'))),
        stdcp_score_type=str(
            config.get(
                'stdcp_score_type',
                getattr(args, 'stdcp_score_type', None)
                or getattr(args, 'score_type', 'absolute'),
            )
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f'acs_true_marg_{alpha_str}_detailed.csv'
    df_results = save_results_to_csv(results, output_csv, global_pooled=global_pooled)

    summary_dir = output_dir / 'summaries'
    df_summary = save_summaries(df_results, summary_dir)
    save_endpoints_analysis(df_results, global_pooled, summary_dir, o_compare=20)

    if len(df_summary):
        print("\n  Coverage summary (mean):")
        for method in ['Donor-HCP', 'Pooled-Income-Quantile', 'HCP']:
            sub = df_summary[df_summary['method'] == method]
            if len(sub) == 0:
                continue
            print(f"    {method}:")
            for _, row in sub.sort_values('o').iterrows():
                print(
                    f"      o={row['o']}: {row['coverage_mean']:.3f} "
                    f"(SE {row['coverage_se']:.4f}), "
                    f"width_income median={row['width_income_median']:,.0f}"
                )
        dhcp20 = df_summary[(df_summary['method'] == 'Donor-HCP') & (df_summary['o'] == 20)]
        if len(dhcp20):
            w_d = float(dhcp20['width_income_median'].iloc[0])
            w_p = global_pooled['width']
            if np.isfinite(w_d) and w_p > 0:
                pct = 100 * (1 - w_d / w_p) if w_d < w_p else -100 * (w_d / w_p - 1)
                cmp_word = "narrower" if w_d < w_p else "wider"
                print(
                    f"\n  Width: GHCP median @ o=20 = ${w_d:,.0f}; "
                    f"global pooled = ${w_p:,.0f} ({abs(pct):.1f}% {cmp_word})"
                )

    print("\nDone!")
    print(f"Results saved to: {output_dir}")
