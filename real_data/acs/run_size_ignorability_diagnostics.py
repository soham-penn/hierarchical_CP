#!/usr/bin/env python3
"""
ACS group-size ignorability diagnostics (observable proxies only).

The paper's size-ignorability condition is N_j ⟂ (U_j, μ_j): retained PUMA size
should carry no information about the PUMA-specific covariate/outcome distribution.
Filtering on foreign-born, recent entry, age, and hours does not establish this.

This script reports falsification checks using observable group summaries:
  (A) Y summaries: mean(Y_j), sd(Y_j)
  (B) X summaries: group-level covariate means/shares
  (C) Residual summaries from the same pooled global RF as the ACS experiment

Equal-size subsampling removes mechanical precision differences across groups.
Stratified and size-band comparisons assess sensitivity when N_j varies widely.

Uses the same cohort filters as code/marginal/run_acs_yoep_fb_min21_permute.py:
  CA, foreign-born, YOEP >= 2000, age 25-54, hours >= 40, min PUMA size 21,
  income scale (no trim), exclude entry_recency and class-of-worker from X.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
REAL_DATA_DIR = REPO_ROOT / "real_data"
DEFAULT_ACS_CSV = REAL_DATA_DIR / "acs" / "data" / "acs_data_all50states.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REAL_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DATA_DIR))

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums  # noqa: E402
from code.paths import PLOTS_MARGINAL  # noqa: E402
from methods.mu_methods import (  # noqa: E402
    create_mu_method_ols_global_only,
    create_mu_method_random_forest_global_only,
)

DEFAULT_OUTPUT_DIR = PLOTS_MARGINAL / "acs" / "diagnostics"

# Match run_acs_experiments.py RF defaults.
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123

GROUP_COL = "puma"
CONTINUOUS_COVARIATES = ("age", "hours", "entry_recency", "age_sq")
CATEGORICAL_COVARIATES = ("educ_level", "female", "married", "english", "cow")
Y_SUMMARY_NAMES = ("y_mean", "y_sd", "income_mean", "income_sd")
RESIDUAL_SUMMARY_NAMES = (
    "resid_mean",
    "resid_sd",
    "resid_q25",
    "resid_median",
    "resid_q75",
)


@dataclass(frozen=True)
class Cohort:
    df: pd.DataFrame
    X: np.ndarray
    x_feature_names: list[str]
    y: np.ndarray
    group_sizes: pd.Series
    eligible_groups: np.ndarray
    exclude_cow: bool = True


def _outcome_transform(outcome_scale: str):
    if outcome_scale == "income":
        return lambda x: np.asarray(x, dtype=float)
    if outcome_scale == "log1p":
        return np.log1p
    raise ValueError(f"Unknown outcome_scale {outcome_scale!r}")


def load_acs_cohort(
    *,
    acs_csv: Path,
    acs_state: str,
    outcome_scale: str,
    drop_top_income_pct: float,
    min_puma_size: int,
    yoep_min_year: int,
    min_income: float | None,
    exclude_entry_recency: bool,
    exclude_cow: bool,
) -> Cohort:
    """Same ACS filtering path as run_acs_experiments.py / min21 suite."""
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=[acs_state],
        age_min=25,
        age_max=54,
        yoep_min_year=yoep_min_year,
        min_hours=40,
        min_income=min_income,
        drop_top_income_fraction=drop_top_income_pct if drop_top_income_pct > 0 else None,
        y_transform=_outcome_transform(outcome_scale),
    )
    if GROUP_COL not in df.columns:
        raise ValueError(f"Missing grouping column {GROUP_COL!r} after ACS cleaning.")
    df = df.dropna(subset=[GROUP_COL]).copy()
    df[GROUP_COL] = df[GROUP_COL].astype(int)

    X, x_names = _build_design_matrix_with_names(
        df,
        exclude_entry_recency=exclude_entry_recency,
        exclude_cow=exclude_cow,
    )
    y = df["y"].to_numpy(dtype=float)
    counts = df.groupby(GROUP_COL).size()
    eligible = counts[counts >= min_puma_size].index.to_numpy(dtype=int)
    if len(eligible) == 0:
        raise ValueError(f"No groups with size >= {min_puma_size}.")

    return Cohort(
        df=df,
        X=X,
        x_feature_names=x_names,
        y=y,
        group_sizes=counts.loc[eligible].astype(int),
        eligible_groups=eligible,
        exclude_cow=exclude_cow,
    )


def _build_design_matrix_with_names(
    df: pd.DataFrame,
    *,
    exclude_entry_recency: bool = False,
    exclude_cow: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Mirror build_design_matrix_acs but return column names for labeling."""
    educ_dummies = pd.get_dummies(df["educ_level"], prefix="educ", drop_first=True)
    english_dummies = pd.get_dummies(df["english"], prefix="eng", drop_first=True)
    continuous_cols = ["age", "age_sq", "hours", "married", "female"]
    if not exclude_entry_recency:
        continuous_cols.insert(3, "entry_recency")
    parts = [df[continuous_cols], educ_dummies, english_dummies]
    if not exclude_cow:
        parts.append(pd.get_dummies(df["cow"], prefix="cow", drop_first=True))
    design = pd.concat(parts, axis=1)
    X = build_design_matrix_acs(
        df,
        exclude_entry_recency=exclude_entry_recency,
        exclude_cow=exclude_cow,
    )
    return X, list(design.columns)


def _make_global_predictor(predictor: str):
    if predictor == "ols":
        return create_mu_method_ols_global_only()
    if predictor == "rf":
        return create_mu_method_random_forest_global_only(
            ntree=RF_NTREE,
            nodesize=RF_NODESIZE,
            random_state=RF_RANDOM_STATE,
        )
    raise ValueError(f"Unknown predictor {predictor!r}; expected 'ols' or 'rf'.")


def fit_pooled_global_model(cohort: Cohort, predictor: str):
    """Fit pooled global mu on all eligible individuals (no group ID in features)."""
    mu_method = _make_global_predictor(predictor)
    groups = cohort.eligible_groups
    Z_list = []
    for grp in groups:
        mask = cohort.df[GROUP_COL].values == grp
        idx = np.where(mask)[0]
        Z_list.append([{"X": cohort.X[i], "Y": cohort.y[i]} for i in idx])

    U_matrix = np.zeros((len(groups), 1))
    train_groups = list(range(len(groups)))
    model = mu_method["fit_global"](U_matrix, Z_list, train_groups)
    if model is None:
        raise RuntimeError("Pooled global model fit returned None.")
    return mu_method, model


def predict_all(mu_method, model, cohort: Cohort) -> np.ndarray:
    """Pooled global predictions for every row (U = 0)."""
    u_zero = np.zeros(1)
    preds = np.empty(len(cohort.df), dtype=float)
    for i in range(len(cohort.df)):
        preds[i] = mu_method["predict_global"](model, cohort.X[i], u_zero)
    return preds


def _proportion(series: pd.Series, value) -> float:
    if len(series) == 0:
        return np.nan
    return float((series == value).mean())


def compute_y_group_summaries(df_sub: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    y_vals = df_sub["y"].astype(float)
    out["y_mean"] = float(y_vals.mean())
    out["y_sd"] = float(y_vals.std(ddof=1)) if len(y_vals) > 1 else 0.0
    if "income" in df_sub.columns:
        inc = df_sub["income"].astype(float)
        out["income_mean"] = float(inc.mean())
        out["income_sd"] = float(inc.std(ddof=1)) if len(inc) > 1 else 0.0
    return out


def compute_x_group_summaries(
    df_sub: pd.DataFrame,
    *,
    include_cow: bool = True,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in CONTINUOUS_COVARIATES:
        if col not in df_sub.columns:
            continue
        vals = df_sub[col].astype(float)
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    if "female" in df_sub.columns:
        out["prop_female"] = _proportion(df_sub["female"], 1)
    if "married" in df_sub.columns:
        out["prop_married"] = _proportion(df_sub["married"], 1)

    if "educ_level" in df_sub.columns:
        for level in df_sub["educ_level"].dropna().unique():
            out[f"prop_educ_{level}"] = _proportion(df_sub["educ_level"], level)

    if "english" in df_sub.columns:
        out["english_mean"] = float(df_sub["english"].astype(float).mean())

    if include_cow and "cow" in df_sub.columns:
        for level in sorted(df_sub["cow"].dropna().unique()):
            out[f"prop_cow_{level}"] = _proportion(df_sub["cow"], level)

    return out


def compute_residual_group_summaries(residuals: np.ndarray) -> dict[str, float]:
    if len(residuals) == 0:
        return {k: np.nan for k in RESIDUAL_SUMMARY_NAMES}
    q25, med, q75 = np.quantile(residuals, [0.25, 0.5, 0.75])
    return {
        "resid_mean": float(np.mean(residuals)),
        "resid_sd": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "resid_q25": float(q25),
        "resid_median": float(med),
        "resid_q75": float(q75),
    }


def _align_summary_columns(rows: list[dict[str, float]]) -> tuple[pd.DataFrame, list[str]]:
    all_keys = sorted({k for row in rows for k in row})
    mat = pd.DataFrame([{k: row.get(k, np.nan) for k in all_keys} for row in rows])
    return mat, all_keys


def compute_full_group_table(
    cohort: Cohort,
    residuals: np.ndarray,
) -> pd.DataFrame:
    """Per-PUMA summaries on the full filtered cohort (no subsampling)."""
    rows = []
    for grp in cohort.eligible_groups:
        mask = cohort.df[GROUP_COL].values == grp
        df_g = cohort.df.loc[mask]
        y_summ = compute_y_group_summaries(df_g)
        x_summ = compute_x_group_summaries(df_g, include_cow=not cohort.exclude_cow)
        r_summ = compute_residual_group_summaries(residuals[mask])
        row = {
            GROUP_COL: int(grp),
            "N_j": int(cohort.group_sizes[grp]),
            **y_summ,
            **x_summ,
            **r_summ,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COL).reset_index(drop=True)


def subsample_equal_size_indices(
    cohort: Cohort,
    m0: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Sample m0 row positions without replacement from each eligible group."""
    picks: dict[int, np.ndarray] = {}
    puma = cohort.df[GROUP_COL].values
    for grp in cohort.eligible_groups:
        idx = np.where(puma == grp)[0]
        if len(idx) < m0:
            raise ValueError(f"Group {grp} has size {len(idx)} < m0={m0}.")
        picks[int(grp)] = rng.choice(idx, size=m0, replace=False)
    return picks


def summaries_from_subsample(
    cohort: Cohort,
    picks: dict[int, np.ndarray],
    residuals: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    y_rows, x_rows, r_rows = [], [], []
    for grp in cohort.eligible_groups:
        idx = picks[int(grp)]
        df_g = cohort.df.iloc[idx]
        y_rows.append(compute_y_group_summaries(df_g))
        x_rows.append(compute_x_group_summaries(df_g, include_cow=not cohort.exclude_cow))
        r_rows.append(compute_residual_group_summaries(residuals[idx]))

    y_mat, y_names = _align_summary_columns(y_rows)
    x_mat, x_names = _align_summary_columns(x_rows)
    r_mat, r_names = _align_summary_columns(r_rows)
    for mat in (y_mat, x_mat, r_mat):
        mat.insert(0, GROUP_COL, cohort.eligible_groups)
        n_j = cohort.group_sizes.loc[cohort.eligible_groups].to_numpy(dtype=float)
        mat.insert(1, "N_j", n_j)
    return y_mat, x_mat, r_mat, y_names, x_names, r_names


def spearman_vector(n_j: np.ndarray, summary_mat: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        s = summary_mat[name].to_numpy(dtype=float)
        if np.allclose(s, s[0]):
            rho, pval = 0.0, 1.0
        else:
            rho, pval = spearmanr(n_j, s)
        rows.append({"summary": name, "spearman_rho": float(rho), "p_value": float(pval)})
    return pd.DataFrame(rows)


def omnibus_statistic(n_j: np.ndarray, summary_mat: pd.DataFrame, names: Iterable[str]) -> float:
    total = 0.0
    for name in names:
        s = summary_mat[name].to_numpy(dtype=float)
        if len(s) < 2 or np.allclose(s, s[0], equal_nan=True):
            continue
        rho, _ = spearmanr(n_j, s)
        if np.isfinite(rho):
            total += float(rho) ** 2
    return total


def omnibus_permutation_pvalue(
    n_j: np.ndarray,
    summary_mat: pd.DataFrame,
    names: Iterable[str],
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    obs = omnibus_statistic(n_j, summary_mat, names)
    count = 0
    for _ in range(n_perm):
        perm_n = rng.permutation(n_j)
        if omnibus_statistic(perm_n, summary_mat, names) >= obs:
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return obs, pval


def run_subsample_diagnostics(
    cohort: Cohort,
    residuals: np.ndarray,
    *,
    m0: int,
    B_sub: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y_spear_rows: list[dict] = []
    x_spear_rows: list[dict] = []
    r_spear_rows: list[dict] = []
    omni_rows: list[dict] = []

    for b in range(B_sub):
        picks = subsample_equal_size_indices(cohort, m0, rng)
        y_mat, x_mat, r_mat, y_names, x_names, r_names = summaries_from_subsample(
            cohort, picks, residuals
        )
        n_j = x_mat["N_j"].to_numpy(dtype=float)

        for label, mat, names, store in (
            ("y_summaries", y_mat, y_names, y_spear_rows),
            ("x_summaries", x_mat, x_names, x_spear_rows),
            ("residual_summaries", r_mat, r_names, r_spear_rows),
        ):
            sp = spearman_vector(n_j, mat, names)
            sp["subsample"] = b
            sp["test"] = label
            store.extend(sp.to_dict("records"))
            stat, pval = omnibus_permutation_pvalue(n_j, mat, names, n_perm=n_perm, rng=rng)
            omni_rows.append(
                {
                    "subsample": b,
                    "test": label,
                    "omnibus_stat": stat,
                    "perm_p_value": pval,
                }
            )

    return (
        pd.DataFrame(y_spear_rows),
        pd.DataFrame(x_spear_rows),
        pd.DataFrame(r_spear_rows),
        pd.DataFrame(omni_rows),
    )


def aggregate_spearman(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("summary", as_index=False)
        .agg(
            median_rho=("spearman_rho", "median"),
            median_p_value=("p_value", "median"),
            frac_p_lt_0p05=("p_value", lambda s: float(np.mean(s < 0.05))),
            mean_rho=("spearman_rho", "mean"),
        )
        .sort_values("median_rho", key=lambda s: s.abs(), ascending=False)
    )


def aggregate_omnibus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test, g in df.groupby("test"):
        stats = g["omnibus_stat"].to_numpy(dtype=float)
        pvals = g["perm_p_value"].to_numpy(dtype=float)
        rows.append(
            {
                "test": test,
                "median_omnibus_stat": float(np.nanmedian(stats)),
                "median_perm_p_value": float(np.nanmedian(pvals)),
                "frac_perm_p_lt_0p05": float(np.nanmean(pvals < 0.05)),
                "mean_omnibus_stat": float(np.nanmean(stats)),
            }
        )
    return pd.DataFrame(rows)


def run_stratified_tertile_comparison(group_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compare group summaries between low vs high N_j tertiles among eligible PUMAs.

    Large mean differences suggest size is associated with composition even when
    income is not in the filter.
    """
    summary_cols = [
        c
        for c in group_table.columns
        if c not in (GROUP_COL, "N_j")
        and group_table[c].dtype != object
    ]
    tert = pd.qcut(group_table["N_j"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    rows = []
    for col in summary_cols:
        low = group_table.loc[tert == "low", col].astype(float)
        high = group_table.loc[tert == "high", col].astype(float)
        if len(low) < 2 or len(high) < 2:
            continue
        rows.append(
            {
                "summary": col,
                "low_tertile_mean": float(low.mean()),
                "high_tertile_mean": float(high.mean()),
                "high_minus_low": float(high.mean() - low.mean()),
                "low_n_pumas": int(len(low)),
                "high_n_pumas": int(len(high)),
            }
        )
    return pd.DataFrame(rows).sort_values("high_minus_low", key=lambda s: s.abs(), ascending=False)


def run_size_band_spearman(group_table: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman rho between N_j and summaries within narrow size bands.

    Restricting to similar N_j reduces confounding from precision; persistent
    association within bands is more concerning for ignorability.
    """
    bands = [
        ("21-35", 21, 35),
        ("36-60", 36, 60),
        ("61+", 61, int(group_table["N_j"].max())),
    ]
    key_summaries = [
        c
        for c in ["y_mean", "y_sd", "age_mean", "hours_mean", "prop_female", "resid_sd"]
        if c in group_table.columns
    ]
    rows = []
    for band_name, lo, hi in bands:
        sub = group_table[(group_table["N_j"] >= lo) & (group_table["N_j"] <= hi)]
        n_j = sub["N_j"].to_numpy(dtype=float)
        if len(sub) < 8 or len(np.unique(n_j)) < 4:
            continue
        for col in key_summaries:
            s = sub[col].to_numpy(dtype=float)
            if np.allclose(s, s[0]):
                rho, pval = 0.0, 1.0
            else:
                rho, pval = spearmanr(n_j, s)
            rows.append(
                {
                    "size_band": band_name,
                    "n_pumas": len(sub),
                    "summary": col,
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_group_size_histogram(group_sizes: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(group_sizes.values, bins=20, color="#4472C4", edgecolor="white")
    ax.set_xlabel("PUMA size $N_j$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of eligible PUMA sizes")
    _save_figure(fig, out_dir / "group_size_histogram")


def plot_scatter_grid(
    table: pd.DataFrame,
    y_cols: list[str],
    *,
    title: str,
    out_path: Path,
) -> None:
    n = len(y_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    x = table["N_j"].to_numpy(dtype=float)
    for ax, col in zip(axes, y_cols, strict=False):
        y = table[col].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.75, edgecolors="k", linewidths=0.4)
        if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            rho, _ = spearmanr(x, y)
            ax.set_title(f"{col}\nSpearman ρ={rho:.2f}")
        else:
            ax.set_title(col)
        ax.set_xlabel("$N_j$")
    for ax in axes[len(y_cols):]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save_figure(fig, out_path)


def plot_spearman_heatmap(agg: pd.DataFrame, title: str, out_path: Path) -> None:
    names = agg["summary"].tolist()
    vals = agg["median_rho"].to_numpy(dtype=float).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(names))))
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["median Spearman ρ"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    _save_figure(fig, out_path)


def _format_table_rows(df: pd.DataFrame, n: int = 5) -> list[str]:
    lines = []
    for _, row in df.head(n).iterrows():
        lines.append(
            f"| {row['summary']} | {row['median_rho']:.3f} | "
            f"{row['median_p_value']:.4f} | {row['frac_p_lt_0p05']:.3f} |"
        )
    return lines


def write_markdown_report(
    path: Path,
    *,
    cohort: Cohort,
    predictor: str,
    outcome_scale: str,
    yoep_min_year: int,
    m0: int,
    B_sub: int,
    y_agg: pd.DataFrame,
    x_agg: pd.DataFrame,
    r_agg: pd.DataFrame,
    omni_agg: pd.DataFrame,
    tertile_cmp: pd.DataFrame,
    band_spear: pd.DataFrame,
) -> None:
    def _omni_row(test: str) -> pd.Series:
        return omni_agg[omni_agg["test"] == test].iloc[0]

    y_omni = _omni_row("y_summaries")
    x_omni = _omni_row("x_summaries")
    r_omni = _omni_row("residual_summaries")

    lines = [
        "# ACS group-size ignorability diagnostics",
        "",
        "## Scope (read this first)",
        "",
        "The real-data ACS study is an **empirical illustration under an approximate",
        "working model**, not a test of the theoretical size-ignorability assumption",
        "`N_j ⟂ (U_j, μ_j)`.",
        "",
        "That assumption requires retained PUMA size to carry **no information** about",
        "the PUMA-specific covariate and outcome distribution. Our filters (foreign-born,",
        "recent entry, age, full-time hours) do **not** depend on income, but PUMAs with",
        "many retained individuals may still differ systematically in occupation, education,",
        "language, housing markets, and income dispersion.",
        "",
        "These diagnostics relate observable group summaries to `N_j` using the same",
        f"cohort and pooled global {predictor.upper()} predictor as the paper ACS runs.",
        "",
        "## What is checked",
        "",
        "1. **Y summaries:** mean(Y_j), sd(Y_j), and raw income moments",
        "2. **X summaries:** group-level covariate means and category shares",
        "3. **Residual summaries:** group-level moments of Y − μ_global(X)",
        "",
        "Equal-size subsampling (`m0` per PUMA) within each replicate removes mechanical",
        "precision differences. Stratified tertile and size-band analyses assess sensitivity.",
        "",
        "## Sample construction",
        "",
        f"- Grouping: `{GROUP_COL}`; eligible PUMAs: {len(cohort.eligible_groups)} (N_j ≥ {int(cohort.group_sizes.min())})",
        f"- Individuals in eligible PUMAs: {int(cohort.group_sizes.sum())}",
        f"- N_j range: {int(cohort.group_sizes.min())}–{int(cohort.group_sizes.max())}",
        f"- YOEP ≥ {yoep_min_year}; outcome scale: `{outcome_scale}`",
        f"- Subsample m0={m0}; replicates B_sub={B_sub}",
        "",
        "## Omnibus permutation tests (median across subsamples)",
        "",
        f"- Y summaries: median stat={y_omni['median_omnibus_stat']:.4f}, "
        f"median perm p={y_omni['median_perm_p_value']:.4f}, "
        f"frac p<0.05={y_omni['frac_perm_p_lt_0p05']:.3f}",
        f"- X summaries: median stat={x_omni['median_omnibus_stat']:.4f}, "
        f"median perm p={x_omni['median_perm_p_value']:.4f}, "
        f"frac p<0.05={x_omni['frac_perm_p_lt_0p05']:.3f}",
        f"- Residual summaries: median stat={r_omni['median_omnibus_stat']:.4f}, "
        f"median perm p={r_omni['median_perm_p_value']:.4f}, "
        f"frac p<0.05={r_omni['frac_perm_p_lt_0p05']:.3f}",
        "",
        "Omnibus statistic: sum of squared Spearman ρ; p-value from permuting N_j across PUMAs.",
        "",
        "## Strongest univariate associations (median Spearman across subsamples)",
        "",
        "### Y summaries",
        "",
        "| summary | median ρ | median p | frac p<0.05 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(_format_table_rows(y_agg))
    lines += [
        "",
        "### X summaries",
        "",
        "| summary | median ρ | median p | frac p<0.05 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(_format_table_rows(x_agg))
    lines += [
        "",
        "### Residual summaries",
        "",
        "| summary | median ρ | median p | frac p<0.05 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(_format_table_rows(r_agg))

    if not tertile_cmp.empty:
        top = tertile_cmp.head(5)
        lines += [
            "",
            "## Stratified sensitivity (high vs low N_j tertile means)",
            "",
            "| summary | low tertile mean | high tertile mean | high − low |",
            "|---|---:|---:|---:|",
        ]
        for _, row in top.iterrows():
            lines.append(
                f"| {row['summary']} | {row['low_tertile_mean']:.4g} | "
                f"{row['high_tertile_mean']:.4g} | {row['high_minus_low']:.4g} |"
            )

    if not band_spear.empty:
        lines += [
            "",
            "## Size-band Spearman (within similar N_j)",
            "",
            "| band | summary | ρ | p | n PUMAs |",
            "|---|---|---:|---:|---:|",
        ]
        for _, row in band_spear.iterrows():
            lines.append(
                f"| {row['size_band']} | {row['summary']} | {row['spearman_rho']:.3f} | "
                f"{row['p_value']:.4f} | {int(row['n_pumas'])} |"
            )

    lines += [
        "",
        "## Interpretation for the paper",
        "",
        "- **Rejecting ignorability outright is not possible** from observables alone.",
        "- **Non-trivial association between N_j and Y/X/residual summaries** indicates",
        "  PUMA size may proxy unmodeled heterogeneity; report alongside coverage results.",
        "- **Weak association is supportive but not definitive**; latent group structure",
        "  not captured by these summaries would remain undetected.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACS group-size ignorability diagnostics (paper min21 cohort)."
    )
    parser.add_argument("--acs_csv", type=Path, default=DEFAULT_ACS_CSV)
    parser.add_argument("--acs_state", type=str, default="CA")
    parser.add_argument(
        "--outcome_scale",
        choices=("income", "log1p"),
        default="income",
        help="Match ACS experiment outcome scale (paper uses income).",
    )
    parser.add_argument(
        "--predictor",
        choices=("ols", "rf"),
        default="rf",
        help="Pooled global predictor (paper ACS uses RF).",
    )
    parser.add_argument(
        "--drop_top_income_pct",
        type=float,
        default=0.0,
        help="Drop top income fraction per state (paper runs use 0).",
    )
    parser.add_argument("--min_puma_size", type=int, default=21)
    parser.add_argument("--yoep_min_year", type=int, default=2000)
    parser.add_argument(
        "--min_income",
        type=float,
        default=0.0,
        help="Minimum income floor (0 = no floor, matching paper).",
    )
    parser.add_argument(
        "--exclude_entry_recency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exclude_cow",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--m0",
        type=int,
        default=None,
        help="Common subsample size per group (default: min eligible group size).",
    )
    parser.add_argument("--B_sub", type=int, default=200, help="Subsample replicates.")
    parser.add_argument("--n_perm", type=int, default=499, help="Permutations for omnibus test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    plot_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    min_income = None if args.min_income <= 0 else float(args.min_income)

    print("Loading ACS cohort (paper min21 filters)...")
    cohort = load_acs_cohort(
        acs_csv=args.acs_csv,
        acs_state=args.acs_state,
        outcome_scale=args.outcome_scale,
        drop_top_income_pct=args.drop_top_income_pct,
        min_puma_size=args.min_puma_size,
        yoep_min_year=args.yoep_min_year,
        min_income=min_income,
        exclude_entry_recency=args.exclude_entry_recency,
        exclude_cow=args.exclude_cow,
    )
    m0 = int(args.m0) if args.m0 is not None else int(cohort.group_sizes.min())
    if m0 > int(cohort.group_sizes.min()):
        raise ValueError(
            f"m0={m0} exceeds minimum eligible group size {int(cohort.group_sizes.min())}."
        )

    print(
        f"  {len(cohort.eligible_groups)} eligible PUMAs; "
        f"N_j in [{cohort.group_sizes.min()}, {cohort.group_sizes.max()}]; m0={m0}"
    )

    print(f"Fitting pooled global {args.predictor} predictor...")
    mu_method, model = fit_pooled_global_model(cohort, args.predictor)
    residuals = cohort.y - predict_all(mu_method, model, cohort)

    group_table = compute_full_group_table(cohort, residuals)
    group_table.to_csv(out_dir / "group_sizes_and_group_summaries.csv", index=False)

    tertile_cmp = run_stratified_tertile_comparison(group_table)
    tertile_cmp.to_csv(out_dir / "stratified_tertile_comparison.csv", index=False)
    band_spear = run_size_band_spearman(group_table)
    band_spear.to_csv(out_dir / "size_band_spearman.csv", index=False)

    print(f"Running {args.B_sub} equal-size subsamples (m0={m0})...")
    y_spear, x_spear, r_spear, omni = run_subsample_diagnostics(
        cohort,
        residuals,
        m0=m0,
        B_sub=args.B_sub,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    y_spear.to_csv(out_dir / "y_summary_spearman_by_subsample.csv", index=False)
    x_spear.to_csv(out_dir / "x_summary_spearman_by_subsample.csv", index=False)
    r_spear.to_csv(out_dir / "residual_summary_spearman_by_subsample.csv", index=False)
    omni.to_csv(out_dir / "omnibus_test_by_subsample.csv", index=False)

    y_agg = aggregate_spearman(y_spear)
    x_agg = aggregate_spearman(x_spear)
    r_agg = aggregate_spearman(r_spear)
    omni_agg = aggregate_omnibus(omni)
    y_agg.to_csv(out_dir / "y_summary_spearman_results.csv", index=False)
    x_agg.to_csv(out_dir / "x_summary_spearman_results.csv", index=False)
    r_agg.to_csv(out_dir / "residual_summary_spearman_results.csv", index=False)
    omni_agg.to_csv(out_dir / "omnibus_test_results.csv", index=False)

    plot_group_size_histogram(cohort.group_sizes, plot_dir)
    y_scatter_cols = [c for c in Y_SUMMARY_NAMES if c in group_table.columns]
    x_scatter_cols = [
        c for c in ["age_mean", "hours_mean", "prop_female", "prop_married", "english_mean"]
        if c in group_table.columns
    ]
    r_scatter_cols = [c for c in RESIDUAL_SUMMARY_NAMES if c in group_table.columns]
    plot_scatter_grid(
        group_table,
        y_scatter_cols,
        title="PUMA size vs Y summaries (full cohort)",
        out_path=plot_dir / "scatter_Nj_vs_y_summaries",
    )
    plot_scatter_grid(
        group_table,
        x_scatter_cols,
        title="PUMA size vs selected X summaries (full cohort)",
        out_path=plot_dir / "scatter_Nj_vs_x_summaries",
    )
    plot_scatter_grid(
        group_table,
        r_scatter_cols,
        title="PUMA size vs residual summaries (full cohort)",
        out_path=plot_dir / "scatter_Nj_vs_residual_summaries",
    )
    plot_spearman_heatmap(
        y_agg,
        "Median Spearman ρ: N_j vs Y summaries",
        plot_dir / "heatmap_y_spearman",
    )
    plot_spearman_heatmap(
        x_agg,
        "Median Spearman ρ: N_j vs X summaries",
        plot_dir / "heatmap_x_spearman",
    )
    plot_spearman_heatmap(
        r_agg,
        "Median Spearman ρ: N_j vs residual summaries",
        plot_dir / "heatmap_residual_spearman",
    )

    write_markdown_report(
        out_dir / "diagnostics_report.md",
        cohort=cohort,
        predictor=args.predictor,
        outcome_scale=args.outcome_scale,
        yoep_min_year=args.yoep_min_year,
        m0=m0,
        B_sub=args.B_sub,
        y_agg=y_agg,
        x_agg=x_agg,
        r_agg=r_agg,
        omni_agg=omni_agg,
        tertile_cmp=tertile_cmp,
        band_spear=band_spear,
    )

    print("\n=== Size ignorability diagnostics summary ===")
    print(f"Output directory: {out_dir}")
    print(f"Eligible PUMAs: {len(cohort.eligible_groups)}; m0={m0}; B_sub={args.B_sub}")
    for _, row in omni_agg.iterrows():
        print(
            f"  {row['test']}: median omnibus stat={row['median_omnibus_stat']:.4f}, "
            f"median perm p={row['median_perm_p_value']:.4f}, "
            f"frac p<0.05={row['frac_perm_p_lt_0p05']:.3f}"
        )
    print("\nTop |median ρ| Y summaries:")
    for _, row in y_agg.head(3).iterrows():
        print(f"  {row['summary']}: median ρ={row['median_rho']:.3f}, median p={row['median_p_value']:.4f}")
    print("Top |median ρ| X summaries:")
    for _, row in x_agg.head(3).iterrows():
        print(f"  {row['summary']}: median ρ={row['median_rho']:.3f}, median p={row['median_p_value']:.4f}")
    print("Top |median ρ| residual summaries:")
    for _, row in r_agg.head(3).iterrows():
        print(f"  {row['summary']}: median ρ={row['median_rho']:.3f}, median p={row['median_p_value']:.4f}")
    print(f"\nReport: {out_dir / 'diagnostics_report.md'}")


if __name__ == "__main__":
    main()
