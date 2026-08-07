#!/usr/bin/env python3
"""
Standalone diagnostics for ACS group-size ignorability (observable proxies only).

Tests whether PUMA size N_j is associated with:
  (A) group-level summaries of covariates X, and
  (B) group-level summaries of pooled-global-model residuals.

This does NOT modify or replace the main ACS experiment code. It reuses the same
filtering, grouping (PUMA), outcome scale, and pooled global predictor (OLS/RF)
as ``code/marginal/run_acs_experiments.py``.

Conceptual target (latent, not directly testable):
    N_j ⟂ (latent group law of group j)

These are falsification / plausibility checks using observable implications only.
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
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "diagnostics_size_ignorability"
DEFAULT_ACS_CSV = REAL_DATA_DIR / "acs" / "data" / "acs_data_all50states.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REAL_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DATA_DIR))

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums  # noqa: E402
from methods.mu_methods import (  # noqa: E402
    create_mu_method_ols_global_only,
    create_mu_method_random_forest_global_only,
)

# Match run_acs_experiments.py RF defaults.
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123

GROUP_COL = "puma"
CONTINUOUS_COVARIATES = ("age", "hours", "entry_recency", "age_sq")
CATEGORICAL_COVARIATES = ("educ_level", "female", "married", "english", "cow")
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
    yoep_min_year: int = 2012,
    min_income: float | None = 10_000.0,
    exclude_entry_recency: bool = True,
    exclude_cow: bool = True,
) -> Cohort:
    """Same ACS filtering path as run_acs_experiments.py main()."""
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
    """
    Fit pooled global mu on all eligible individuals (no group ID in features).

    Uses the same mu_method factory as ACS baseline HCP (global-only OLS/RF).
    """
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


def compute_x_group_summaries(
    df_sub: pd.DataFrame,
    *,
    include_cow: bool = True,
) -> dict[str, float]:
    """Group-level X summaries on a (possibly subsampled) within-group slice."""
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
    """Per-group summaries on the full filtered cohort (no subsampling)."""
    rows = []
    for grp in cohort.eligible_groups:
        mask = cohort.df[GROUP_COL].values == grp
        df_g = cohort.df.loc[mask]
        x_summ = compute_x_group_summaries(df_g, include_cow=not cohort.exclude_cow)
        r_summ = compute_residual_group_summaries(residuals[mask])
        row = {
            GROUP_COL: int(grp),
            "N_j": int(cohort.group_sizes[grp]),
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    x_rows, r_rows = [], []
    for grp in cohort.eligible_groups:
        idx = picks[int(grp)]
        df_g = cohort.df.iloc[idx]
        x_rows.append(compute_x_group_summaries(df_g, include_cow=not cohort.exclude_cow))
        r_rows.append(compute_residual_group_summaries(residuals[idx]))

    x_mat, x_names = _align_summary_columns(x_rows)
    r_mat, r_names = _align_summary_columns(r_rows)
    x_mat.insert(0, GROUP_COL, cohort.eligible_groups)
    r_mat.insert(0, GROUP_COL, cohort.eligible_groups)
    n_j = cohort.group_sizes.loc[cohort.eligible_groups].to_numpy(dtype=float)
    x_mat.insert(1, "N_j", n_j)
    r_mat.insert(1, "N_j", n_j)
    return x_mat, r_mat, x_names, r_names


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
    """Sum of squared Spearman correlations across summary components."""
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
    """Permute N_j across groups; return (observed stat, two-sided p-value)."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x_spear_rows: list[dict] = []
    r_spear_rows: list[dict] = []
    omni_rows: list[dict] = []

    for b in range(B_sub):
        picks = subsample_equal_size_indices(cohort, m0, rng)
        x_mat, r_mat, x_names, r_names = summaries_from_subsample(cohort, picks, residuals)
        n_j = x_mat["N_j"].to_numpy(dtype=float)

        x_sp = spearman_vector(n_j, x_mat, x_names)
        r_sp = spearman_vector(n_j, r_mat, r_names)
        x_sp["subsample"] = b
        r_sp["subsample"] = b
        x_spear_rows.extend(x_sp.to_dict("records"))
        r_spear_rows.extend(r_sp.to_dict("records"))

        x_stat, x_p = omnibus_permutation_pvalue(n_j, x_mat, x_names, n_perm=n_perm, rng=rng)
        r_stat, r_p = omnibus_permutation_pvalue(n_j, r_mat, r_names, n_perm=n_perm, rng=rng)
        omni_rows.append(
            {
                "subsample": b,
                "test": "x_summaries",
                "omnibus_stat": x_stat,
                "perm_p_value": x_p,
            }
        )
        omni_rows.append(
            {
                "subsample": b,
                "test": "residual_summaries",
                "omnibus_stat": r_stat,
                "perm_p_value": r_p,
            }
        )

    x_df = pd.DataFrame(x_spear_rows)
    r_df = pd.DataFrame(r_spear_rows)
    omni_df = pd.DataFrame(omni_rows)
    return x_df, r_df, omni_df


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


def write_markdown_report(
    path: Path,
    *,
    cohort: Cohort,
    predictor: str,
    outcome_scale: str,
    m0: int,
    B_sub: int,
    x_agg: pd.DataFrame,
    r_agg: pd.DataFrame,
    omni_agg: pd.DataFrame,
) -> None:
    x_omni = omni_agg[omni_agg["test"] == "x_summaries"].iloc[0]
    r_omni = omni_agg[omni_agg["test"] == "residual_summaries"].iloc[0]
    top_x = x_agg.head(5)
    top_r = r_agg.head(5)

    lines = [
        "# ACS group-size ignorability diagnostics",
        "",
        "## What is being tested",
        "",
        "These are **observable diagnostics**, not direct tests of the latent assumption",
        "`N_j ⟂ (latent group law of group j)`.",
        "",
        "We check two observable implications on PUMA groups:",
        "",
        "1. **X-summary diagnostic:** is PUMA size `N_j` associated with group-level",
        "   summaries of covariates?",
        "2. **Residual-summary diagnostic:** after fitting the same pooled global predictor",
        f"   as the ACS experiment ({predictor}, outcome scale `{outcome_scale}`), is `N_j`",
        "   associated with group-level summaries of residuals?",
        "",
        "Equal-size subsampling (`m0` per group) is used within each subsample replicate",
        "so precision differences across groups do not mechanically drive associations.",
        "",
        "## Sample construction",
        "",
        f"- Grouping variable: `{GROUP_COL}`",
        f"- Eligible groups: {len(cohort.eligible_groups)} (size ≥ {cohort.group_sizes.min()})",
        f"- Total individuals in eligible groups: {int(cohort.group_sizes.sum())}",
        f"- PUMA size range: {int(cohort.group_sizes.min())}–{int(cohort.group_sizes.max())}",
        f"- Subsample size m0: {m0}; subsample replicates B_sub: {B_sub}",
        "",
        "## Census PUMA vs analytic `N_j` (read this first)",
        "",
        "Census PUMAs are **geographic areas** designed to contain ~100,000+ residents.",
        "They are **not** formed using education, income, or other covariates in this analysis.",
        "",
        "In this script, `N_j` is **not** total PUMA population. It is the number of ACS rows",
        "in the **heavily filtered analytic subpopulation** per PUMA (foreign-born, recent entry",
        "since 2012, age 25–54, full-time hours, income ≥ \\$10K, etc.). After filtering, most",
        "CA PUMAs contain very few such individuals (median ~9); only PUMAs with `N_j ≥ 21` enter",
        "the diagnostic (41 of 258 PUMAs with any data).",
        "",
        "So a strong `N_j` vs X association means: **PUMAs where more recent immigrants in our",
        "niche subpopulation happen to live also tend to have different covariate mixes**",
        "(e.g. higher BA+ share, better English scores). That reflects **spatial sorting /",
        "concentration of the subpopulation**, not how Census drew PUMA boundaries.",
        "",
        "## Interpretation",
        "",
        "- **Association between `N_j` and X summaries** suggests group size may depend on",
        "  group composition (observable covariate mix).",
        "- **Association between `N_j` and residual summaries** suggests group size may depend",
        "  on unexplained group-specific outcome behavior after controlling for X via the",
        "  pooled global model.",
        "- **Lack of strong association is supportive but not definitive** for size ignorability;",
        "  latent group heterogeneity not captured by these summaries would not be detected.",
        "",
        "## Omnibus permutation tests (median across subsamples)",
        "",
        f"- X summaries: median omnibus stat = {x_omni['median_omnibus_stat']:.4f}, "
        f"median perm p = {x_omni['median_perm_p_value']:.4f}, "
        f"fraction p<0.05 = {x_omni['frac_perm_p_lt_0p05']:.3f}",
        f"- Residual summaries: median omnibus stat = {r_omni['median_omnibus_stat']:.4f}, "
        f"median perm p = {r_omni['median_perm_p_value']:.4f}, "
        f"fraction p<0.05 = {r_omni['frac_perm_p_lt_0p05']:.3f}",
        "",
        "Omnibus statistic: sum of squared Spearman ρ across components; p-value from",
        "permuting `N_j` across groups.",
        "",
        "## Strongest univariate associations (median Spearman across subsamples)",
        "",
        "### X summaries",
        "",
        "| summary | median ρ | median p | frac p<0.05 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in top_x.iterrows():
        lines.append(
            f"| {row['summary']} | {row['median_rho']:.3f} | "
            f"{row['median_p_value']:.4f} | {row['frac_p_lt_0p05']:.3f} |"
        )
    lines += [
        "",
        "### Residual summaries",
        "",
        "| summary | median ρ | median p | frac p<0.05 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in top_r.iterrows():
        lines.append(
            f"| {row['summary']} | {row['median_rho']:.3f} | "
            f"{row['median_p_value']:.4f} | {row['frac_p_lt_0p05']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACS group-size ignorability diagnostics (observable proxies)."
    )
    parser.add_argument("--acs_csv", type=Path, default=DEFAULT_ACS_CSV)
    parser.add_argument("--acs_state", type=str, default="CA")
    parser.add_argument(
        "--outcome_scale",
        choices=("income", "log1p"),
        default="income",
        help="Match ACS experiment outcome scale (paper income runs use income).",
    )
    parser.add_argument(
        "--predictor",
        choices=("ols", "rf"),
        default="ols",
        help="Pooled global predictor (matches ACS baseline global mu).",
    )
    parser.add_argument(
        "--drop_top_income_pct",
        type=float,
        default=0.02,
        help="Drop top income fraction per state (paper runs use 0.02).",
    )
    parser.add_argument(
        "--min_puma_size",
        type=int,
        default=21,
        help="Minimum PUMA size for eligibility (matches ACS experiment default).",
    )
    parser.add_argument(
        "--yoep_min_year",
        type=int,
        default=2012,
        help="Minimum year of entry to US (matches ACS experiment default).",
    )
    parser.add_argument(
        "--min_income",
        type=float,
        default=10000.0,
        help="Minimum income floor (default 10000). Set to 0 to disable.",
    )
    parser.add_argument(
        "--exclude_entry_recency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop entry_recency (YOEP-derived) from pooled OLS/RF predictors (default: True).",
    )
    parser.add_argument(
        "--exclude_cow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop class-of-worker from predictors and X ignorability summaries (default: True).",
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

    print("Loading ACS cohort (same filters as run_acs_experiments.py)...")
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

    print(f"Fitting pooled global {args.predictor} predictor on all eligible individuals...")
    mu_method, model = fit_pooled_global_model(cohort, args.predictor)
    residuals = cohort.y - predict_all(mu_method, model, cohort)

    group_table = compute_full_group_table(cohort, residuals)
    group_table.to_csv(out_dir / "group_sizes_and_group_summaries.csv", index=False)

    print(f"Running {args.B_sub} equal-size subsamples (m0={m0})...")
    x_spear, r_spear, omni = run_subsample_diagnostics(
        cohort,
        residuals,
        m0=m0,
        B_sub=args.B_sub,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    x_spear.to_csv(out_dir / "x_summary_spearman_by_subsample.csv", index=False)
    r_spear.to_csv(out_dir / "residual_summary_spearman_by_subsample.csv", index=False)
    omni.to_csv(out_dir / "omnibus_test_by_subsample.csv", index=False)

    x_agg = aggregate_spearman(x_spear)
    r_agg = aggregate_spearman(r_spear)
    omni_agg = aggregate_omnibus(omni)
    x_agg.to_csv(out_dir / "x_summary_spearman_results.csv", index=False)
    r_agg.to_csv(out_dir / "residual_summary_spearman_results.csv", index=False)
    omni_agg.to_csv(out_dir / "omnibus_test_results.csv", index=False)

    # Plots
    plot_group_size_histogram(cohort.group_sizes, plot_dir)
    x_scatter_cols = [
        c for c in ["age_mean", "hours_mean", "entry_recency_mean", "prop_female", "prop_married"]
        if c in group_table.columns
    ]
    r_scatter_cols = [c for c in RESIDUAL_SUMMARY_NAMES if c in group_table.columns]
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
        x_agg,
        "Median Spearman ρ: N_j vs X summaries (across subsamples)",
        plot_dir / "heatmap_x_spearman",
    )
    plot_spearman_heatmap(
        r_agg,
        "Median Spearman ρ: N_j vs residual summaries (across subsamples)",
        plot_dir / "heatmap_residual_spearman",
    )

    write_markdown_report(
        out_dir / "diagnostics_report.md",
        cohort=cohort,
        predictor=args.predictor,
        outcome_scale=args.outcome_scale,
        m0=m0,
        B_sub=args.B_sub,
        x_agg=x_agg,
        r_agg=r_agg,
        omni_agg=omni_agg,
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
    print("\nTop |median ρ| X summaries:")
    for _, row in x_agg.head(3).iterrows():
        print(f"  {row['summary']}: median ρ={row['median_rho']:.3f}, median p={row['median_p_value']:.4f}")
    print("Top |median ρ| residual summaries:")
    for _, row in r_agg.head(3).iterrows():
        print(f"  {row['summary']}: median ρ={row['median_rho']:.3f}, median p={row['median_p_value']:.4f}")
    print(f"\nReport: {out_dir / 'diagnostics_report.md'}")


if __name__ == "__main__":
    main()
