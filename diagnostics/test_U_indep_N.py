#!/usr/bin/env python3
"""
Test whether group-level U is independent of group size N for eligible PUMAs.

This script treats each CSV row as one eligible PUMA (already aggregated). It tests
U ⟂ N using a permutation distance-covariance procedure, where U is a
multidimensional vector of numeric group-level summaries and N is group size.

The main inferential result is from the permutation distance-covariance test.
Feature-wise Spearman correlations, OLS, and random-forest prediction are included
as diagnostics only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation distance-covariance test of U independent of N"
    )
    parser.add_argument("--summary_csv", required=True, help="Path to PUMA-level summary CSV")
    parser.add_argument("--n_col", required=True, help="Column containing group size N")
    parser.add_argument("--id_col", required=True, help="Unique group identifier column")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument(
        "--exclude_cols",
        default="",
        help="Comma-separated extra columns to exclude from U",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=10000,
        help="Number of permutations for distance-covariance test",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--standardize",
        choices=["robust", "zscore"],
        default="robust",
        help="Standardization method for U",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=20,
        help="Minimum number of rows required after filtering",
    )
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")

    args = parser.parse_args()

    if args.n_permutations <= 0:
        raise ValueError("--n_permutations must be positive")
    if args.min_rows <= 1:
        raise ValueError("--min_rows must be at least 2")

    return args


def robust_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    was_1d = x.ndim == 1
    if was_1d:
        x = x.reshape(-1, 1)

    med = np.median(x, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    iqr = q75 - q25

    std = np.std(x, axis=0, ddof=0)
    scale = np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0))

    z = (x - med) / scale
    if was_1d:
        return z.ravel()
    return z


def zscore_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    was_1d = x.ndim == 1
    if was_1d:
        x = x.reshape(-1, 1)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=0)
    scale = np.where(std > 0, std, 1.0)

    z = (x - mean) / scale
    if was_1d:
        return z.ravel()
    return z


def load_and_prepare_data(args: argparse.Namespace) -> Dict[str, object]:
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_csv not found: {summary_path}")

    df = pd.read_csv(summary_path)

    if args.id_col not in df.columns:
        raise ValueError(f"id column not found: {args.id_col}")
    if args.n_col not in df.columns:
        raise ValueError(f"N column not found: {args.n_col}")

    if df[args.id_col].duplicated().any():
        dup_values = df.loc[df[args.id_col].duplicated(), args.id_col].astype(str).head(10)
        raise ValueError(
            "id_col must be unique. Example duplicate values: "
            + ", ".join(dup_values.tolist())
        )

    n_numeric = pd.to_numeric(df[args.n_col], errors="coerce")
    invalid_mask = df[args.n_col].notna() & n_numeric.isna()
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        raise ValueError(
            f"n_col contains {n_invalid} non-numeric, non-missing values: {args.n_col}"
        )

    keep_rows = n_numeric.notna()
    df_used = df.loc[keep_rows].copy()
    n_raw = n_numeric.loc[keep_rows].to_numpy(dtype=float)

    if len(df_used) < args.min_rows:
        raise ValueError(
            f"Only {len(df_used)} rows remain after dropping missing N; min_rows={args.min_rows}."
        )

    if np.any(n_raw < 0):
        raise ValueError("n_col contains negative values; expected non-negative sample sizes.")
    if np.std(n_raw, ddof=0) == 0:
        raise ValueError("n_col has zero variance; dependence testing is not possible.")

    exclude_cols = {args.id_col, args.n_col}
    if args.exclude_cols.strip():
        user_exclude = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
        missing_excludes = [c for c in user_exclude if c not in df_used.columns]
        if missing_excludes:
            raise ValueError(
                "Some --exclude_cols are not in the CSV: " + ", ".join(missing_excludes)
            )
        exclude_cols.update(user_exclude)

    numeric_cols = df_used.select_dtypes(include=[np.number]).columns.tolist()
    u_candidate_cols = [c for c in numeric_cols if c not in exclude_cols]
    if not u_candidate_cols:
        raise ValueError("No numeric candidate U columns found after exclusions.")

    # Drop zero-variance columns before imputation, using non-missing values.
    u_keep_cols = []
    for col in u_candidate_cols:
        n_unique = df_used[col].nunique(dropna=True)
        if n_unique > 1:
            u_keep_cols.append(col)

    if not u_keep_cols:
        raise ValueError("All candidate U columns had zero variance.")

    u_df_raw = df_used[u_keep_cols].copy()

    # Median imputation for missing U values.
    medians = u_df_raw.median(axis=0, skipna=True)
    if medians.isna().any():
        bad_cols = medians[medians.isna()].index.tolist()
        raise ValueError("Unable to impute columns with all-missing values: " + ", ".join(bad_cols))
    u_df_imputed = u_df_raw.fillna(medians)

    u_matrix_raw = u_df_imputed.to_numpy(dtype=float)
    if args.standardize == "robust":
        u_matrix_std = robust_standardize(u_matrix_raw)
    else:
        u_matrix_std = zscore_standardize(u_matrix_raw)

    if not np.all(np.isfinite(u_matrix_std)):
        raise ValueError("Non-finite values found in standardized U matrix.")

    n_std = zscore_standardize(n_raw)
    n_log_std = zscore_standardize(np.log1p(n_raw))

    return {
        "df_used": df_used,
        "ids": df_used[args.id_col].astype(str).to_numpy(),
        "n_raw": n_raw,
        "n_std": n_std,
        "n_log_std": n_log_std,
        "u_df_imputed": u_df_imputed,
        "u_matrix_std": u_matrix_std,
        "u_columns": u_keep_cols,
    }


def pairwise_distance_matrix_U(u_matrix: np.ndarray) -> np.ndarray:
    u_matrix = np.asarray(u_matrix, dtype=float)
    sq_norm = np.sum(u_matrix * u_matrix, axis=1, keepdims=True)
    dist_sq = sq_norm + sq_norm.T - 2.0 * (u_matrix @ u_matrix.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def pairwise_distance_matrix_scalar(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return np.abs(x[:, None] - x[None, :])


def double_center(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    row_means = d.mean(axis=1, keepdims=True)
    col_means = d.mean(axis=0, keepdims=True)
    grand_mean = d.mean()
    return d - row_means - col_means + grand_mean


def distance_covariance_statistic(a_centered: np.ndarray, b_centered: np.ndarray) -> float:
    return float(np.mean(a_centered * b_centered))


def permutation_test_dcov(
    a_centered: np.ndarray,
    n_vector: np.ndarray,
    n_permutations: int,
    seed: int,
) -> Tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)

    b_obs = pairwise_distance_matrix_scalar(n_vector)
    b_obs_centered = double_center(b_obs)
    observed_t = distance_covariance_statistic(a_centered, b_obs_centered)

    perm_stats = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_idx = rng.permutation(len(n_vector))
        b_perm = pairwise_distance_matrix_scalar(n_vector[perm_idx])
        b_perm_centered = double_center(b_perm)
        perm_stats[i] = distance_covariance_statistic(a_centered, b_perm_centered)

    p_value = (1.0 + float(np.sum(perm_stats >= observed_t))) / (n_permutations + 1.0)
    return observed_t, p_value, perm_stats


def run_featurewise_spearman(u_df_imputed: pd.DataFrame, n_raw: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for col in u_df_imputed.columns:
        rho, pval = spearmanr(u_df_imputed[col].to_numpy(dtype=float), n_raw)
        rows.append({"feature": col, "rho": float(rho), "p_value": float(pval)})

    diag = pd.DataFrame(rows)
    _, qvals, _, _ = multipletests(diag["p_value"].fillna(1.0).to_numpy(), method="fdr_bh")
    diag["q_value"] = qvals
    diag["abs_rho"] = diag["rho"].abs()
    diag = diag.sort_values(["q_value", "abs_rho"], ascending=[True, False]).reset_index(drop=True)
    return diag.drop(columns=["abs_rho"])


def run_ols_diagnostic(u_matrix_std: np.ndarray, n_std: np.ndarray) -> Dict[str, float]:
    x = sm.add_constant(u_matrix_std, has_constant="add")
    model = sm.OLS(n_std, x).fit()

    f_pvalue = model.f_pvalue
    if f_pvalue is None:
        f_pvalue = np.nan

    return {
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f_pvalue": float(f_pvalue),
    }


def run_rf_diagnostic(u_matrix_std: np.ndarray, n_std: np.ndarray, seed: int) -> Dict[str, float]:
    n_rows = len(n_std)
    n_splits = min(5, n_rows)
    if n_splits < 2:
        raise ValueError("Need at least 2 rows for RF cross-validation.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    scores = cross_val_score(rf, u_matrix_std, n_std, scoring="r2", cv=cv)

    return {
        "cv_r2_mean": float(np.mean(scores)),
        "cv_r2_std": float(np.std(scores, ddof=0)),
    }


def make_plots(
    outdir: Path,
    primary_stat: float,
    primary_perm_stats: np.ndarray,
    logn_stat: float,
    logn_perm_stats: np.ndarray,
    u_matrix_std: np.ndarray,
    u_df_imputed: pd.DataFrame,
    n_raw: np.ndarray,
    spearman_df: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(primary_perm_stats, bins=40, color="#7da8c7", alpha=0.85, edgecolor="black")
    ax.axvline(primary_stat, color="crimson", linestyle="--", linewidth=2, label="Observed T")
    ax.set_title("Permutation Null Distribution (Primary N_std)")
    ax.set_xlabel("Distance-covariance statistic")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "permutation_hist_primary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(logn_perm_stats, bins=40, color="#9ecb88", alpha=0.85, edgecolor="black")
    ax.axvline(logn_stat, color="crimson", linestyle="--", linewidth=2, label="Observed T")
    ax.set_title("Permutation Null Distribution (Sensitivity log1p(N))")
    ax.set_xlabel("Distance-covariance statistic")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "permutation_hist_logN.png", dpi=160)
    plt.close(fig)

    if u_matrix_std.shape[1] >= 2:
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(u_matrix_std)
    else:
        # If only one feature remains, create a second zero axis for plotting.
        pca_scores = np.column_stack([u_matrix_std[:, 0], np.zeros(u_matrix_std.shape[0])])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_scores[:, 0],
        pca_scores[:, 1],
        c=n_raw,
        cmap="viridis",
        s=55,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("PCA of U (colored by N)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("N (raw)")
    fig.tight_layout()
    fig.savefig(outdir / "pca_U_colored_by_N.png", dpi=160)
    plt.close(fig)

    top_features = spearman_df["feature"].head(6).tolist()
    n_top = len(top_features)
    ncols = 3
    nrows = int(np.ceil(n_top / ncols)) if n_top > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, feature in enumerate(top_features):
        ax = axes_flat[i]
        ax.scatter(
            u_df_imputed[feature].to_numpy(dtype=float),
            n_raw,
            alpha=0.8,
            s=36,
            edgecolor="black",
            linewidth=0.3,
        )
        feature_row = spearman_df.loc[spearman_df["feature"] == feature].iloc[0]
        ax.set_xlabel(feature)
        ax.set_ylabel("N (raw)")
        ax.set_title(
            f"{feature}\n"
            f"rho={feature_row['rho']:.3f}, q={feature_row['q_value']:.3g}"
        )
        ax.grid(alpha=0.25)

    for j in range(n_top, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Top Feature Diagnostics vs N (smallest FDR q-values)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "top_feature_vs_N.png", dpi=160)
    plt.close(fig)


def save_outputs(
    outdir: Path,
    summary: Dict[str, object],
    spearman_df: pd.DataFrame,
    u_columns: List[str],
    primary_perm_stats: np.ndarray,
    logn_perm_stats: np.ndarray,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "independence_test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    spearman_df.to_csv(outdir / "feature_spearman_diagnostics.csv", index=False)

    with (outdir / "selected_U_columns.txt").open("w", encoding="utf-8") as f:
        for col in u_columns:
            f.write(f"{col}\n")

    np.save(outdir / "permutation_stats_primary.npy", primary_perm_stats)
    np.save(outdir / "permutation_stats_logN.npy", logn_perm_stats)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    prepared = load_and_prepare_data(args)
    u_matrix_std = prepared["u_matrix_std"]
    u_df_imputed = prepared["u_df_imputed"]
    n_raw = prepared["n_raw"]
    n_std = prepared["n_std"]
    n_log_std = prepared["n_log_std"]
    u_columns = prepared["u_columns"]

    a_matrix = pairwise_distance_matrix_U(u_matrix_std)
    a_centered = double_center(a_matrix)

    primary_stat, primary_p, primary_perm_stats = permutation_test_dcov(
        a_centered=a_centered,
        n_vector=n_std,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    logn_stat, logn_p, logn_perm_stats = permutation_test_dcov(
        a_centered=a_centered,
        n_vector=n_log_std,
        n_permutations=args.n_permutations,
        seed=args.seed + 1,
    )

    spearman_df = run_featurewise_spearman(u_df_imputed=u_df_imputed, n_raw=n_raw)
    ols_diag = run_ols_diagnostic(u_matrix_std=u_matrix_std, n_std=n_std)
    rf_diag = run_rf_diagnostic(u_matrix_std=u_matrix_std, n_std=n_std, seed=args.seed)

    interpretation = {
        "primary_reject_at_0_05": bool(primary_p < 0.05),
        "logN_reject_at_0_05": bool(logn_p < 0.05),
    }

    summary = {
        "n_rows_used": int(len(n_raw)),
        "n_features_used": int(len(u_columns)),
        "n_col": args.n_col,
        "id_col": args.id_col,
        "u_columns": list(u_columns),
        "primary_test": {
            "statistic": float(primary_stat),
            "p_value": float(primary_p),
            "n_permutations": int(args.n_permutations),
        },
        "logN_sensitivity_test": {
            "statistic": float(logn_stat),
            "p_value": float(logn_p),
            "n_permutations": int(args.n_permutations),
        },
        "ols_diagnostic": {
            "r2": float(ols_diag["r2"]),
            "adj_r2": float(ols_diag["adj_r2"]),
            "f_pvalue": float(ols_diag["f_pvalue"]),
        },
        "rf_diagnostic": {
            "cv_r2_mean": float(rf_diag["cv_r2_mean"]),
            "cv_r2_std": float(rf_diag["cv_r2_std"]),
        },
        "interpretation": interpretation,
    }

    save_outputs(
        outdir=outdir,
        summary=summary,
        spearman_df=spearman_df,
        u_columns=u_columns,
        primary_perm_stats=primary_perm_stats,
        logn_perm_stats=logn_perm_stats,
    )

    if args.plot:
        make_plots(
            outdir=outdir,
            primary_stat=primary_stat,
            primary_perm_stats=primary_perm_stats,
            logn_stat=logn_stat,
            logn_perm_stats=logn_perm_stats,
            u_matrix_std=u_matrix_std,
            u_df_imputed=u_df_imputed,
            n_raw=n_raw,
            spearman_df=spearman_df,
        )

    print(f"PUMAs used: {len(n_raw)}")
    print(f"U features used: {len(u_columns)}")
    print("Final U columns:")
    for col in u_columns:
        print(f"  - {col}")
    print(f"Primary test p-value (N_std): {primary_p:.6g}")
    print(f"LogN sensitivity p-value: {logn_p:.6g}")

    if primary_p >= 0.05 and logn_p >= 0.05:
        print(
            "Interpretation: No strong evidence against U independent of N at alpha=0.05. "
            "This does not prove independence; it indicates lack of strong detected dependence."
        )
    else:
        print(
            "Interpretation: At least one permutation test p-value is small (alpha=0.05), "
            "which provides evidence against U independent of N."
        )

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
