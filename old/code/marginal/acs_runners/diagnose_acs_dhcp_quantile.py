#!/usr/bin/env python3
"""Diagnose Donor-HCP quantile / width vs o on ACS (permute vs no-permute)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code.marginal.run_acs_experiments import (  # noqa: E402
    TARGET_INDEX,
    _make_mu_baseline,
    _make_mu_hcp,
    _baseline_group_scores,
    _fit_baseline_score_aux,
    _baseline_interval,
    _result_record,
    _tau_override,
    make_quantile_seed,
    compute_hcp_interval_radius,
    get_hcp_train_cal_split,
    sample_calibration_and_target_uniform,
)
from methods.donor_hcp import compute_donor_hcp_randomized_interval  # noqa: E402


def _load_yoep_fb_cohort(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    xcols = [c for c in df.columns if c.startswith("x_")]
    X = df[xcols].to_numpy(dtype=float)
    return df, X


def _finite_scores(meta: list[dict]) -> np.ndarray:
    vals = []
    for m in meta:
        if m.get("is_inf"):
            continue
        y, mu = m.get("y"), m.get("mu")
        if y is None or mu is None:
            continue
        vals.append(abs(float(y) - float(mu)))
    return np.asarray(vals, dtype=float)


def _run_replicate(
    *,
    df,
    X,
    eligible_groups,
    replicate_idx: int,
    o_values: list[int],
    alpha: float,
    permute_rows: bool,
    seed: int = 456,
    quantile_base_seed: int = 456,
) -> pd.DataFrame:
    alpha_sel = 0.5
    quantile_mode = "deterministic"
    config = {
        "seed": seed,
        "alpha": alpha,
        "quantile_mode": quantile_mode,
        "quantile_base_seed": quantile_base_seed,
        "within_group": True,
        "permute_rows": permute_rows,
        "predictor": "rf",
        "within_group_mode": "mean",
        "score_type": "absolute",
        "outcome_scale": "income",
    }

    sel_rng = np.random.default_rng(seed + replicate_idx * 1009)
    group_counts = df.groupby("puma").size().to_dict()
    calib_groups, test_group = sample_calibration_and_target_uniform(
        eligible_groups,
        group_counts,
        selection_seed=seed + replicate_idx * 1009,
        n_calib_groups=20,
        min_target_size=21,
    )
    if calib_groups is None:
        raise RuntimeError(f"replicate {replicate_idx}: could not sample calib/target")

    row_rng = (
        np.random.default_rng(seed + replicate_idx * 1009 + 811)
        if permute_rows
        else None
    )
    group_data = {}
    for grp in calib_groups + [test_group]:
        grp_idx = np.where((df["puma"] == grp).values)[0]
        if row_rng is not None:
            grp_idx = row_rng.permutation(grp_idx)
        group_data[grp] = {
            "X": X[grp_idx],
            "Y": df.iloc[grp_idx]["y"].values,
            "income": df.iloc[grp_idx]["income"].values.astype(float),
        }

    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{"X": group_data[g]["X"][i], "Y": group_data[g]["Y"][i]}
         for i in range(len(group_data[g]["Y"]))]
        for g in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))
    sample_sizes = [len(z) for z in Z_calibration_full]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes, o_observed=0, alpha_selection=alpha_sel,
    )
    train_idx = list(train_idx)
    calib_idx = list(calib_idx)

    mu_baseline = _make_mu_baseline(config)
    mu_hcp = _make_mu_hcp(config)
    model_baseline = mu_baseline["fit_global"](
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
            mu_baseline["predict_global"](model_baseline, z["X"], Uj) for z in Zj
        ])
        scores_list.append(_baseline_group_scores(Zj, Uj, muj, score_aux))
    T_hcp = compute_hcp_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "hcp"),
    )

    Z_test_full = [
        {"X": group_data[test_group]["X"][i], "Y": group_data[test_group]["Y"][i]}
        for i in range(len(group_data[test_group]["Y"]))
    ]
    U_test = np.zeros((1, 1))
    target_index = TARGET_INDEX
    x_target = group_data[test_group]["X"][target_index]
    true_y = group_data[test_group]["Y"][target_index]
    mu_hat_hcp = mu_baseline["predict_global"](model_baseline, x_target, U_test[0])
    hcp_interval = _baseline_interval(mu_hat_hcp, T_hcp, x_target, U_test[0], score_aux)
    hcp_rec = _result_record(hcp_interval, true_y, outcome_scale="income")

    rows = []
    for o in o_values:
        dhcp_seed = seed + (replicate_idx + 1) * 1009 + (o + 1) * 131 + 17
        res = compute_donor_hcp_randomized_interval(
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
                quantile_base_seed, replicate_idx, target_index, o, "donor_hcp",
            ),
            return_intermediates=True,
        )
        rec = _result_record(res["interval"], true_y, outcome_scale="income")
        inter = res.get("intermediates") or {}
        meta = inter.get("scores_meta") or []
        finite = _finite_scores(meta)
        n_inf = sum(1 for m in meta if m.get("is_inf"))
        n_test_finite = sum(
            1 for m in meta
            if not m.get("is_inf") and m.get("y") is not None
            and inter.get("tau", 0) <= 0  # placeholder; count test tail below
        )
        # test scores = those after calib groups in meta with test PUMA characteristics
        # simpler: use tau and o
        tau = int(inter.get("tau", 0))
        n_test_finite = max(0, o - tau) if o > tau else 0
        rows.append({
            "permute": permute_rows,
            "replicate": replicate_idx,
            "o": o,
            "alpha": alpha,
            "test_puma": test_group,
            "n_test_puma": len(Z_test_full),
            "tau": tau,
            "q": float(inter.get("q", np.nan)),
            "n_scores_total": len(meta),
            "n_finite": int(np.isfinite(finite).sum()),
            "n_inf": n_inf,
            "n_test_finite": n_test_finite,
            "med_finite_score": float(np.median(finite)) if len(finite) else np.nan,
            "p90_finite_score": float(np.quantile(finite, 0.9)) if len(finite) else np.nan,
            "max_finite_score": float(np.max(finite)) if len(finite) else np.nan,
            "mu_center": float(inter.get("target", {}).get("mu", np.nan)),
            "mu_global_target": float(inter.get("target", {}).get("mu_g", np.nan)),
            "width": rec["width_income"],
            "half_width": rec["width_income"] / 2 if np.isfinite(rec["width_income"]) else np.nan,
            "coverage": rec["coverage"],
            "hcp_width_o0": hcp_rec["width_income"],
            "hcp_half_width": hcp_rec["width_income"] / 2,
            "donor": inter.get("donor"),
            "S_size": inter.get("S_size"),
            "N_donor": inter.get("N_donor"),
            "income_target": float(group_data[test_group]["income"][target_index]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--replicates", default="0,1,2,3,4")
    p.add_argument("--o_values", default="0,5,10,15,20")
    p.add_argument(
        "--cohort_csv",
        type=Path,
        default=REPO_ROOT / "plots_marginal/acs/yoep_fb_extended/data/acs_ca_yoep2000_fb_extended.csv",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df, X = _load_yoep_fb_cohort(args.cohort_csv)
    sizes = df.groupby("puma").size()
    eligible = sorted(int(g) for g in sizes.index if sizes[g] >= 21)
    o_values = [int(x) for x in args.o_values.split(",") if x.strip()]
    reps = [int(x) for x in args.replicates.split(",") if x.strip()]

    pieces = []
    for permute in (False, True):
        for rep in reps:
            pieces.append(_run_replicate(
                df=df, X=X, eligible_groups=eligible,
                replicate_idx=rep, o_values=o_values, alpha=args.alpha,
                permute_rows=permute,
            ))
    out = pd.concat(pieces, ignore_index=True)

    out_dir = args.out or (
        REPO_ROOT / "plots_marginal/acs/yoep_fb_extended/min21_permute/diagnostics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"alpha{int(round(args.alpha*100)):02d}"
    path = out_dir / f"dhcp_quantile_diagnostic_{tag}.csv"
    out.to_csv(path, index=False)

    print(f"Wrote {path}\n")
    for permute in (False, True):
        sub = out[out.permute == permute]
        print(f"=== permute={permute} alpha={args.alpha} (aggregate over reps {reps}) ===")
        g = sub.groupby("o").agg(
            q_med=("q", "median"),
            halfw_med=("half_width", "median"),
            width_med=("width", "median"),
            cov=("coverage", "mean"),
            n_finite=("n_finite", "median"),
            n_inf=("n_inf", "median"),
            n_test_finite=("n_test_finite", "median"),
            med_score=("med_finite_score", "median"),
            p90_score=("p90_finite_score", "median"),
        ).round(0)
        print(g.to_string())
        # o=0 GHCP vs HCP
        s0 = sub[sub.o == 0]
        print(f"  o=0 GHCP halfw med=${s0.half_width.median():,.0f}  HCP halfw med=${s0.hcp_half_width.median():,.0f}")
        print(f"  o=0 width match frac: {np.isclose(s0.half_width, s0.hcp_half_width, rtol=0.02).mean():.2f}")

    # permute vs no-permute same replicate correlation
    wide = out.pivot_table(
        index=["replicate", "o"], columns="permute", values="half_width",
    )
    if False in wide.columns and True in wide.columns:
        diff = (wide[True] - wide[False]).dropna()
        print(f"\npermute - no_permute half_width: med=${diff.median():,.0f}  "
              f"mean=${diff.mean():,.0f}  max|diff|=${diff.abs().max():,.0f}")
        corr = wide[False].corr(wide[True])
        print(f"corr(half_width permute, no_permute) across rep×o: {corr:.4f}")


if __name__ == "__main__":
    main()
