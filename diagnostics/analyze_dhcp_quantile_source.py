#!/usr/bin/env python3
"""
Diagnose whether D-HCP conformal threshold q(o) is driven by target-group residuals.

For each replicate and o, rebuilds the same weighted score pool as donor_hcp and records:
  - whether q_upper / q matches a target-group finite score
  - weight mass from target vs calibration at the beta knot
  - cal-only vs full-mixture quantiles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DGP import create_dgp_specification_default
from code.shared.dgp.joint_xy_marginal_common import draw_group_joint_xy
from methods.donor_hcp import (
    _select_s_tilde_with_tie_randomization,
    compute_donor_hcp_randomized_interval,
)
from methods.mu_methods import create_mu_method_ols_offset
from scores import _aggregate_weighted_support, conformal_threshold


def build_dhcp_score_pool(
    U_calibration,
    Z_calibration,
    U_test,
    Z_test,
    o_observed,
    alpha_selection,
    mu_method,
    random_seed,
    tau_override=None,
):
    """Mirror donor_hcp score construction with per-score source tags."""
    K = len(Z_calibration)
    N = np.array([len(z) for z in Z_calibration])
    test_idx = K

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if tau_override is None:
        tau = int(np.floor(o_observed / 2))
        if o_observed > 0 and tau >= o_observed:
            tau = o_observed - 1
    else:
        tau = max(0, min(int(tau_override), max(0, o_observed - 1)))

    # No cal groups large enough for this o (e.g. o > fixed_n): test-only pool.
    if len(S_tilde) == 0:
        U_all = U_test
        Z_all = [Z_test]
        global_model = mu_method["fit_global"](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K)),
        )
        if tau > 0:
            offset_test = mu_method["fit_group_adjustment"](
                model_global=global_model,
                u_group_vector=U_test[0, :],
                Z_group_list=Z_test,
                training_index_vector=list(range(tau)),
            )
        else:
            offset_test = 0.0
        scores, weights, sources = [], [], []
        if o_observed >= tau + 1:
            cal_idx = list(range(tau, o_observed))
            for i in cal_idx:
                z = Z_test[i]
                mu = mu_method["predict_group_mu"](
                    model_global=global_model,
                    group_adjustment=offset_test,
                    x_vector=z["X"],
                    u_group_vector=U_test[0, :],
                )
                scores.append(float(np.abs(z["Y"] - mu)))
                weights.append(1.0)
                sources.append("test_finite")
            scores.append(np.inf)
            weights.append(1.0)
            sources.append("test_inf")
        w = np.asarray(weights, dtype=float)
        if w.size:
            w = w / w.sum()
        return {
            "scores": np.asarray(scores, dtype=float),
            "weights": w,
            "sources": np.asarray(sources),
            "tau": tau,
            "S_size": 1,
            "donor": None,
            "test_finite": np.asarray(
                [s for s, src in zip(scores, sources) if src == "test_finite"], dtype=float
            ),
            "test_only_path": True,
        }

    donor_rng = np.random.default_rng(random_seed)
    donor = int(donor_rng.choice(S_tilde))
    N_donor = N[donor]
    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S_size = len(S_cal) + 1

    U_all = np.vstack([U_calibration, U_test])
    Z_all = list(Z_calibration) + [Z_test]
    S_comp = np.setdiff1d(np.arange(K + 1), np.sort(np.concatenate([S_cal, [test_idx]])))

    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method["fit_global"](
            U_matrix=U_all,
            Z_list=Z_all,
            group_index_vector=list(S_comp),
        )

    scores = []
    weights = []
    sources = []  # 'cal', 'test_finite', 'test_inf'
    cal_group_ids = []

    for j in S_cal:
        if N[j] <= tau:
            continue
        if tau > 0:
            offset_j = mu_method["fit_group_adjustment"](
                model_global=global_model,
                u_group_vector=U_calibration[j, :],
                Z_group_list=Z_calibration[j],
                training_index_vector=list(range(tau)),
            )
        else:
            offset_j = 0.0
        idx_tail = list(range(tau, N[j]))
        w_j = 1.0 / (S_size * len(idx_tail))
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu = mu_method["predict_group_mu"](
                model_global=global_model,
                group_adjustment=offset_j,
                x_vector=z["X"],
                u_group_vector=U_calibration[j, :],
            )
            scores.append(float(np.abs(z["Y"] - mu)))
            weights.append(w_j)
            sources.append("cal")
            cal_group_ids.append(int(j))

    if tau > 0:
        offset_test = mu_method["fit_group_adjustment"](
            model_global=global_model,
            u_group_vector=U_test[0, :],
            Z_group_list=Z_test,
            training_index_vector=list(range(tau)),
        )
    else:
        offset_test = 0.0

    idx_tail_test = list(range(tau, o_observed)) if o_observed > tau else []
    test_finite = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu = mu_method["predict_group_mu"](
            model_global=global_model,
            group_adjustment=offset_test,
            x_vector=z["X"],
            u_group_vector=U_test[0, :],
        )
        test_finite.append(float(np.abs(z["Y"] - mu)))

    n_inf = max(0, N_donor - o_observed)
    n_total_test = len(test_finite) + n_inf
    if n_total_test > 0:
        w_test = 1.0 / (S_size * n_total_test)
        for s in test_finite:
            scores.append(s)
            weights.append(w_test)
            sources.append("test_finite")
            cal_group_ids.append(-1)
        for _ in range(n_inf):
            scores.append(np.inf)
            weights.append(w_test)
            sources.append("test_inf")
            cal_group_ids.append(-1)

    return {
        "scores": np.asarray(scores, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "sources": np.asarray(sources),
        "tau": tau,
        "S_size": S_size,
        "donor": donor,
        "offset_test": offset_test,
        "global_model": global_model,
        "test_finite": np.asarray(test_finite, dtype=float),
        "test_only_path": False,
    }


def analyze_quantile_source(pool, alpha, quantile_random_seed=0):
    """Attribute beta-quantile knot to calibration vs target group."""
    scores = pool["scores"]
    weights = pool["weights"]
    sources = pool["sources"]
    beta = 1.0 - alpha

    q_info = conformal_threshold(
        scores,
        weights,
        alpha=alpha,
        quantile_mode="deterministic",
        return_info=True,
    )
    q = float(q_info["q_randomized"])
    q_lo = float(q_info["q_lower"]) if np.isfinite(q_info["q_lower"]) else np.nan
    q_hi = float(q_info["q_upper"])

    support, masses, cdf = _aggregate_weighted_support(scores, weights)
    j = int(np.searchsorted(cdf, beta, side="left"))
    if j >= support.size:
        j = support.size - 1
    knot_val = float(support[j])

    atol = 1e-9
    mask_knot = np.abs(scores - knot_val) < atol if np.isfinite(knot_val) else np.isinf(scores)
    w_knot = weights[mask_knot]
    src_knot = sources[mask_knot]
    mass_knot = float(np.sum(w_knot))
    mass_knot_test = float(np.sum(w_knot[src_knot == "test_finite"]))
    mass_knot_cal = float(np.sum(w_knot[src_knot == "cal"]))

    finite = np.isfinite(scores)
    w = weights / np.sum(weights)
    mass_test_total = float(np.sum(w[sources == "test_finite"]))
    mass_cal_total = float(np.sum(w[sources == "cal"]))
    mass_inf_total = float(np.sum(w[sources == "test_inf"]))

    if np.isfinite(q):
        mass_below_q_test = float(np.sum(w[(sources == "test_finite") & (scores <= q + atol)]))
        mass_below_q_cal = float(np.sum(w[(sources == "cal") & (scores <= q + atol)]))
    else:
        mass_below_q_test = mass_below_q_cal = np.nan

    cal_mask = sources == "cal"
    test_fin_mask = sources == "test_finite"
    w_cal = weights[cal_mask]
    s_cal = scores[cal_mask]
    q_cal_only = np.inf
    if w_cal.size and np.sum(w_cal) > 0:
        q_cal_only = float(
            conformal_threshold(s_cal, w_cal, alpha=alpha, quantile_mode="deterministic")
        )

    # Threshold if test finite scores removed (keep cal + inf padding only).
    keep = (sources != "test_finite")
    q_no_test_finite = np.inf
    if np.any(keep) and np.sum(weights[keep]) > 0:
        q_no_test_finite = float(
            conformal_threshold(
                scores[keep], weights[keep], alpha=alpha, quantile_mode="deterministic"
            )
        )
    q_changes_without_test = bool(
        np.isfinite(q) and np.isfinite(q_no_test_finite) and abs(q - q_no_test_finite) > atol
    )

    test_fin = pool["test_finite"]
    max_test = float(np.max(test_fin)) if test_fin.size else np.nan
    med_cal = float(np.median(s_cal)) if s_cal.size else np.nan
    p90_cal = float(np.quantile(s_cal, 0.9, method="linear")) if s_cal.size else np.nan

    # q_hi equals knot used for deterministic when gamma=1; check membership
    test_vals = scores[test_fin_mask & finite]
    q_is_test_value = bool(
        test_vals.size and np.any(np.abs(test_vals - q_hi) < atol)
    )
    q_hi_is_test_value = q_is_test_value
    q_hi_is_cal_only = bool(
        np.isfinite(q_hi)
        and np.any(np.abs(s_cal - q_hi) < atol)
        and not q_hi_is_test_value
    )
    # shared knot: same value in cal and test
    q_hi_shared = bool(
        np.isfinite(q_hi)
        and np.any(np.abs(s_cal - q_hi) < atol)
        and np.any(np.abs(test_vals - q_hi) < atol)
    ) if test_vals.size and s_cal.size else False

    return {
        "q": q,
        "q_lower": q_lo,
        "q_upper": q_hi,
        "knot_value": knot_val,
        "mass_knot_test_frac": mass_knot_test / mass_knot if mass_knot > 0 else np.nan,
        "mass_knot_cal_frac": mass_knot_cal / mass_knot if mass_knot > 0 else np.nan,
        "q_upper_is_test_value": q_hi_is_test_value,
        "q_upper_is_cal_only": q_hi_is_cal_only,
        "q_upper_shared_cal_test": q_hi_shared,
        "q_equals_max_test_finite": bool(
            test_fin.size and np.isfinite(q) and np.abs(q - max_test) < atol
        ),
        "q_cal_only": q_cal_only,
        "q_over_q_cal_only": q / q_cal_only if np.isfinite(q) and np.isfinite(q_cal_only) and q_cal_only > 0 else np.nan,
        "q_no_test_finite": q_no_test_finite,
        "q_changes_without_test_finite": q_changes_without_test,
        "max_test_finite": max_test,
        "p90_cal_scores": p90_cal,
        "median_cal_scores": med_cal,
        "mass_weight_test": mass_test_total,
        "mass_weight_cal": mass_cal_total,
        "mass_weight_inf": mass_inf_total,
        "mass_below_q_test": mass_below_q_test,
        "mass_below_q_cal": mass_below_q_cal,
        "n_cal_scores": int(np.sum(cal_mask)),
        "n_test_finite": int(np.sum(test_fin_mask)),
        "n_test_inf": int(np.sum(sources == "test_inf")),
    }


def run_diagnostic(
    n_replicates=500,
    alpha=0.1,
    o_values=None,
    seed=0,
    output_csv=None,
):
    if o_values is None:
        o_values = [0, 5, 10, 15, 20, 25, 30, 35]

    K, fixed_n, target_index = 20, 21, 35
    target_n = target_index + 1
    mu = create_mu_method_ols_offset()
    rows = []

    for b in range(n_replicates):
        np.random.seed(seed + b)
        u_cal = np.random.uniform(1.0, 5.0, size=(K, 5))
        z_cal = [draw_group_joint_xy(u_cal[j], fixed_n, rho=0.5) for j in range(K)]
        u_test = np.random.uniform(1.0, 5.0, size=(1, 5))
        z_test = draw_group_joint_xy(u_test[0], target_n, rho=0.5)
        Z_test = [{"X": z["X"], "Y": z["Y"]} for z in z_test]

        for o in o_values:
            pool = build_dhcp_score_pool(
                u_cal, z_cal, u_test, Z_test,
                o_observed=o,
                alpha_selection=0.5,
                mu_method=mu,
                random_seed=seed + b,
            )
            stats = analyze_quantile_source(pool, alpha=alpha)
            ref = compute_donor_hcp_randomized_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=0.5,
                mu_method=mu,
                test_index_target=target_index,
                random_seed=seed + b,
                quantile_mode="deterministic",
                return_quantile_info=True,
            )
            lo, hi = ref["interval"]
            width = hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan
            rows.append({
                "replicate": b,
                "o": o,
                "tau": pool["tau"],
                "alpha": alpha,
                "width": width,
                "q_method": ref.get("q_randomized"),
                **stats,
            })

    df = pd.DataFrame(rows)
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")

    summary = (
        df.groupby("o")
        .agg(
            width_med=("width", "median"),
            q_med=("q", "median"),
            q_cal_only_med=("q_cal_only", "median"),
            frac_q_upper_test=("q_upper_is_test_value", "mean"),
            frac_q_upper_cal_only=("q_upper_is_cal_only", "mean"),
            frac_q_upper_shared=("q_upper_shared_cal_test", "mean"),
            frac_q_equals_max_test=("q_equals_max_test_finite", "mean"),
            frac_q_changes_no_test=("q_changes_without_test_finite", "mean"),
            mass_knot_test_mean=("mass_knot_test_frac", "mean"),
            mass_weight_test_mean=("mass_weight_test", "mean"),
            q_ratio_med=("q_over_q_cal_only", "median"),
            max_test_med=("max_test_finite", "median"),
            p90_cal_med=("p90_cal_scores", "median"),
        )
        .reset_index()
    )
    return df, summary


def dhcp_quantile_rank_table_tau0(
    o_values=None,
    alpha=0.1,
    n_cal_groups=10,
    n_per_group=21,
    tau=0,
):
    """
    Closed-form D-HCP weighted-quantile rank with fixed tau.

    When tau=0 and o <= n_per_group, every score has weight 1/(S_size * n_per_group)
    and M = S_size * n_per_group is constant. Then k = ceil((1-alpha)*M) is constant
    and only F(o) = n_cal_groups*n_per_group + o grows with o.
    """
    if o_values is None:
        o_values = [0, 5, 10, 15, 20, 21]

    beta = 1.0 - alpha
    s_size = n_cal_groups + 1

    rows = []
    for o in o_values:
        n_inf = max(0, n_per_group - o)
        n_test_fin = max(0, o - tau)
        n_cal = n_cal_groups * (n_per_group - tau)
        n_test_block = n_test_fin + n_inf
        n_finite = n_cal + n_test_fin

        equal_weights = n_test_block == (n_per_group - tau)
        if equal_weights:
            m = n_per_group - tau
            m_total = s_size * m
            w = 1.0 / m_total
            k = int(np.ceil(beta * m_total))
        else:
            w_cal = 1.0 / (s_size * (n_per_group - tau))
            w_test = 1.0 / (s_size * n_test_block) if n_test_block > 0 else np.nan
            w = np.nan
            m_total = n_finite + n_inf
            masses = []
            if n_cal > 0:
                masses.append((n_cal, w_cal))
            if n_test_fin > 0:
                masses.append((n_test_fin, w_test))
            if n_inf > 0:
                masses.append((n_inf, w_test))
            cum, k = 0.0, 0
            for cnt, w_block in masses:
                if cum + cnt * w_block >= beta - 1e-15:
                    need = beta - cum
                    k += int(np.ceil(need / w_block - 1e-15))
                    break
                cum += cnt * w_block
                k += cnt

        k_finite = min(k, n_finite)
        rows.append({
            "o": o,
            "tau": tau,
            "equal_weights": equal_weights,
            "M_atoms": m_total,
            "w": w,
            "k": k,
            "F_finite": n_finite,
            "n_cal_finite": n_cal,
            "n_test_finite": n_test_fin,
            "n_inf": n_inf,
            "k_on_finite": k_finite,
            "pct_of_finite": k_finite / n_finite if n_finite else np.nan,
            "mass_cal": n_cal / m_total if equal_weights else n_cal * w_cal,
            "mass_test_finite": n_test_fin / m_total if equal_weights else n_test_fin * (w_test if n_test_block else 0),
            "mass_inf": n_inf / m_total if equal_weights else n_inf * (w_test if n_inf else 0),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--o-values",
        type=str,
        default="0,5,10,15,20,25,30,35",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "diagnostics" / "dhcp_quantile_source_by_o.csv",
    )
    parser.add_argument(
        "--rank-table-tau0",
        action="store_true",
        help="Print closed-form quantile-rank table with tau=0 only.",
    )
    args = parser.parse_args()
    o_values = [int(x) for x in args.o_values.split(",")]

    if args.rank_table_tau0:
        tbl = dhcp_quantile_rank_table_tau0(o_values=o_values, alpha=args.alpha)
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(f"\n=== D-HCP rank table (tau=0, alpha={args.alpha}, |S_cal|=10, N=21) ===\n")
        print("Formulas for o<=21: M=231, k=ceil(0.9*M)=208, F(o)=210+o, pct_finite=208/F(o)\n")
        print(tbl.to_string(index=False))
        return

    df, summary = run_diagnostic(
        n_replicates=args.n_replicates,
        alpha=args.alpha,
        o_values=o_values,
        seed=args.seed,
        output_csv=args.output_csv,
    )

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== D-HCP quantile source diagnostic (DGP fixedN21, alpha={}) ===\n".format(args.alpha))
    print(summary.to_string(index=False))
    print("\nInterpretation:")
    print("  frac_q_upper_test: q_upper equals some target-group finite residual in the pool")
    print("  frac_q_upper_cal_only: q_upper only appears among cal scores (not target)")
    print("  q_cal_only_med: (1-alpha) quantile using only cal scores with original weights")
    print("  q_ratio_med: q / q_cal_only — how much test+tau changes threshold vs cal-only")


if __name__ == "__main__":
    main()
