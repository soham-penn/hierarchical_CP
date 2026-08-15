#!/usr/bin/env python3
"""ACS Std-CP abs vs studentized with nodesize=1 (same seeds as paper)."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "real_data")]

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums
from code.marginal import run_acs_experiments as acs
from scores import make_quantile_seed

SEED = 456
QSEED = 456
TARGET = 20
O_VALUES = [0, 5, 10, 15, 20]
OUT = Path("paper-results/acs/stdcp_nodesize1")
NODE = 1

_DF = _X = _ELIG = _CNT = None


def init(df, X, el, cnt):
    global _DF, _X, _ELIG, _CNT
    _DF, _X, _ELIG, _CNT = df, X, el, cnt
    acs._set_target_index(TARGET)


def one(args):
    r, alpha, score = args
    df, X, el, cnt = _DF, _X, _ELIG, _CNT
    calib, test = acs.sample_calibration_and_target_uniform(
        eligible_groups=el,
        group_counts=cnt,
        selection_seed=SEED + r * 1009,
        n_calib_groups=20,
        min_target_size=acs.MIN_TARGET_PUMA_SIZE,
    )
    if test is None:
        return None
    # Match paper: permute calib groups then target (same RNG stream).
    rng = np.random.default_rng(SEED + r * 1009 + 811)
    group_data = {}
    for grp in list(calib) + [test]:
        gidx = rng.permutation(np.where((df.puma == grp).values)[0])
        group_data[grp] = {
            "X": X[gidx],
            "Y": df.iloc[gidx]["y"].to_numpy(float),
            "income": (
                df.iloc[gidx]["income"].to_numpy(float)
                if "income" in df.columns
                else df.iloc[gidx]["y"].to_numpy(float)
            ),
        }
    Xg = group_data[test]["X"]
    Yg = group_data[test]["Y"]
    income = group_data[test]["income"]
    xt, yt, inc = Xg[TARGET], float(Yg[TARGET]), float(income[TARGET])
    rows = []
    for o in O_VALUES:
        if len(Yg) <= max(o, TARGET) or o <= 0:
            rec = acs._result_record((-np.inf, np.inf), yt, outcome_scale="income")
        else:
            s_rng = np.random.default_rng(
                make_quantile_seed(QSEED, r, TARGET, o, "stdcp_split")
            )
            interval = acs._compute_std_cp_interval(
                x_hist=Xg[:o],
                y_hist=Yg[:o],
                x_target=xt,
                alpha=alpha,
                rng=s_rng,
                quantile_mode="randomized",
                quantile_random_seed=make_quantile_seed(QSEED, r, TARGET, o, "stdcp"),
                nodesize=NODE,
                score_type=score,
            )
            rec = acs._result_record(interval, yt, outcome_scale="income")
        rows.append(
            dict(
                replicate=r,
                method="Std-CP",
                o=o,
                score_type=score,
                nodesize=NODE,
                income_target=inc,
                **rec,
            )
        )
    return rows


def run(alpha, score, B=1000, workers=6):
    tag = f"alpha{int(round(alpha * 100)):02d}"
    print(f"=== nodesize={NODE} {score} {tag} ===", flush=True)
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init, initargs=(_DF, _X, _ELIG, _CNT)
    ) as ex:
        futs = [ex.submit(one, (r, alpha, score)) for r in range(B)]
        done = 0
        for fut in as_completed(futs):
            part = fut.result()
            if part is None:
                raise RuntimeError("replicate failed")
            rows.extend(part)
            done += 1
            if done % 200 == 0 or done == B:
                print(f"  {done}/{B}", flush=True)
    df = pd.DataFrame(rows).sort_values(["replicate", "o"]).reset_index(drop=True)
    out = OUT / tag / f"stdcp_ns1_{score}_{tag}_detailed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote", out, flush=True)
    return df


def main():
    global _DF, _X, _ELIG, _CNT
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading cohort...", flush=True)
    acs._set_target_index(TARGET)
    df = load_and_clean_acs_pums(
        str(REPO / "real_data/acs/data/acs_data_all50states.csv"),
        states_keep=["CA"],
        age_min=25,
        age_max=54,
        yoep_min_year=2000,
        min_hours=40,
        min_income=None,
        drop_top_income_fraction=None,
        y_transform=acs._outcome_transform("income"),
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)
    X = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=True)
    cnt = df.groupby("puma").size()
    el = cnt[cnt >= 21].index.to_numpy().tolist()
    _DF, _X, _ELIG, _CNT = df, X, el, cnt
    print(f"n={len(df)} eligible={len(el)}", flush=True)
    (OUT / "seeds_manifest.json").write_text(
        json.dumps(
            dict(
                nodesize=NODE,
                seed=SEED,
                note="same seeds as paper; leaf=1 so RF-σ splits",
            ),
            indent=2,
        )
    )

    for a in (0.1, 0.2):
        for score in ("absolute", "studentized"):
            run(a, score)

    rows = []
    for a in (0.1, 0.2):
        tag = f"alpha{int(round(a * 100)):02d}"
        ab = pd.read_csv(OUT / tag / f"stdcp_ns1_absolute_{tag}_detailed.csv")
        st = pd.read_csv(OUT / tag / f"stdcp_ns1_studentized_{tag}_detailed.csv")
        paper = pd.read_csv(
            f"paper-results/acs/results/"
            f"true_marginal_permuted_rf_income_notrim_yoep2000_{tag}/"
            f"acs_true_marg_{tag}_detailed.csv"
        )
        paper = paper[paper["method"].astype(str) == "Std-CP"]
        for o in O_VALUES:
            for name, g in (
                ("abs_ns1", ab[ab.o == o]),
                ("stud_ns1", st[st.o == o]),
                ("paper_ns5", paper[paper.o == o]),
            ):
                w = g.width.astype(float)
                fin = np.isfinite(w)
                rows.append(
                    dict(
                        alpha=a,
                        o=o,
                        method=name,
                        n_fin=int(fin.sum()),
                        cov=float(g.coverage.mean()),
                        w_mean=float(w[fin].mean()) if fin.any() else np.nan,
                        w_med=float(np.median(w[fin])) if fin.any() else np.nan,
                    )
                )
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary_abs_stud_ns1_vs_ns5.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for a in (0.1, 0.2):
        tag = f"alpha{int(round(a * 100)):02d}"
        ab = pd.read_csv(OUT / tag / f"stdcp_ns1_absolute_{tag}_detailed.csv")
        st = pd.read_csv(OUT / tag / f"stdcp_ns1_studentized_{tag}_detailed.csv")
        m = ab.merge(st, on=["replicate", "o"], suffixes=("_a", "_s"))
        for o in [10, 15, 20]:
            sub = m[m.o == o]
            wa, ws = sub.width_a.astype(float), sub.width_s.astype(float)
            ok = np.isfinite(wa) & np.isfinite(ws)
            if not ok.any():
                continue
            r = wa[ok] / ws[ok]
            print(
                f"α={a} o={o}: n={ok.sum()} mean abs/stud={r.mean():.3f} "
                f"med={np.median(r):.3f} cov_abs={sub.coverage_a.mean():.3f} "
                f"cov_stud={sub.coverage_s.mean():.3f}"
            )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
