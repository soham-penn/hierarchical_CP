# Archived material

Nothing under `old/` is needed to reproduce the paper figures. Live paths:

- [`paper-results/`](../paper-results/README.md)
- [`real_data/acs/`](../real_data/acs/README.md)
- `code/marginal/` (Section 3.1–3.2 and Appendix D launchers)

## Not used in the paper

These exist in `methods/` or under `old/` but were **not** used for any paper
table or figure:

- merger exponent $c\neq 1$
- empirical-Bayes / random-effects weight
  $w_g=(\hat\sigma^2/\tau)/(\hat\tau_B^2+\hat\sigma^2/\tau)$ (depends on other
  groups’ residuals; not covered by Appendix B.1). Gated on
  `mu_method["merger"] in {"bayes_re","bayes","re"}`; Section 3.1/3.2 runners
  never set that flag.
- ridge residual correction with clip $0.5$
- local random-forest offset in place of the group mean
- blood-pressure clinic data, 50-state ACS, and per-state top-25% income
  filters
- early name “HCP++” (the paper method is GHCP)
- sorting ACS records by income or score percentiles (the paper permutes
  within PUMA)

Exploratory empirical-Bayes plots also sit under
`paper-results/dgp/figures/merger_compare/`, `re_vs_std/`, and
`oracleB_vs_re/`.

## Layout

```
old/
├── code/            # superseded runners (capture, conditional, alt ACS)
├── real_data/
│   ├── acs/         # old ACS plots / results
│   ├── blood_pressure/
│   └── scripts/     # bootstrap / stratified / early true-marginal runners
├── dgp/             # archived simulation plot/result trees
├── conditional/
└── diagnostics/
```

## What moved here from live trees

| Former path | Now |
|-------------|-----|
| `real_data/blood_pressure/` | `old/real_data/blood_pressure/` |
| `real_data/repeated_experiments*.py`, `acs_true_marginal_experiments.py` | `old/real_data/scripts/` |
| `real_data/acs/results/`, `acs/plots/` | `old/real_data/acs/legacy_local/` |
| `real_data/acs/data/acs_top25_filtered.csv` | `old/real_data/acs/data/` |
