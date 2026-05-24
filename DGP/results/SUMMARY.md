# DGP Results Overview (Maintained Experiments)

This repository now maintains two joint-XY marginal DGP experiment variants only:

1. Fixed group size: `N_i = 51`
2. Poisson group size: `N_i = 1 + Poisson(20)`

Both variants share the same model and hyperparameters; only the group-size mechanism differs.

## Shared DGP model

- Dimension: `d = 5`
- Calibration groups: `K = 20`
- Group covariates: `U_i ~ Unif([0,1]^5)`
- Joint draw per observation:
	- `Z = (X, Y) ~ N(mu(U), Sigma(U))`
	- `mu(U) = (U_1^2, ..., U_5^2)`
	- `Sigma(U) = (1-rho) diag(U) + rho 11^T`, with `rho = 0.5`

## Experimental hyperparameters

- `alpha = 0.2`
- `alpha_selection = 0.5`
- `o_values = {0, 5, 10, 15, 20, 25, 30, 35, 40}`
- `target_index = 40` (0-based)
- `number_experiments = 50`
- `number_test_groups = 100`
- `number_subsampling_repetitions = 50`
- Baseline mu model: `create_mu_method_ols_global_only()`
- HCP mu model: `create_mu_method_ols_offset()`
- Includes o-dependent Std-CP baseline columns:
	- `coverage_stdcp`, `width_stdcp`, `infinite_stdcp`

## Retained output tags

- `fixedN51_K20_d5_u01_rho05_joint_xy_marginal`
- `poissonNmean21_K20_d5_u01_rho05_joint_xy_marginal`

## File layout

For each tag above:

- Raw CSV:
	- `DGP/results/files/<tag>/results_effect_of_o_joint_xy_marginal.csv`
- Summary CSV:
	- `DGP/results/<tag>/summary_effect_of_o_joint_xy_marginal.csv`
- Plots:
	- `DGP/plots/<tag>/set1_side_by_side_dhcp_hcp_o_upto20.pdf`
	- `DGP/plots/<tag>/set2_coverage_dhcp_stdcp_hcp_by_o_upto40.pdf`
	- `DGP/plots/<tag>/set2_width_dhcp_stdcp_hcp_by_o_upto40.pdf`
	- `DGP/plots/<tag>/set3_coverage_dhcp_with_baselines_no_stdcp_o_upto40.pdf`
	- `DGP/plots/<tag>/set3_width_dhcp_with_baselines_no_stdcp_o_upto40.pdf`
	- `DGP/plots/<tag>/effect_of_o_coverage_donor_focus_lambda*.pdf`
	- `DGP/plots/<tag>/effect_of_o_width_donor_focus_lambda*.pdf`
	- `DGP/plots/<tag>/effect_of_o_coverage_sample_focus_lambda*.pdf`
	- `DGP/plots/<tag>/effect_of_o_width_sample_focus_lambda*.pdf`

## Aggregated overview

- `DGP/results/dgp_overview_summary.csv`

This file consolidates method-level mean/std summaries (coverage, width, infinite-rate) across retained DGP result CSV files.
