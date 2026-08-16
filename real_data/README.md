# Real data

Paper experiments use the ACS path only:

| Path | Role |
|------|------|
| [`acs/`](acs/README.md) | Download, cleaning, design matrix, paper plotter, size-ignorability diagnostics |

Raw extract (gitignored): `acs/data/acs_data_all50states.csv`  
Download: `python real_data/acs/download_acs_ca_pums.py`

Blood-pressure experiments, bootstrap/stratified ACS runners, and early true-marginal scripts live under [`old/real_data/`](../old/real_data/).
