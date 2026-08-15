# ACS data and preprocessing

Entry point for the paper ACS suite: `code/marginal/run_acs_yoep_fb_min21_permute.py`  
Low-level runner: `code/marginal/run_acs_experiments.py`  
Filters / design matrix: `real_data/acs/data_processing.py`

---

## Setup (run from repo root)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

| Package | Role |
|---------|------|
| `numpy` | arrays / RNG |
| `pandas` | ACS CSV load, filters, recodes |
| `scikit-learn` | `RandomForestRegressor` (global + Std-CP local μ/σ) |
| `matplotlib` | paper figures |
| `folktables` | download 2018 ACS 1-year CA PUMS |
| `xgboost` | optional (non-paper Ding XGB suites only) |

Dev env used for the shipped suite: Python **3.13.1**, packages pinned in `requirements.txt`.

Then fetch the person extract (required; **not** committed to git):

```bash
.venv/bin/python real_data/acs/download_acs_ca_pums.py
```

---

## Where the data lives

| Artifact | Path | In git? |
|----------|------|---------|
| Raw CA PUMS extract (Census columns only) | `real_data/acs/data/acs_data_all50states.csv` | **No** (gitignored; ~14 MB / 378,817 rows) |
| Folktables download cache | `real_data/acs/data/folktables_cache/` | **No** |
| Filtered paper cohort (12,285 rows) | **Not a separate file** — built at runtime by `load_and_clean_acs_pums` | — |
| Experiment CSVs / figures | `paper-results/acs/` | figures/summaries/diagnostics yes; `*_detailed.csv` gitignored |

There is **no** shipped “pre-cleaned” cohort CSV. Reproducers download the raw extract once, then every runner applies the same filters in code.

---

## What we do / do not do to the data

**We do not** invent person rows, impute incomes, add survey weights, inflate with `ADJINC`, winsorize, or append synthetic PUMAs. The download script keeps only the Census columns listed below and writes them as released.

**We only** (in order, paper settings):

1. Rename PUMS codes → internal names (`AGEP`→`age`, …).
2. **Filter / drop** rows (CA, foreign-born, YOEP≥2000, age 25–54, hours≥40, positive income, complete cases).
3. **Recode** for the design matrix: education bins, `married`/`female` indicators, `age_sq`, optional `entry_recency` (excluded from **X** in the paper run).
4. Set outcome `y = PINCP` (raw dollars; no log).

Paper path uses `min_hours=40`, so the optional `hours` NA→0 fill (only when hours filter is disabled) **does not** apply.

---

## Data version (paper)

| Item | Value |
|------|--------|
| **ACS year** | **2018** (single calendar year) |
| **PUMS horizon** | **1-year** (`horizon="1-Year"`, not 5-year) |
| **Record type** | Person (`survey="person"`) |
| **Geography** | California only (`ST = 06`) |
| **PUMA vintage** | **2010 Census PUMA definitions** (5-digit `PUMA`; use with `ST` for unique areas) |
| **Download tool** | [`folktables`](https://github.com/zykls/folktables) `ACSDataSource` (`folktables==0.0.12` in dev env) |
| **Official source** | U.S. Census Bureau 2018 ACS 1-year PUMS ([FTP](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/), [data dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt)) |
| **Local CSV** | `real_data/acs/data/acs_data_all50states.csv` (378,817 CA person rows; filename is legacy) |
| **Download script** | `real_data/acs/download_acs_ca_pums.py` |

---

## Major items for the main paper (reproducibility checklist)

These are the **minimum** details reviewers expect in the real-data section. All values below match the shipped ACS suite (`paper-results/acs/seeds_manifest.json`).

### 1. Sample definition

- **Subpopulation:** California (`ST=06`), **foreign-born** (`NATIVITY=2`), **year of entry ≥ 2000** (`YOEP≥2000`), **ages 25–54** (`AGEP`), **usual hours worked ≥ 40** (`WKHP`).
- **Not an occupational “working class” filter:** we do **not** restrict by occupation or `COW` (class of worker). The labor-force filter is **full-time hours** (`WKHP≥40`). `COW` is **excluded from predictors** in paper runs.
- **Grouping:** PUMA (`PUMA` + `ST`); eligible PUMAs have **≥ 21** retained individuals (needed for target index 20).
- **Final extract:** **12,285** individuals in **265** CA PUMAs with any data; **212** PUMAs eligible (size ≥ 21).

### 2. PUMS variables and recodings

Raw columns retained in the CSV (see `download_acs_ca_pums.py`):

| PUMS name | Internal name | Use |
|-----------|---------------|-----|
| `ST` | `state_fips` → `state_abb` | CA filter |
| `PUMA` | `puma` | Group ID |
| `AGEP` | `age` | Filter + predictor |
| `NATIVITY` | `nativity` | Filter (`==2`) |
| `YOEP` | `yoep` | Filter (`≥2000`) |
| `WKHP` | `hours` | Filter (`≥40`) + predictor |
| `PINCP` | `income` | Outcome (raw dollars) |
| `SEX` | `sex` | `female = 1{SEX=2}` |
| `SCHL` | `educ` → `educ_level` | Predictor (binned, then dummies) |
| `MAR` | `marital` | `married = 1{MAR=1}` |
| `ENG` | `english` | Predictor (one-hot) |
| `COW` | `cow` | **Excluded** from paper predictors |

**Education bins** (`SCHL` → `educ_level`, `pd.cut`):

| `educ_level` | SCHL codes |
|--------------|------------|
| `<HS` | ≤ 15 |
| `HS` | 16–17 |
| `SomeCollege` | 18–20 |
| `BAplus` | ≥ 21 |

**English (`ENG`):** treated as **unordered categorical** (one-hot with drop-first), not as ordinal. Census codes: 1=Very well, 2=Well, 3=Not well, 4=Not at all.

**Derived:** `entry_recency = max(YOEP) − YOEP` (excluded from **X** on the paper YOEP-filtered run); `age_sq = age²` **is included** in **X**.

### 3. Missing values

Applied in order (`real_data/acs/data_processing.py`):

1. Drop rows with missing `state_abb` (from FIPS map).
2. After hours filter: drop rows with **missing `PINCP`**.
3. Drop **non-positive income** (`PINCP ≤ 0`).
4. Drop rows with missing values in **`y, age, age_sq, hours, entry_recency, educ_level, married, female, english, cow`** (list used for complete-case filter even when `entry_recency`/`cow` are excluded from **X**).

No imputation. Rows failing any step are removed.

### 4. Income handling

- **Outcome scale:** **raw dollars** (`--outcome_scale income`; `y = PINCP`, no log transform).
- **Top-coding:** Census **pre-top-codes** `PINCP` in PUMS (2018 range **$1–$4,209,995**; see PUMS data dictionary). We use values **as released**; no further top-code or winsorization in paper runs.
- **Inflation:** **`ADJINC` not used**; income is not adjusted to constant dollars.
- **Additional filters:** **no** minimum-income floor and **no** top-income trim in paper runs (`--min_income 0`, `--drop_top_income_pct 0`).

### 5. Survey weights

- **PUMS person weights (`PWGTP`) are not downloaded and not used.** All analyses are **unweighted**; each retained ACS person row counts equally.

### 6. Predictor matrix (`build_design_matrix_acs`)

Paper min21 settings (`--exclude_entry_recency`, `--exclude_cow` / `include_cow=False`).
Sec. 3.2 lists age, hours, marital status, sex, education, and English proficiency.
The implementation uses that list and also includes `age_sq=age²`:

- **Continuous:** `age`, `age_sq`, `hours`, `married`, `female`
- **Categorical (dummy-coded, drop-first):** `educ_level` (3 dummies), `english` (3 dummies)
- **Group covariate \(U\):** scalar **0** (no PUMA-level features)

### 7. Global μ-model (random forest)

| Setting | Value |
|---------|--------|
| Implementation | `sklearn.ensemble.RandomForestRegressor` |
| Trees | `n_estimators = 50` |
| Min leaf size | `min_samples_leaf = 5` |
| `max_features` | \(\lfloor\sqrt{p_X+d_U}\rfloor\) |
| `random_state` | **123** |
| `n_jobs` | 1 |
| Merger | paper **(3)**: \(\widetilde\mu=(1-\lambda_{\mathrm{local}})\widehat\mu^{\mathrm{global}}+\lambda_{\mathrm{local}}\overline Y\), \(\tau=\lfloor o/2\rfloor\), \(\lambda_{\mathrm{local}}=\tau/(|S_{\mathrm{train}}|+\tau)\) with \(c=1\) (`--within_group_mode mean`) |
| Baseline μ (HCP / pooling / …) | Same RF, **pure global** (\(\tau=0\)) |

GHCP fits the global RF on groups in \(S_{\mathrm{train}}\) for the replicate; \(\overline Y\) uses indices \(0,\ldots,\tau-1\) in the (permuted) stream. History size \(o\) is the number of initially observed test-group records; it is **not** equal to \(\tau\).

### 8. Conformal protocol (per replicate)

Design: **`uniform_one_target`** (`paper-results/acs/seeds_manifest.json`), matching Sec. 3.2.

1. **PUMA selection seed:** `456 + replicate_idx × 1009`
   - Draw **20 calibration PUMAs** uniformly without replacement from eligible PUMAs (size \(\ge 21\)).
   - Draw **1 target PUMA** uniformly from the remainder.
2. **Row permutation seed:** `456 + replicate_idx × 1009 + 811`
   - Permute row order within each selected PUMA.
3. **Target individual:** **index 20** (paper: individual at **position 21**); history of size \(o\) uses indices \(0,\ldots,o-1\).
4. **\(o\in\{0,5,10,15,20\}\)**. Restricted GHCP with **\(\eta=0.5\)**.
5. No row bootstrap; PUMA sizes equal observed ACS counts.
6. **\(B=1000\)**; **\(\alpha\in\{0.05,0.10,0.15,0.20\}\)** (main text Table 5 / Fig. 5 use \(\alpha=0.1\)).

**Scores:** GHCP / S-HCP / baselines use **absolute** residuals. Std-CP: **studentized** local RF (min leaf 5) with **randomized** quantiles; GHCP quantiles **deterministic** (`quantile_base_seed=456`).

**HCP split:** among the 20 calibration PUMAs, `alpha_selection = 0.5` (equal training/calibration split in HCP).

### 9. Software and seeds (summary table for paper)

| Component | Value |
|-----------|--------|
| Python | 3.13 (dev); any recent 3.x with the packages below |
| Install | `pip install -r requirements.txt` (pinned: numpy, pandas, scikit-learn, matplotlib, folktables, xgboost) |
| Suite seed | **456** |
| Quantile base seed | **456** (deterministic GHCP/HCP quantiles) |
| RF `random_state` | **123** |
| GHCP interval seed | `456 + (replicate_idx+1)×1009 + (o+1)×131 + 17` |
| Manifest | `paper-results/acs/seeds_manifest.json` |

### 10. Interpretation (size ignorability)

The ACS study is an **empirical illustration under an approximate working model**, not validation of **N_j ⟂ (U_j, μ_j)**. See `run_size_ignorability_diagnostics.py` and `paper-results/acs/diagnostics/`.

---

## Quick commands

PUMA-level summary:

```bash
python real_data/acs/export_puma_summary.py
```

Reproduce experiments (production):

```bash
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py \
  --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
```

Plot from saved results:

```bash
python real_data/acs/plot_dgp_style_paper_plots.py --suite rf_yoep_fb_min21
```

Size-ignorability diagnostics:

```bash
python real_data/acs/run_size_ignorability_diagnostics.py
```

---

## Cohort filters (code reference)

Applied in order in `data_processing.py` (`load_and_clean_acs_pums`):

| Step | Paper filter |
|------|----------------|
| State | California |
| Nativity | Foreign-born (`NATIVITY == 2`) |
| Age | 25–54 |
| Year of entry | YOEP ≥ 2000 |
| Labor force | Usual hours ≥ 40 |
| Income | Drop missing; drop non-positive |
| Covariates | Complete-case on model columns |
| Income floor / trim | None |

## Archive

Superseded ACS runners, bootstrap/stratified experiments, blood-pressure study, and old local
plots/results: [`old/real_data/`](../../old/real_data/). See [`old/README.md`](../../old/README.md).
