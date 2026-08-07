# ACS Stratified Sampling: Stratification Explanation

## Summary

The ACS data is stratified by **educational attainment level** across PUMAs to create 5 quantile-based strata. This document explains the exact process, variable definitions, and methodology.

---

## 1. Education Categories (Base ACS Data)

ACS SCHL (school attainment) codes are recoded into **4 education categories**:

| Category | ACS SCHL Code Range | Description |
|----------|-------------------|-------------|
| `<HS` | 1-15 | Less than high school |
| `HS` | 16-17 | High school graduate |
| `SomeCollege` | 18-20 | Some college or Associate's degree |
| `BAplus` | 21+ | Bachelor's degree or higher |

These 4 categories are the education **levels** in the individual-level ACS data.

---

## 2. Stratification Variable: BA+ Share (Proportion)

**The stratification variable is NOT an individual education category.**

Instead, the stratification variable is a **PUMA-level aggregate**: the **share (proportion) of individuals with BA+ (Bachelor's degree or higher)** within each PUMA.

$$\text{share\_baplus}_{j} = \frac{\#\{\text{individuals with SCHL} \geq 21 \text{ in PUMA } j\}}{\text{\# total individuals in PUMA } j}$$

This proportion ranges from ~0.04 to ~0.66 across PUMAs, representing the percentage of college-educated individuals in that geographic area.

---

## 3. Stratification Process (Step-by-Step)

### Step 1: Filter to Eligible PUMAs
- Load ACS PUMS data (California, foreign-born, age 25-54, 40+ work hours, income ≥ $10,000)
- Keep only PUMAs with ≥ 20 individuals
- **Result**: 45 eligible PUMAs (out of ~57 total California PUMAs)

### Step 2: Compute BA+ Share for Each PUMA
- For each of the 45 eligible PUMAs, compute `share_baplus`:
  - Count individuals with `educ_level == "BAplus"`
  - Divide by total individuals in that PUMA
- **Result**: Each PUMA has a share_baplus value (e.g., 0.045, 0.15, 0.48, 0.66)

### Step 3: Create Quantile-Based Strata
- Use `pandas.qcut(share_baplus, q=5, duplicates='drop')` to bin PUMAs into quantile-based strata
- This divides the range of share_baplus into 5 intervals such that:
  - Each interval has (approximately) equal number of PUMAs
  - Lower quantiles = lower BA+ share (less educated PUMAs)
  - Higher quantiles = higher BA+ share (more educated PUMAs)
- **Result**: 5 strata, each containing 9 PUMAs (45 / 5 = 9)

### Step 4: Strata Intervals
The 5 quantile bins typically look like:

| Stratum | Quantile | BA+ Share Interval | Interpretation |
|---------|----------|-------------------|-----------------|
| Q1 | 0-20% | (0.045, 0.496] | Lowest BA+ (least educated) |
| Q2 | 20-40% | (0.496, 0.580] | Low BA+ |
| Q3 | 40-60% | (0.580, 0.630] | Medium BA+ |
| Q4 | 60-80% | (0.630, 0.653] | High BA+ |
| Q5 | 80-100% | (0.653, 0.667] | Highest BA+ (most educated) |

(Exact intervals vary based on the data; consult `acs_stratified_data_overview.csv` for precise values.)

---

## 4. Why This Approach?

**Motivation**: We want to compare how HCP coverage/width varies across **geographic regions with different educational compositions**. By stratifying on BA+ share:

- **Q1 (low BA+ share)**: PUMAs where fewer individuals have college degrees (working-class, immigrant communities)
- **Q5 (high BA+ share)**: PUMAs where more individuals have college degrees (educated, professional communities)

This allows us to measure whether inference performance differs across these educational contexts.

---

## 5. Output: `acs_stratified_data_overview.csv`

The overview CSV contains one row per stratum with the following key columns:

| Column | Meaning |
|--------|---------|
| `STRATUM_RANK` | Quantile rank (1-5) |
| `STRATUM_INTERPRETATION` | Q1-Q5 with descriptor ("lowest/highest BA+ share") |
| `STRATUM_SHARE_BAPLUS_INTERVAL` | The interval of share_baplus values in this stratum |
| `STRATUM_SHARE_BAPLUS_MIN` | Minimum share_baplus in this stratum |
| `STRATUM_SHARE_BAPLUS_MAX` | Maximum share_baplus in this stratum |
| `STRATIFICATION_VARIABLE` | "share_baplus (computed as proportion of individuals with BA+ degree within each PUMA)" |
| `EDUCATION_CATEGORY_USED` | "BAplus (ACS educ code >= 21; Bachelor's degree or higher)" |
| `EDUCATION_CATEGORIES_AVAILABLE` | "<HS \| HS \| SomeCollege \| BAplus" |
| `STRATIFICATION_METHOD` | "Quantile binning via pandas pd.qcut(...)" |
| `n_pumas` | Number of PUMAs in this stratum |
| `median_income_individual_level` | Median individual income in this stratum |
| `BA_PLUS_SHARE_MEDIAN_IN_STRATUM` | Median BA+ share across PUMAs in this stratum |

---

## 6. Key Points to Remember

1. **BA+ is a proportion, not a category**: It's the share of individuals with a Bachelor's degree or higher within a PUMA (0-100%), not an individual-level category assignment.

2. **Stratification happens at the PUMA level, not individual level**: We divide PUMAs into 5 groups based on their BA+ share, then sample within those groups.

3. **Quantile binning is used**: This ensures equal (or nearly equal) representation: 45 PUMAs / 5 strata = 9 per stratum.

4. **Purpose**: To evaluate inference methods (HCP, D-HCP, etc.) across regions with different educational compositions.

---

## 7. References

- **Data source**: [generate_stratified_data_overview.py](generate_stratified_data_overview.py)
- **Experiments**: [repeated_experiments_stratified_acs.py](../repeated_experiments_stratified_acs.py)
- **Output plots**: [acs/NEW_PLOTS/stratified/](../acs/NEW_PLOTS/stratified/)
- **Output data**: [acs/results/stratified/](../acs/results/stratified/)
