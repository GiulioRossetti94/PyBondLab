# PyBondLab Examples

Self-contained scripts demonstrating the full PyBondLab workflow for empirical asset pricing research in corporate bond markets. Each script showcases distinct package features and can be run independently.

---

## Setup

All scripts share `_config.py`, which loads a monthly bond panel in the standard PyBondLab format (`date`, `ID`, `ret`, `VW`, `RATING_NUM`). Place your data files in the `data/` subdirectory:

```
examples/
├── _config.py                  # shared paths, column mapping, load_panel()
├── data/
│   ├── bond_panel.parquet      # bond-level panel
│   ├── factors.parquet         # risk-free rate
│   └── bbw_factors.parquet     # bond market factor (MKTB)
├── 00_pre_analysis.py
├── 01_single_sorts.py
│   ...
└── FF3/                        # Fama-French factor replication
```

---

## Scripts

### 00 — Pre-Analysis Statistics

**File:** `00_pre_analysis.py`

Descriptive statistics and data validation before portfolio formation. Showcases `PreAnalysisStats`, `validate_panel`, and `check_duplicates`.

**Features demonstrated:**
- Cross-sectional summary statistics (mean, median, percentiles)
- Time-series coverage plots (bonds per month, observations)
- Rating distribution and maturity subsets
- Issuer-level bond counts
- Signal availability checks

---

### 01 — Single-Sort Portfolios

**File:** `01_single_sorts.py`

The core portfolio formation workflow. Sort bonds into portfolios by a single characteristic and compute value-weighted and equal-weighted returns.

**Features demonstrated:**
- `SingleSort` with `StrategyFormation`
- Built-in strategies: `Momentum`, `LTreversal`
- Holding periods (HP=1, HP=3 staggered)
- Turnover computation
- Characteristics tracking (`chars` parameter)
- Banding (threshold-based rebalancing)
- Custom breakpoints and breakpoint universe filters
- Subset filters (e.g., maturity > 5 years)
- Non-monthly rebalancing (`rebalance_frequency='quarterly'`)
- `StrategyFormationConfig` for parameter bundles

---

### 02 — Double-Sort Portfolios

**File:** `02_double_sorts.py`

Bivariate portfolio sorts using two characteristics simultaneously.

**Features demonstrated:**
- `DoubleSort` with conditional and unconditional sorting
- 2D portfolio grids via `get_ptf()`
- Custom breakpoints on the second variable
- Rating subsample analysis

---

### 03 — Within-Firm Sorts

**File:** `03_within_firm_sorts.py`

Sort bonds within the same firm (issuer), isolating bond-level characteristics from firm-level effects.

**Features demonstrated:**
- `WithinFirmSort` (HP=1, 2 portfolios: HIGH/LOW)
- `BatchWithinFirmSortFormation` for multiple signals
- Custom rating bins
- Within-firm characteristics tracking
- Hierarchical aggregation: within-firm VW → across firms → across rating terciles

---

### 04 — Batch Formation

**File:** `04_batch_formation.py`

Process multiple signals in a single call for large-scale portfolio studies.

**Features demonstrated:**
- `BatchStrategyFormation` for 10+ signals at once
- Rating and maturity filtering (`rating`, `subset_filter`)
- `extract_panel` with `NamingConfig` for tidy output
- Banding and staggered holding periods in batch mode
- Ultra-fast numba batch path

---

### 05 — Rolling Beta Estimation

**File:** `05_rolling_betas.py`

Estimate time-varying factor exposures for individual bonds using rolling regressions.

**Features demonstrated:**
- `RollingBeta` with numba and numpy engines
- Univariate betas (single factor)
- Multi-factor models (e.g., BBW four factors)
- Idiosyncratic volatility extraction
- Sorting on estimated betas (beta-sorted portfolios)

---

### 06 — Anomaly Assaying

**File:** `06_anomaly_assaying.py`

Assess robustness of anomaly returns across methodological choices (Tier 2 specification curve). This example is organized in recommended public-API order: fast single-signal first, then fast multi-signal, then the richer slow-path workflow.

**Features demonstrated:**
- `assay_anomaly_fast` (single-signal quick assessment)
- `BatchAssayAnomaly` (multi-signal batch)
- `AssayAnomaly` (default slow-path anomaly workflow)
- `AssayAnomalyRunner` (advanced/internal runner control, not the default entry point)
- `SpecificationValidator` for input checking
- Specification dimensions: weighting × nport × breakpoint scheme × breakpoint universe × rating × maturity

---

### 07 — Data Uncertainty Analysis

**File:** `07_data_uncertainty.py`

Quantify sensitivity of portfolio returns to data-cleaning choices (Tier 1 specification curve).

**Features demonstrated:**
- `DataUncertaintyAnalysis` with filter grids
- Return trimming, price filters, bounce-back exclusion, winsorization
- EA (ex-ante) and EP (ex-post) return pairs
- Multi-signal and subsample (IG/NIG) analysis
- Summary tables with Newey-West t-statistics

---

### 08 — BBW Factor Replication

**File:** `08_bbw_factors.py`

Replicate the Bai, Bali & Wen (2019) four-factor model for corporate bonds: MKTB, DRF, CRF, LRF.

**Features demonstrated:**
- `DoubleSort` 5×5 unconditional sorts (RATING × signal)
- `get_ptf()` to extract 2D portfolio grids
- Long-short factor construction from double-sorted portfolios
- Lagged value-weighted market return (MKTB)
- Comparison with reference factors

---

### 09 — Beta-Sorted Portfolios

**File:** `09_beta_sorted_portfolios.py`

End-to-end workflow combining rolling beta estimation with portfolio formation and characteristics tracking. Uses CPI inflation betas as the running example.

**Features demonstrated:**
- `RollingBeta` → `SingleSort` → `StrategyFormation` pipeline
- ARMA(1,1) innovation extraction for factor construction
- Tracking multiple characteristics within sorted portfolios
- `get_characteristics()` to extract portfolio-level averages
- Newey-West t-statistics for long-short spreads

---

## FF3 — Fama-French Factor Replication

**Directory:** `FF3/`

Replication of Fama-French three factors (MKT, SMB, HML) adapted for corporate bond markets.

| File | Description |
|------|-------------|
| `download_data.py` | Download required WRDS data |
| `FamaFrench_factors.py` | Replicate FF3 factors and compare to published series |
