# PreAnalysisStats User Guide

`PreAnalysisStats` computes cross-sectional summary statistics for panel data before portfolio formation. For each time period, it computes the distribution of specified variables (mean, std, skewness, kurtosis, percentiles) and aggregates these statistics over time. Supports optional return-based filtering, rating filters, and characteristic subset filters.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Reference](#api-reference)
   - [PreAnalysisStats Parameters](#preanalysisstats-parameters)
   - [compute() Method](#compute-method)
   - [PreAnalysisResult Methods](#preanalysisresult-methods)
3. [Filtering](#filtering)
   - [Return Filters](#return-filters)
   - [Rating Filters](#rating-filters)
   - [Characteristic Subset Filters](#characteristic-subset-filters)
4. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Multiple Variables](#multiple-variables)
   - [With Issuer Statistics](#with-issuer-statistics)
   - [With Return Filtering](#with-return-filtering)
   - [With Rating Filter](#with-rating-filter)
   - [Comparing Raw vs Filtered](#comparing-raw-vs-filtered)
   - [LaTeX Export](#latex-export)
   - [Visualization](#visualization)
5. [Complete Example](#complete-example)
6. [Summary](#summary)

---

## Quick Start

```python
from PyBondLab.describe import PreAnalysisStats

# Compute summary statistics for duration
stats = PreAnalysisStats(data=bond_data, variables='duration')
result = stats.compute()

# View summary table
print(result.summary())

# Cross-sectional stats over time
cs_df = result.get_cs_stats()

# Export to LaTeX
latex = result.to_latex()
```

---

## API Reference

### PreAnalysisStats Parameters

```python
from PyBondLab.describe import PreAnalysisStats

PreAnalysisStats(
    data: pd.DataFrame,                    # Panel data
    variables: str | list[str],            # Variable(s) to analyze
    date_col: str = 'date',               # Date column
    id_col: str = 'ID',                   # Entity identifier column
    issuer_col: str | None = None,        # Issuer column (for issuer counts)
    percentiles: list[float] = [5, 25, 50, 75, 95],  # Percentiles to compute
    filter_type: str | None = None,       # 'trim', 'wins', 'price', or 'bounce'
    filter_value: float | list = None,    # Filter threshold(s)
    filter_location: str = 'both',        # For winsorize: 'both', 'left', 'right'
    ret_col: str = 'ret',                 # Return column (for filtering)
    rating: str | tuple | None = None,    # 'IG', 'NIG', or (min, max) bounds
    subset_filter: dict | None = None,    # Characteristic filters
    rating_col: str = 'RATING_NUM',       # Numeric rating column
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | *required* | Panel data with observations over time |
| `variables` | str or list | *required* | Variable(s) to compute statistics for |
| `date_col` | str | `'date'` | Date column name |
| `id_col` | str | `'ID'` | Entity identifier column name |
| `issuer_col` | str or None | `None` | Issuer identifier (e.g., `'PERMNO'`). If provided, computes issuer-level counts |
| `percentiles` | list | `[5, 25, 50, 75, 95]` | Percentiles to compute (values between 0 and 100) |
| `filter_type` | str or None | `None` | Return filter type: `'trim'`, `'wins'`, `'price'`, or `'bounce'` |
| `filter_value` | float or list | `None` | Filter threshold(s). Required when `filter_type` is set |
| `filter_location` | str | `'both'` | Winsorize location: `'both'`, `'left'`, or `'right'` |
| `ret_col` | str | `'ret'` | Return column name (used for filtering) |
| `rating` | str, tuple, or None | `None` | Rating filter: `'IG'` (1-10), `'NIG'` (11-22), `'all'`, or custom `(min, max)` |
| `subset_filter` | dict or None | `None` | Characteristic filters as `{col: (min, max)}` pairs |
| `rating_col` | str | `'RATING_NUM'` | Numeric rating column name |

### compute() Method

```python
result = stats.compute(
    include_nw: bool = False,  # Compute Newey-West t-statistics
    nw_lag: int = 0,           # Lags for Newey-West standard errors
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_nw` | bool | `False` | Compute Newey-West t-statistics for time-series aggregates |
| `nw_lag` | int | `0` | Number of lags for Newey-West standard errors |

**Returns:** `PreAnalysisResult` object.

### PreAnalysisResult Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary(variable=None)` | DataFrame | Time-series aggregates (mean, std, median, min, max of CS stats) |
| `get_cs_stats(variable=None)` | DataFrame | Cross-sectional statistics for each period (dates as index) |
| `get_ts_stats(variable=None)` | DataFrame | Time-series aggregates (simple, without NW) |
| `compare(other, variable=None)` | DataFrame | Side-by-side comparison with another result |
| `to_latex(variable=None, caption=None, label=None, precision=3)` | str | LaTeX table export |
| `plot(variable=None, stats=('mean', 'std'), figsize=(12,6))` | Figure | Plot CS statistics over time |
| `to_dict()` | dict | Convert results to dictionary |

**Static method:**

| Method | Returns | Description |
|--------|---------|-------------|
| `PreAnalysisResult.compare_results(raw, filtered, variable=None)` | DataFrame | Compare raw vs filtered results side by side |

---

## Filtering

### Return Filters

Exclude or adjust observations based on return characteristics. When a filter is applied, bonds with filtered-out returns are excluded from the analysis.

> **Note:** Filters here are applied contemporaneously (ex-post) — a bond is excluded based on its return or price in the *same* month. This is appropriate for descriptive statistics, where the goal is to characterize the cross-section after removing outliers. The ex-ante vs ex-post distinction only matters for portfolio formation (see `StrategyFormation`, which provides both `get_long_short()` for ex-ante and `get_long_short_ex_post()` for ex-post factor returns).

| Filter | Description | Example `filter_value` |
|--------|-------------|----------------------|
| `trim` | Exclude extreme returns | `0.5` (exclude \|ret\| > 50%) or `[-0.5, 0.5]` |
| `wins` | Winsorize tails at percentiles | `1.0` (1st/99th percentile) |
| `price` | Exclude extreme prices | `[20, 150]` (exclude price < 20 or > 150) |
| `bounce` | Exclude return reversals | `0.01` |

### Rating Filters

| Rating | RATING_NUM Range | Description |
|--------|------------------|-------------|
| `'IG'` | 1–10 | Investment grade bonds |
| `'NIG'` | 11–22 | Non-investment grade (high yield) |
| `(min, max)` | Custom | Custom rating range |
| `'all'` / `None` | All | No rating filter |

### Characteristic Subset Filters

Filter data by characteristic ranges before computing statistics:

```python
stats = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    subset_filter={
        'maturity': (1, 30),     # 1 to 30 years
        'duration': (0, 15),     # 0 to 15 years
    }
)
```

---

## Examples

### Basic Usage

```python
from PyBondLab.describe import PreAnalysisStats

stats = PreAnalysisStats(data=bond_data, variables='duration')
result = stats.compute()

# Time-series averages of cross-sectional statistics
print(result.summary())
#           mean    std  median    min    max
# mean      5.23   3.12   4.78   0.10  25.30
# std       0.45   0.28   0.41   0.02   2.10
# ...
```

### Multiple Variables

```python
stats = PreAnalysisStats(
    data=bond_data,
    variables=['duration', 'maturity', 'credit_spread'],
)
result = stats.compute()

# Access each variable
print(result.summary('duration'))
print(result.summary('credit_spread'))
```

### With Issuer Statistics

```python
stats = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    issuer_col='PERMNO',
)
result = stats.compute()

# Summary includes n_issuers and bonds_per_issuer
print(result.summary())
```

### With Return Filtering

```python
# Trim: exclude bonds with |ret| > 50%
stats_trim = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    filter_type='trim',
    filter_value=[-0.5, 0.5],
)
result_trim = stats_trim.compute()

# Winsorize: cap at 1st/99th percentile
stats_wins = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    filter_type='wins',
    filter_value=1.0,
    filter_location='both',
)
result_wins = stats_wins.compute()
```

### With Rating Filter

```python
# Investment grade only
stats_ig = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    rating='IG',
)
result_ig = stats_ig.compute()

# High yield only
stats_hy = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    rating='NIG',
)
result_hy = stats_hy.compute()
```

### Comparing Raw vs Filtered

```python
# Compute both raw and filtered
raw = PreAnalysisStats(data=bond_data, variables='duration').compute()
filtered = PreAnalysisStats(
    data=bond_data,
    variables='duration',
    filter_type='trim',
    filter_value=[-0.5, 0.5],
).compute()

# Compare side by side
comparison = PreAnalysisResult.compare_results(raw, filtered)
print(comparison)
#          raw  filtered   diff  pct_diff
# mean    5.23      5.21  0.02      0.38
# std     3.12      2.98  0.14      4.49
# ...

# Or use the instance method
comparison = raw.compare(filtered)
```

### LaTeX Export

```python
result = stats.compute()

# Basic export
latex = result.to_latex(
    variable='duration',
    caption='Summary Statistics: Bond Duration',
    label='tab:duration_stats',
    precision=3,
)
print(latex)
```

### Visualization

```python
result = stats.compute()

# Plot mean and std over time
fig = result.plot('duration', stats=('mean', 'std'))

# Plot with custom statistics
fig = result.plot('duration', stats=('mean', 'p50', 'skew'))
```

---

## Complete Example

```python
import pandas as pd
from PyBondLab.describe import PreAnalysisStats
from PyBondLab.describe.results import PreAnalysisResult
from PyBondLab.pbl_test import generate_synthetic_data

# ================================================================
# Generate sample data
# ================================================================
data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)

# ================================================================
# Example 1: Basic summary statistics
# ================================================================
print("=" * 60)
print("Example 1: Basic Summary Statistics")
print("=" * 60)

stats = PreAnalysisStats(
    data=data,
    variables=['signal1', 'signal2'],
)
result = stats.compute()

print(result.summary('signal1'))

# ================================================================
# Example 2: With Newey-West t-statistics
# ================================================================
print("\n" + "=" * 60)
print("Example 2: With Newey-West t-statistics")
print("=" * 60)

result_nw = stats.compute(include_nw=True, nw_lag=3)
print(result_nw.summary('signal1'))

# ================================================================
# Example 3: Impact of trimming on characteristics
# ================================================================
print("\n" + "=" * 60)
print("Example 3: Raw vs Trimmed")
print("=" * 60)

raw = PreAnalysisStats(data=data, variables='signal1').compute()
trimmed = PreAnalysisStats(
    data=data,
    variables='signal1',
    filter_type='trim',
    filter_value=0.5,
).compute()

comparison = PreAnalysisResult.compare_results(raw, trimmed, 'signal1')
print(comparison)

# ================================================================
# Example 4: Cross-sectional stats over time
# ================================================================
print("\n" + "=" * 60)
print("Example 4: Cross-Sectional Stats Over Time")
print("=" * 60)

cs_df = result.get_cs_stats('signal1')
print(f"Shape: {cs_df.shape}")
print(cs_df.head())

# ================================================================
# Example 5: Export to LaTeX
# ================================================================
latex = result.to_latex(
    variable='signal1',
    caption='Pre-Analysis Summary: Signal 1',
    label='tab:signal1_stats',
)
print(f"\nLaTeX output ({len(latex)} characters)")
```

---

## Summary

| Feature | Description |
|---------|-------------|
| Cross-sectional stats | Mean, std, skewness, kurtosis, min, max, percentiles per period |
| Time-series aggregates | Average of CS statistics across all periods |
| Newey-West | Optional NW t-statistics for time-series aggregates |
| Return filters | Trim, winsorize, price, bounce — see how filtering affects distributions |
| Rating filters | IG, NIG, or custom bounds |
| Subset filters | Filter by characteristic ranges (e.g., duration, maturity) |
| Issuer stats | Count unique issuers and bonds per issuer |
| Comparison | Compare raw vs filtered results side by side |
| LaTeX export | Publish-ready table output |
| Visualization | Plot CS statistics over time |

`PreAnalysisStats` helps understand the cross-sectional distribution of bond characteristics before portfolio formation, and quantify how data filters affect those distributions.
