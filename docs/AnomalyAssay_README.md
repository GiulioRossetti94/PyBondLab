# Anomaly Assay User Guide

PyBondLab exposes three anomaly-assay entry points for normal users:

- `assay_anomaly_fast`: fast single-signal assay
- `BatchAssayAnomaly`: fast multi-signal assay
- `AssayAnomaly`: slower but richer workflow with fuller output

All three test factor robustness across **specification grids** spanning weighting schemes, portfolio structures, rating filters, breakpoint universes, and maturity filters. This is the package's Tier 2 robustness layer for methodological/specification uncertainty.

## Which Anomaly Tool Should I Use?

- Start with `assay_anomaly_fast` if you want one signal and a specification grid as fast as possible.
- Use `BatchAssayAnomaly` when you want the same fast workflow across many signals.
- Use `AssayAnomaly` when you need the slower, fuller workflow with bond counts, turnover-aware inspection, and easier downstream exploration.
- Treat `AssayAnomalyRunner` as advanced/internal. It exists for custom control, not as the default public starting point.

### Public API Map

| API | Recommended use | Output style | Default audience |
|---|---|---|---|
| `assay_anomaly_fast` | One signal, speed-first | `AnomalyAssayResult` with returns grid and summary | Most users |
| `BatchAssayAnomaly` | Many signals, speed-first | Batch wrapper around many `AnomalyAssayResult`s | Screening / multiverse runs |
| `AssayAnomaly` | One signal, richer slow-path inspection | `AnomalyResults` with fuller time-series panel | Users who want detail over speed |
| `AssayAnomalyRunner` | Custom runner control | Internal/advanced interface | Advanced users only |

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Key Concepts](#key-concepts)
   - [Specification Grid](#specification-grid)
   - [Breakpoint Universes](#breakpoint-universes)
   - [Invalid Specifications](#invalid-specifications)
3. [Full API Reference](#full-api-reference)
4. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Custom Column Names](#custom-column-names)
   - [Custom Breakpoints](#custom-breakpoints)
   - [Rating and Maturity Filters](#rating-and-maturity-filters)
5. [Accessing Results](#accessing-results)
   - [Returns DataFrame](#returns-dataframe)
   - [Summary Statistics](#summary-statistics)
6. [Specification Naming Convention](#specification-naming-convention)
7. [Performance](#performance)
8. [Troubleshooting](#troubleshooting)
9. [Complete Example](#complete-example)

---

## Quick Start

The recommended default is `assay_anomaly_fast` for a single signal:

```python
from PyBondLab import assay_anomaly_fast

# Define specification grid
specs = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        (5, 'quintiles', None),
        (3, 'extreme', [10, 90]),
    ],
    'rating_filters': {'all': None, 'ig': (1, 10)},
    'bp_universes': {'all': None},
    'maturity_filters': {'all': None, 'short': (0, 5)},
}

# Run anomaly assay
result = assay_anomaly_fast(
    data=data,
    signal='momentum',
    specs=specs,
    IDvar='cusip',
    DATEvar='date',
    RETvar='ret_vw',
    VWvar='mcap_s',
    RATINGvar='RATING_NUM',
)

# View summary statistics
print(result.summary())

# Access returns
returns_df = result.returns_df
```

---

## Key Concepts

### Specification Grid

A specification grid defines all combinations of:

| Dimension | Description | Example Values |
|-----------|-------------|----------------|
| **Weighting** | Portfolio weighting scheme | `'EW'` (equal), `'VW'` (value) |
| **Portfolio Structure** | Number of portfolios and breakpoints | `(5, 'quintiles', None)`, `(3, 'extreme', [10, 90])` |
| **Rating Filter** | Which bonds to include | `None` (all), `(1, 10)` (IG), `(11, 22)` (HY/NIG) |
| **BP Universe** | Which bonds define breakpoints | `None` (all), `lambda df: df['RATING_NUM'] <= 10` |
| **Maturity Filter** | Maturity range filter | `None` (all), `(0, 5)` (short), `(10, 100)` (long) |

**Total specifications** = weighting x structures x rating_filters x bp_universes x maturity_filters

### Breakpoint Universes

The `bp_universes` dimension controls which bonds are used to compute breakpoints:

```python
'bp_universes': {
    'all': None,                              # Use all bonds for breakpoints
    'ig_only': lambda df: df['RATING_NUM'] <= 10,  # Use only IG bonds
}
```

This is useful for testing whether results change when breakpoints are computed on a different universe than the sort universe.

### Invalid Specifications

Some combinations are invalid (e.g., `ig_only` breakpoints with `hy` rating filter creates disjoint populations). By default, these are automatically skipped with `skip_invalid=True`.

---

## Full API Reference

```python
assay_anomaly_fast(
    data: pd.DataFrame,
    signal: str,
    specs: Dict[str, Any],
    *,
    holding_period: int = 1,
    dynamic_weights: bool = True,
    validate: bool = False,
    validate_sample_size: int = 3,
    skip_invalid: bool = True,
    verbose: bool = True,
    IDvar: str = None,
    DATEvar: str = None,
    RETvar: str = None,
    VWvar: str = None,
    RATINGvar: str = None,
) -> AnomalyAssayResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | **required** | Bond panel data |
| `signal` | str | **required** | Column name of the sorting signal |
| `specs` | Dict | **required** | Specification grid dictionary (see below) |
| `holding_period` | int | 1 | Holding period in months |
| `dynamic_weights` | bool | True | Use VW from d-1 (True) or formation date (False) |
| `validate` | bool | False | Run validation against slow path |
| `validate_sample_size` | int | 3 | Number of specs to validate (random sample) |
| `skip_invalid` | bool | True | Skip invalid spec combinations |
| `verbose` | bool | True | Print progress messages |
| `IDvar` | str | 'ID' | Column name for bond identifier |
| `DATEvar` | str | 'date' | Column name for date |
| `RETvar` | str | 'ret' | Column name for returns |
| `VWvar` | str | 'VW' | Column name for value weights |
| `RATINGvar` | str | 'RATING_NUM' | Column name for numeric rating |

### Specs Dictionary Structure

```python
specs = {
    'weighting': List[str],           # ['EW', 'VW']
    'portfolio_structures': List[Tuple],  # [(n_ports, name, breakpoints), ...]
    'rating_filters': Dict[str, Any], # {'name': filter_value, ...}
    'bp_universes': Dict[str, Any],   # {'name': func_or_none, ...}
    'maturity_filters': Dict[str, Any],  # {'name': (min, max), ...}
}
```

### Returns

`AnomalyAssayResult` object with:
- `returns_df`: DataFrame of long-short returns (dates x specs)
- `metadata`: Dict with run information
- `summary()`: Method to compute t-statistics

If you need the slower, richer workflow instead, use `AssayAnomaly` from the main package API. Use `AssayAnomalyRunner` only when you explicitly need custom runner control.

---

## Examples

### Basic Usage

```python
import pandas as pd
from PyBondLab import assay_anomaly_fast

# Simple spec grid
specs = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        (5, 'quintiles', None),           # Standard quintiles
        (10, 'deciles', None),            # Standard deciles
    ],
    'rating_filters': {
        'all': None,                       # All bonds
    },
    'bp_universes': {
        'all': None,                       # Breakpoints from all bonds
    },
    'maturity_filters': {
        'all': None,                       # All maturities
    },
}

result = assay_anomaly_fast(
    data=data,
    signal='momentum',
    specs=specs,
    IDvar='cusip',
    DATEvar='date',
    RETvar='ret_vw',
    VWvar='mcap_s',
    RATINGvar='RATING_NUM',
)

print(f"Computed {result.returns_df.shape[1]} specifications")
print(result.summary())
```

---

### Custom Column Names

```python
# Your data has: date, bond_id, monthly_ret, mkt_cap, rating_num, signal_col

result = assay_anomaly_fast(
    data=data,
    signal='signal_col',
    specs=specs,
    IDvar='bond_id',      # Bond identifier
    DATEvar='date',       # Date column
    RETvar='monthly_ret', # Return column
    VWvar='mkt_cap',      # Value weight
    RATINGvar='rating_num',  # Rating column
)
```

---

### Custom Breakpoints

```python
specs = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        # Standard equal-spaced portfolios
        (3, 'terciles', None),            # 0-33-67-100
        (5, 'quintiles', None),           # 0-20-40-60-80-100
        (10, 'deciles', None),            # 0-10-20-...-100

        # Custom extreme breakpoints (3 portfolios with 10/90 cutoffs)
        (3, 'extreme_10_90', [10, 90]),   # Bottom 10%, Middle 80%, Top 10%

        # Moderate breakpoints
        (3, 'moderate_20_80', [20, 80]),  # Bottom 20%, Middle 60%, Top 20%

        # Quintiles with extreme tails
        (5, 'quint_extreme', [10, 30, 70, 90]),
    ],
    'rating_filters': {'all': None},
    'bp_universes': {'all': None},
    'maturity_filters': {'all': None},
}
```

**Breakpoint interpretation:**
- `None`: Equal percentile spacing (e.g., quintiles = 20, 40, 60, 80)
- `[10, 90]`: Custom percentiles creating 3 portfolios (0-10, 10-90, 90-100)
- `[10, 30, 70, 90]`: 5 portfolios with custom cutoffs

---

### Rating and Maturity Filters

```python
specs = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        (5, 'quintiles', None),
    ],
    'rating_filters': {
        'all': None,           # All bonds
        'ig': (1, 10),         # Investment Grade (AAA to BBB-)
        'hy': (11, 22),        # High Yield / NIG
        'bbb': (7, 10),        # BBB only
    },
    'bp_universes': {
        'all': None,
        'ig_only': lambda df: df['RATING_NUM'] <= 10,  # BP from IG only
    },
    'maturity_filters': {
        'all': None,           # All maturities
        'short': (0, 5),       # 0-5 years
        'mid': (5, 10),        # 5-10 years
        'long': (10, 100),     # 10+ years
    },
}

result = assay_anomaly_fast(
    data=data,
    signal='credit_spread',
    specs=specs,
    skip_invalid=True,  # Skip invalid combinations (e.g., ig_only BP + hy filter)
    verbose=True,
    IDvar='cusip',
    DATEvar='date',
    RETvar='ret_vw',
    VWvar='mcap_s',
    RATINGvar='RATING_NUM',
)

# Check how many specs were skipped
print(f"Valid specs: {result.returns_df.shape[1]}")
```

---

## Accessing Results

### Returns DataFrame

```python
# Get the returns DataFrame
returns_df = result.returns_df

# Shape: (n_dates, n_specs)
print(f"Shape: {returns_df.shape}")

# Index is dates, columns are spec IDs
print(returns_df.head())
#             EW_5p_quintiles_all_all_all  VW_5p_quintiles_all_all_all  ...
# date
# 2015-01-31                      0.0123                       0.0098  ...
# 2015-02-28                      0.0087                       0.0076  ...

# Access specific specification
spec_returns = returns_df['EW_5p_quintiles_all_all_all']

# Compute statistics
print(f"Mean: {spec_returns.mean():.4f}")
print(f"Std: {spec_returns.std():.4f}")
print(f"Sharpe: {spec_returns.mean() / spec_returns.std() * 12**0.5:.2f}")
```

### Summary Statistics

```python
# Get summary with t-statistics
summary = result.summary(lag=0)  # lag=0 for OLS, lag>0 for HAC

print(summary)
#                              spec_id  mean_ret   std_ret  t_stat  p_value  n_obs
# 0   EW_5p_quintiles_all_all_all       0.0123    0.0456    2.98    0.003    120
# 1   VW_5p_quintiles_all_all_all       0.0098    0.0412    2.62    0.010    120
# ...

# Sort by absolute t-statistic
best_specs = summary.nlargest(10, 't_stat')
print(best_specs[['spec_id', 'mean_ret', 't_stat', 'p_value']])

# Count significant specs
n_sig_5 = (summary['p_value'] < 0.05).sum()
n_sig_1 = (summary['p_value'] < 0.01).sum()
print(f"Significant at 5%: {n_sig_5}/{len(summary)}")
print(f"Significant at 1%: {n_sig_1}/{len(summary)}")
```

### Metadata

```python
# Access run metadata
print(result.metadata)
# {
#     'signal': 'momentum',
#     'n_specs': 120,
#     'n_specs_run': 108,  # After filtering invalid
#     'holding_period': 1,
#     'dynamic_weights': True,
#     'date_range': ('2015-01-31', '2024-12-31'),
#     'runtime_seconds': 5.23,
# }
```

---

## Specification Naming Convention

Spec IDs follow this pattern:
```
{weighting}_{n_ports}p_{bp_scheme}_{bp_universe}_{rating_filter}_{maturity_filter}
```

Examples:
- `EW_5p_quintiles_all_all_all` - EW, quintiles, all BP universe, all ratings, all maturities
- `VW_3p_extreme_10_90_ig_only_ig_short` - VW, 3 portfolios with 10/90 cutoffs, IG-only BP, IG filter, short maturity
- `EW_10p_deciles_all_hy_long` - EW, deciles, all BP, HY filter, long maturity

---

## Performance

The implementation uses numba-accelerated computations achieving high performance:

| Dataset Size | Specs | Time | Per Spec |
|-------------|-------|------|----------|
| 1M rows | 120 | ~6s | 50ms |
| 1M rows | 280 | ~10s | 36ms |

**Key optimizations:**
1. Single numpy conversion at start
2. Pre-computed filter masks
3. Vectorized numba threshold computation
4. Batched rank computation per filter group

---

## Troubleshooting

### Common Issues

**1. "Signal 'xxx' not found in data columns"**
```python
# Check signal exists
print(signal in data.columns)
print(data.columns.tolist())
```

**2. "No valid specifications to run"**
```python
# All specs are invalid - check your grid
from PyBondLab import get_valid_spec_list
valid_specs, result = get_valid_spec_list(specs, verbose=True)
print(f"Valid: {len(valid_specs)}, Invalid: {len(result.invalid_specs)}")
```

**3. Missing data in results**
```python
# Check for NaN in signal
print(f"Signal NaN: {data[signal].isna().sum()}")

# Check date range
print(f"Date range: {data['date'].min()} to {data['date'].max()}")
```

**4. Validation against slow path**
```python
# Use validate=True to compare against SingleSort slow path
result = assay_anomaly_fast(
    data=data,
    signal='momentum',
    specs=specs,
    validate=True,          # Enable validation
    validate_sample_size=5, # Number of specs to validate
    verbose=True,
    ...
)
```

---

## Complete Example

```python
import pandas as pd
import numpy as np
from PyBondLab import assay_anomaly_fast

# Load data
data = pd.read_parquet('bond_data.parquet')

# Define comprehensive spec grid
TIER2_SPECS = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        (3, 'terciles', None),
        (3, 'extreme_10_90', [10, 90]),
        (3, 'moderate_20_80', [20, 80]),
        (5, 'quintiles', None),
        (5, 'extreme', [10, 30, 70, 90]),
        (10, 'deciles', None),
    ],
    'rating_filters': {
        'all': None,
        'ig': (1, 10),
        'hy': (11, 22),
    },
    'bp_universes': {
        'all': None,
        'ig_only': lambda df: df['RATING_NUM'] <= 10,
    },
    'maturity_filters': {
        'all': None,
        'short': (0, 5),
        'mid': (5, 10),
        'long': (10, 100),
    },
}

# Signals to test
signals = ['cs', 'ytm', 'sze', 'ami', 'str', 'mom12_1']

# Run for each signal
results = {}
for signal in signals:
    if signal not in data.columns:
        print(f"Skipping {signal} (not in data)")
        continue

    print(f"\nProcessing {signal}...")
    result = assay_anomaly_fast(
        data=data,
        signal=signal,
        specs=TIER2_SPECS,
        holding_period=1,
        dynamic_weights=True,
        skip_invalid=True,
        verbose=False,
        IDvar='cusip',
        DATEvar='date',
        RETvar='ret_vw',
        VWvar='mcap_s',
        RATINGvar='RATING_NUM',
    )
    results[signal] = result

    # Quick summary
    summary = result.summary(lag=0)
    n_sig = (summary['p_value'] < 0.05).sum()
    best_t = summary['t_stat'].abs().max()
    print(f"  {result.returns_df.shape[1]} specs | {n_sig} sig (5%) | best |t|={best_t:.2f}")

# Aggregate analysis
print("\n" + "="*60)
print("SIGNAL COMPARISON")
print("="*60)

for signal, result in results.items():
    summary = result.summary(lag=0)
    mean_t = summary['t_stat'].abs().mean()
    pct_sig = (summary['p_value'] < 0.05).mean() * 100
    best_idx = summary['t_stat'].abs().idxmax()
    best_spec = summary.loc[best_idx, 'spec_id']
    print(f"{signal:<12} mean|t|={mean_t:.2f}, {pct_sig:.0f}% sig, best: {best_spec}")
```

---

## Summary

`assay_anomaly_fast` provides:

1. **Comprehensive specification testing** across weighting, portfolios, ratings, breakpoints, and maturities
2. **Automatic invalid spec detection** to skip impossible combinations
3. **OLS and HAC t-statistics** via the `summary()` method
4. **High performance** with numba-accelerated computations
5. **Flexible column mapping** for any data format

For processing multiple signals in parallel, see `BatchAssayAnomaly`.
