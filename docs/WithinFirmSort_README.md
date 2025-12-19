# WithinFirmSort Strategy

## Overview

`WithinFirmSort` implements a within-firm high-low sorting methodology for constructing bond factors. Unlike standard cross-sectional sorting (SingleSort), this strategy sorts bonds **within each firm**, isolating within-firm bond dispersion from cross-firm differences.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Key Differences from SingleSort](#key-differences-from-singlesort)
3. [Methodology](#methodology)
4. [Full API Reference](#full-api-reference)
5. [Custom Column Names](#custom-column-names)
6. [Fast Path vs Slow Path](#fast-path-vs-slow-path)
7. [Feature Support](#feature-support)
8. [Performance](#performance)
9. [Examples](#examples)
10. [Architecture](#architecture)
11. [Validation](#validation)

---

## Quick Start

```python
import PyBondLab as pbl

# Initialize WithinFirmSort strategy
strategy = pbl.WithinFirmSort(
    holding_period=1,              # Monthly rebalancing
    sort_var='CS',                 # Sort on credit spread
    firm_id_col='PERMNO',          # Firm identifier column
    min_bonds_per_firm=2,          # Require at least 2 bonds per firm
)

# Run strategy formation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    verbose=True
).fit()

# Get long-short returns
ew_ls, vw_ls = result.get_long_short()
print(f"Mean return: {vw_ls.mean()*100:.3f}% per month")
```

---

## Key Differences from SingleSort

| Aspect | SingleSort | WithinFirmSort |
|--------|-----------|----------------|
| **Grouping** | None (cross-sectional) | Date × Rating Tercile × Firm |
| **Percentiles** | Global (e.g., 20/40/60/80 for quintiles) | Within-firm (33.3/66.7) |
| **Portfolios** | N portfolios (typically 5) | 2 (HIGH/LOW only) |
| **Return Aggregation** | Simple VW average across bonds | Firm-cap-weighted → Rating-averaged |

---

## Methodology

### Step 1: Rating Tercile Assignment

Bonds are grouped into rating terciles:
- **IG+ (Tercile 1)**: Ratings 1-7 (AAA to A-)
- **IG- (Tercile 2)**: Ratings 8-10 (BBB+ to BBB-)
- **SG (Tercile 3)**: Ratings 11+ (BB+ and below)

Custom bins can be specified via `rating_bins` parameter.

### Step 2: Within-Firm Portfolio Formation

For each (date, rating tercile, firm) group:
1. Require minimum `min_bonds_per_firm` bonds (default: 2)
2. Compute 33.3rd and 66.7th percentile thresholds of the signal
3. Assign bonds:
   - Signal < 33.3rd percentile → **Low portfolio (Q1)**
   - Signal > 66.7th percentile → **High portfolio (Q2)**
   - Middle tercile bonds → **Unassigned** (excluded)

### Step 3: Hierarchical Return Aggregation

Returns are aggregated in a hierarchical manner:

```
For each date:
    For each rating tercile (1, 2, 3):
        For each firm in this rating tercile:
            - Compute VW return for HIGH portfolio (Q2)
            - Compute VW return for LOW portfolio (Q1)
            - Compute firm-level H-L factor = Q2 - Q1

        Aggregate across firms (cap-weighted):
            rating_factor = Σ(firm_weight × firm_HL) / Σ(firm_weight)

    Average across rating terciles:
        overall_factor = mean(rating_factors)
```

---

## Full API Reference

### WithinFirmSort Strategy

```python
pbl.WithinFirmSort(
    holding_period: int,
    sort_var: str,
    firm_id_col: str = 'PERMNO',
    min_bonds_per_firm: int = 2,
    rating_bins: list = [-np.inf, 7, 10, np.inf],
    num_portfolios: int = 2,
    rebalance_frequency: str = 'monthly',
    rebalance_month: int = 6,
    verbose: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holding_period` | int | **required** | Holding period (must be 1 for now) |
| `sort_var` | str | **required** | Signal column name (e.g., 'CS', 'eff_yld') |
| `firm_id_col` | str | `'PERMNO'` | Column name for firm identifier |
| `min_bonds_per_firm` | int | `2` | Minimum bonds required per firm-date-rating group |
| `rating_bins` | list | `[-inf, 7, 10, inf]` | Rating bin edges for terciles |
| `num_portfolios` | int | `2` | Always 2 (HIGH/LOW), other values trigger warning |
| `rebalance_frequency` | str/int | `'monthly'` | Rebalancing frequency |
| `rebalance_month` | int/list | `6` | Month(s) for non-monthly rebalancing |
| `verbose` | bool | `True` | Print initialization details |

---

## Custom Column Names

If your data uses different column names than PyBondLab expects, use the `fit()` method parameters:

```python
import PyBondLab as pbl

strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='credit_spread',        # Your signal column name
    firm_id_col='firm_identifier',   # Your firm ID column name
)

result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
).fit(
    IDvar='cusip_id',         # Bond identifier (default: 'ID')
    RETvar='ret_vw_bgn',      # Return column (default: 'ret')
    VWvar='mcap_e',           # Value weight column (default: 'VW')
    RATINGvar='spc_rat',      # Rating column (default: 'RATING_NUM')
    DATEvar='date',           # Date column (default: 'date')
    PRICEvar='prc_eom',       # Price column (default: 'PRICE', optional)
)

ew_ls, vw_ls = result.get_long_short()
```

### Default Column Names

| Parameter | Default Name | Description |
|-----------|--------------|-------------|
| `IDvar` | `'ID'` | Bond identifier (e.g., CUSIP) |
| `DATEvar` | `'date'` | Date column |
| `RETvar` | `'ret'` | Monthly return |
| `VWvar` | `'VW'` | Value weight (market value) |
| `RATINGvar` | `'RATING_NUM'` | Numeric rating (1=AAA, ..., 21=D) |
| `PRICEvar` | `'PRICE'` | Price (optional, for price filters) |

**Note:** The `firm_id_col` is specified in the `WithinFirmSort` constructor, not in `fit()`.

### Complete Example with Custom Columns

```python
import PyBondLab as pbl
import pandas as pd

# Your data has non-standard column names
data = pd.read_parquet('my_bond_data.parquet')
print(data.columns)
# ['date', 'cusip_id', 'ret_vw_bgn', 'mcap_e', 'spc_rat', 'firm_id', 'credit_spread']

# Initialize strategy with your firm ID and signal column names
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='credit_spread',     # Your signal column
    firm_id_col='firm_id',        # Your firm identifier column
    min_bonds_per_firm=2,
    verbose=True
)

# Run with column mapping in fit()
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
    chars=['credit_spread'],      # Aggregate this characteristic
    verbose=True
)

result = sf.fit(
    IDvar='cusip_id',
    RETvar='ret_vw_bgn',
    VWvar='mcap_e',
    RATINGvar='spc_rat'
)

# Access results
ew_ls, vw_ls = result.get_long_short()
print(f"VW Sharpe: {vw_ls.mean() / vw_ls.std() * 12**0.5:.2f}")
```

---

## Fast Path vs Slow Path

WithinFirmSort automatically uses a fast numba-optimized path when conditions are met.

### When Fast Path is Used

The **fast path** (33x speedup) is automatically used when ALL of these conditions are met:

| Condition | Required Value |
|-----------|---------------|
| `holding_period` | `1` |
| `turnover` | `False` |
| `chars` | `None` |
| `rebalance_frequency` | `'monthly'` |

### When Slow Path is Used

The **slow path** is used when ANY of these conditions apply:

- `holding_period > 1` (currently disabled, raises error)
- `turnover=True`
- `chars` is specified
- Non-monthly rebalancing

### How to Check Which Path is Used

```python
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,      # Fast path eligible
    chars=None,          # Fast path eligible
    verbose=True         # Will print which path is used
)
result = sf.fit()

# With verbose=True, you'll see:
# "Using ULTRA-FAST WithinFirmSort returns-only path..." (fast path)
# OR
# "Starting portfolio formation..." (slow path)
```

### Performance Comparison

| Configuration | Fast Path | Slow Path | Speedup |
|---------------|-----------|-----------|---------|
| HP=1, no turnover, no chars | **0.03s** | 1.0s | **33x** |
| HP=1, with turnover | N/A | 0.94s | (slow only) |
| HP=1, with chars | N/A | ~1.0s | (slow only) |

---

## Feature Support

| Feature | Supported | Path Used | Notes |
|---------|-----------|-----------|-------|
| **Turnover** | ✅ YES | Slow | Uses optimized Phase 4 numba kernels |
| **Characteristics** | ✅ YES | Slow | Hierarchical aggregation (same as returns) |
| **HP=1** | ✅ YES | Fast or Slow | Full support |
| **HP>1 (Staggered)** | ❌ DISABLED | N/A | Raises ValueError (known bugs) |
| **Banding** | ❌ NO | N/A | Not applicable (see below) |
| **Rating filter** | ✅ YES | Both | Via subset_filter or rating parameter |

### Why No Banding?

WithinFirmSort only has 2 portfolios (HIGH and LOW). Banding prevents reassignment
when a bond's rank changes by less than `banding/nport`. With `nport=2` and typical
`banding=1`, this would require a rank change of 0.5 (i.e., moving from one portfolio
to the other), which is always the case when rank changes between HIGH and LOW.
Therefore, banding is meaningless for WithinFirmSort and is not implemented.

### Why HP>1 Disabled?

HP>1 (staggered rebalancing) has known bugs in the cohort averaging logic and is
currently disabled. Attempting to use `holding_period > 1` will raise a `ValueError`.

---

## Performance

### Current Performance (Phase 16 Optimizations)

| Configuration | Time | Notes |
|---------------|------|-------|
| HP=1, turnover=False, chars=None | **0.03s** | Fast path (33x speedup) |
| HP=1, turnover=True | 0.94s | Slow path with optimized turnover |
| HP=1, chars=['char1', 'char2'] | ~1.0s | Slow path with chars aggregation |

*Test data: 11,940 rows, 199 bonds, 50 firms, 60 dates*

### Large Data Performance

| Dataset Size | Fast Path | Slow Path |
|--------------|-----------|-----------|
| 100K rows | ~0.1s | ~3s |
| 1M rows | ~0.5s | ~15s |
| 2.4M rows | ~1s | ~35s |

---

## Examples

### Basic Usage

```python
import PyBondLab as pbl

# Initialize strategy
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',                 # Credit spread
    firm_id_col='PERMNO',
    min_bonds_per_firm=2,
)

# Run formation (fast path)
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    verbose=True
).fit()

# Get results
ew_ls, vw_ls = result.get_long_short()
print(f"Mean: {vw_ls.mean()*12:.2%}")
print(f"Sharpe: {vw_ls.mean()/vw_ls.std()*12**0.5:.2f}")
```

### With Turnover

```python
# Slow path with turnover computation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,               # Enables turnover tracking
    verbose=True
).fit()

# Get turnover
ew_turn, vw_turn = result.get_turnover()
print(f"Average turnover: {vw_turn.mean().mean():.1%}")
```

### With Characteristics

```python
# Slow path with characteristics aggregation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    chars=['duration', 'spread', 'rating'],  # Aggregate these
    verbose=True
).fit()

# Get characteristics (tuple of dicts)
ew_chars, vw_chars = result.get_characteristics()

# Access specific characteristic
print(ew_chars['duration'])  # DataFrame with LOW, HIGH columns
print(vw_chars['spread'])
```

### Custom Rating Bins

```python
import numpy as np

# Custom rating terciles
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',
    firm_id_col='PERMNO',
    # Custom bins: [AAA-AA] [A-BBB] [BB and below]
    rating_bins=[-np.inf, 4, 10, np.inf],
    min_bonds_per_firm=3,        # Require 3+ bonds per firm
)
```

### Comparison with SingleSort

```python
import PyBondLab as pbl

# WithinFirmSort - isolates within-firm dispersion
wfs = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',
    firm_id_col='PERMNO',
    verbose=False
)
result_wfs = pbl.StrategyFormation(data, strategy=wfs, turnover=False).fit()
ew_wfs, vw_wfs = result_wfs.get_long_short()

# SingleSort - cross-sectional sorting
ss = pbl.SingleSort(
    holding_period=1,
    sort_var='CS',
    num_portfolios=5,
    verbose=False
)
result_ss = pbl.StrategyFormation(data, strategy=ss, turnover=False).fit()
ew_ss, vw_ss = result_ss.get_long_short()

# Compare results
print(f"WithinFirmSort VW: {vw_wfs.mean()*12:.2%} (annual)")
print(f"SingleSort VW:     {vw_ss.mean()*12:.2%} (annual)")
print(f"Correlation:       {vw_wfs.corr(vw_ss):.3f}")
```

---

## Architecture

### File Structure

```
PyBondLab/
├── StrategyClass.py       # WithinFirmSort class definition
├── utils_within_firm.py   # Core within-firm computation functions
├── numba_core.py          # Fast path numba kernels
├── precompute.py          # Integration with precomputation (line 370+)
└── PyBondLab.py           # Integration with aggregation (line 2185+)
```

### Key Functions

1. **`WithinFirmSort`** (StrategyClass.py)
   - Strategy class with parameters and validation

2. **`compute_within_firm_portfolios()`** (utils_within_firm.py)
   - Creates rating terciles
   - Groups by (date, rating_terc, firm)
   - Calls numba kernel for portfolio assignment

3. **`compute_within_firm_assignments_numba()`** (utils_within_firm.py)
   - Numba-compiled core
   - Computes within-firm percentile thresholds
   - Assigns bonds to HIGH/LOW portfolios

4. **`_fit_withinfirm_fast()`** (PyBondLab.py)
   - Ultra-fast path bypassing pandas
   - Direct numpy/numba computation

5. **`_aggregate_within_firm_results()`** (PyBondLab.py)
   - Hierarchical return aggregation
   - Characteristics aggregation

6. **`compute_within_firm_aggregation_fast()`** (numba_core.py)
   - Numba kernel for return aggregation
   - Handles EW and VW separately

7. **`compute_within_firm_chars_aggregation()`** (numba_core.py)
   - Numba kernel for characteristics aggregation
   - Uses same hierarchical structure as returns

---

## Validation

### Run Validation Script

```bash
python examples/validate_withinfirmsort.py
```

This validates:
1. Basic execution (fast and slow paths)
2. Turnover computation
3. Characteristics aggregation
4. Fast path vs slow path consistency
5. Difference from SingleSort (confirming within-firm logic)

### Manual Validation

```python
import PyBondLab as pbl

# Run both paths and compare
strategy = pbl.WithinFirmSort(holding_period=1, sort_var='signal')

# Fast path
result_fast = pbl.StrategyFormation(
    data, strategy, turnover=False, verbose=True
).fit()

# Force slow path by enabling turnover
result_slow = pbl.StrategyFormation(
    data, strategy, turnover=True, verbose=True
).fit()

# Compare returns (should match exactly)
ew_fast, vw_fast = result_fast.get_long_short()
ew_slow, vw_slow = result_slow.get_long_short()

diff = (vw_fast - vw_slow).abs().max()
print(f"Max difference: {diff:.2e}")  # Should be ~0
```

---

## Summary

`WithinFirmSort` provides:

1. **Within-firm sorting** that isolates firm-level bond dispersion
2. **Hierarchical aggregation** (firm-cap-weighted → rating-averaged)
3. **Fast path** (33x speedup) for returns-only computation
4. **Full feature support** for turnover and characteristics
5. **Custom column mapping** via `fit()` parameters

For batch processing of multiple signals, see [BatchWithinFirmSortFormation_README.md](BatchWithinFirmSortFormation_README.md).
