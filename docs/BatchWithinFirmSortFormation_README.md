# BatchWithinFirmSortFormation User Guide

`BatchWithinFirmSortFormation` is a high-performance tool for running within-firm portfolio sorts across **multiple signals** efficiently. Instead of running `StrategyFormation` with `WithinFirmSort` one signal at a time, `BatchWithinFirmSortFormation` processes all your signals in a single call with automatic optimization.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Full API Reference](#full-api-reference)
3. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Custom Column Names](#custom-column-names)
   - [Fast Path (Returns Only)](#fast-path-returns-only)
   - [With Turnover and Characteristics](#with-turnover-and-characteristics)
   - [Parallel Processing](#parallel-processing)
4. [Accessing Results](#accessing-results)
5. [Fast Path vs Slow Path](#fast-path-vs-slow-path)
6. [Performance](#performance)
7. [Comparison with StrategyFormation](#comparison-with-strategyformation)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from PyBondLab import BatchWithinFirmSortFormation

# Run within-firm sorts on multiple signals
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['credit_spread', 'yield_spread', 'duration'],
    firm_id_col='PERMNO',
    turnover=False,
    verbose=True
)
results = batch.fit()

# Access results for each signal
ew_ls, vw_ls = results['credit_spread'].get_long_short()
print(f"Credit Spread Sharpe: {vw_ls.mean() / vw_ls.std() * 12**0.5:.2f}")
```

---

## Full API Reference

```python
BatchWithinFirmSortFormation(
    data: pd.DataFrame,
    signals: List[str],
    firm_id_col: str = 'PERMNO',
    rating_bins: List[float] = [-np.inf, 7, 10, np.inf],
    min_bonds_per_firm: int = 2,
    turnover: bool = False,
    chars: List[str] = None,
    rating: Union[str, Tuple[int, int]] = None,
    subset_filter: Dict[str, Tuple[float, float]] = None,
    columns: Dict[str, str] = None,
    n_jobs: int = 1,
    verbose: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | **required** | Bond panel data with date, ID, returns, weights, rating, firm ID, and signal columns |
| `signals` | List[str] | **required** | Column names to use as sorting signals |
| `firm_id_col` | str | `'PERMNO'` | Column name for firm identifier |
| `rating_bins` | List[float] | `[-inf, 7, 10, inf]` | Rating bin edges for creating terciles (IG+/IG-/SG) |
| `min_bonds_per_firm` | int | `2` | Minimum bonds required per firm-date-rating group |
| `turnover` | bool | `False` | Compute turnover statistics (uses slow path) |
| `chars` | List[str] | `None` | Characteristics to aggregate (uses slow path) |
| `rating` | str/tuple | `None` | Rating filter: `'IG'`, `'NIG'`, or `(min, max)` tuple |
| `subset_filter` | Dict | `None` | Characteristic filter: `{'col': (min, max)}` |
| `columns` | Dict | `None` | Column name mapping (see [Custom Column Names](#custom-column-names)) |
| `n_jobs` | int | `1` | Number of parallel workers (for slow path only) |
| `verbose` | bool | `True` | Print progress messages |

### Returns

`Dict[str, StrategyResults]` - Dictionary mapping signal names to results objects.

---

## Examples

### Basic Usage

Process multiple signals with default settings:

```python
import pandas as pd
from PyBondLab import BatchWithinFirmSortFormation

# Your data should have these columns (at minimum):
# - date: datetime
# - ID: bond identifier
# - ret: monthly return
# - VW: value weight (market cap)
# - RATING_NUM: numeric rating (1-22)
# - PERMNO (or your firm_id_col): firm identifier
# - signal columns: your sorting variables

# Run batch processing
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['spread_1', 'spread_2', 'duration_spread', 'yield_spread'],
    firm_id_col='PERMNO',
    min_bonds_per_firm=2,
    turnover=False,
    verbose=True,
)
results = batch.fit()

# View results for each signal
for signal in batch.signals:
    ew_ls, vw_ls = results[signal].get_long_short()
    sharpe = vw_ls.mean() / vw_ls.std() * 12**0.5
    print(f"{signal}: VW Sharpe = {sharpe:.2f}")
```

**Output:**
```
FAST BATCH PATH: Processing 4 signals...
  [1/4] Processing spread_1...
  [2/4] Processing spread_2...
  [3/4] Processing duration_spread...
  [4/4] Processing yield_spread...
Batch processing complete: 4 signals processed

spread_1: VW Sharpe = 1.23
spread_2: VW Sharpe = 1.18
duration_spread: VW Sharpe = 0.87
yield_spread: VW Sharpe = 1.45
```

---

### Custom Column Names

If your data uses different column names than PyBondLab expects, use the `columns` parameter:

```python
# Your data has these columns:
# - date, cusip_id, ret_vw_bgn, mcap_e, spc_rat, firm_identifier, my_signal

batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['my_signal', 'other_signal'],
    firm_id_col='firm_identifier',      # Your firm ID column
    columns={
        'ID': 'cusip_id',               # Bond identifier
        'ret': 'ret_vw_bgn',            # Return column
        'VW': 'mcap_e',                 # Value weight
        'RATING_NUM': 'spc_rat',        # Rating column
    },
    turnover=False,
    verbose=True,
)
results = batch.fit()
```

**Default column mapping:**
```python
{
    'date': 'date',
    'ID': 'ID',
    'ret': 'ret',
    'VW': 'VW',
    'RATING_NUM': 'RATING_NUM',
}
```

Only specify columns that differ from defaults. The `firm_id_col` is specified separately as a constructor parameter.

---

### Fast Path (Returns Only)

When you only need long-short returns (no turnover or characteristics), the **fast batch path** is automatically used. This provides significant speedup:

```python
# Fast path is used automatically when:
# - turnover=False
# - chars=None

batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3', 'sig4', 'sig5',
             'sig6', 'sig7', 'sig8', 'sig9', 'sig10'],
    firm_id_col='PERMNO',
    turnover=False,      # Required for fast path
    chars=None,          # Required for fast path
    rating='IG',         # Rating filter works with fast path
    subset_filter={'MATURITY': (1, 5)},  # Subset filter works with fast path
    verbose=True,
)
results = batch.fit()
# Prints: "FAST BATCH PATH: Processing 10 signals..."
```

**What makes it fast:**
- Pre-computes rating terciles ONCE for all signals
- Pre-computes firm groupings ONCE for all signals
- Uses vectorized numba kernels for portfolio assignment
- Processes each signal with minimal overhead

---

### With Turnover and Characteristics

Track portfolio turnover and aggregate characteristics (uses slow path with multiprocessing):

```python
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['credit_spread', 'yield_spread'],
    firm_id_col='PERMNO',
    turnover=True,                       # Compute turnover (slow path)
    chars=['duration', 'spread'],        # Aggregate characteristics
    n_jobs=4,                            # Use 4 parallel workers
    verbose=True,
)
results = batch.fit()
# Prints: "SLOW BATCH PATH: Processing 2 signals with 4 workers..."

# Access turnover for a specific signal
ew_turn, vw_turn = results['credit_spread'].get_turnover()
print(f"Credit Spread EW turnover: {ew_turn.mean().mean():.1%}")

# Access characteristics
ew_chars, vw_chars = results['credit_spread'].get_characteristics()
print(ew_chars['duration'])  # DataFrame with LOW, HIGH columns
```

---

### Parallel Processing

For slow path with multiple signals, use parallel workers:

```python
# Use 4 workers for parallel processing
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3', 'sig4', 'sig5', 'sig6'],
    firm_id_col='PERMNO',
    turnover=True,         # Requires slow path
    n_jobs=4,              # Use 4 parallel workers
    verbose=True,
)
results = batch.fit()
```

**Recommended `n_jobs` settings:**

| Scenario | `n_jobs` | Notes |
|----------|----------|-------|
| Fast path | 1 | `n_jobs` ignored - numba parallelism used instead |
| Few signals (2-3) | 1 | Multiprocessing overhead not worth it |
| Many signals (4+), slow path | 2-4 | Good balance |
| Memory constrained | 1-2 | Limit concurrent workers |

---

### Rating and Subset Filters

Apply rating or characteristic-based filters:

```python
# Investment Grade bonds only
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['signal1', 'signal2'],
    firm_id_col='PERMNO',
    rating='IG',                        # Investment Grade (ratings 1-10)
    turnover=False,
    verbose=True,
)
results_ig = batch.fit()

# Custom rating range
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['signal1', 'signal2'],
    firm_id_col='PERMNO',
    rating=(7, 10),                     # BBB+ to BBB- only
    turnover=False,
    verbose=True,
)
results_bbb = batch.fit()

# Subset by characteristics
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['signal1', 'signal2'],
    firm_id_col='PERMNO',
    subset_filter={
        'MATURITY': (1, 5),             # Maturity 1-5 years
        'DURATION': (2, 8),             # Duration 2-8 years
    },
    turnover=False,
    verbose=True,
)
results_filtered = batch.fit()
```

---

### Custom Rating Bins

Customize the rating tercile definitions:

```python
import numpy as np

# Custom rating bins: [AAA-AA] [A-BBB] [BB and below]
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['signal1', 'signal2'],
    firm_id_col='PERMNO',
    rating_bins=[-np.inf, 4, 10, np.inf],   # Custom tercile boundaries
    min_bonds_per_firm=3,                    # Require 3+ bonds per firm
    turnover=False,
    verbose=True,
)
results = batch.fit()
```

**Default rating bins:** `[-inf, 7, 10, inf]`
- Tercile 1 (IG+): Ratings 1-7 (AAA to A-)
- Tercile 2 (IG-): Ratings 8-10 (BBB+ to BBB-)
- Tercile 3 (SG): Ratings 11+ (BB+ and below)

---

## Accessing Results

### Dictionary-like Access

```python
# Get results for a specific signal
cs_result = results['credit_spread']

# Check available signals
print(list(results.keys()))
# ['credit_spread', 'yield_spread', 'duration_spread']

# Iterate over all results
for signal, result in results.items():
    ew_ls, vw_ls = result.get_long_short()
    print(f"{signal}: VW mean = {vw_ls.mean()*12:.2%}")
```

### Long-Short Returns

```python
# Get long-short portfolio returns
ew_ls, vw_ls = results['credit_spread'].get_long_short()

# ew_ls is a pandas Series with DatetimeIndex
print(ew_ls.head())
# date
# 2010-01-31    0.0123
# 2010-02-28    0.0087
# 2010-03-31   -0.0045
# ...

# Compute statistics
print(f"Mean:   {vw_ls.mean() * 12:.2%}")      # Annualized mean
print(f"Std:    {vw_ls.std() * 12**0.5:.2%}")  # Annualized std
print(f"Sharpe: {vw_ls.mean() / vw_ls.std() * 12**0.5:.2f}")
```

### Portfolio Returns

```python
# Get HIGH and LOW portfolio returns
ew_port, vw_port = results['credit_spread'].get_portfolio_returns()

# Returns DataFrame with LOW, HIGH columns
print(vw_port.head())
#                   LOW      HIGH
# date
# 2010-01-31    0.0123    0.0234
# 2010-02-28    0.0087    0.0145
# ...
```

### Turnover (Slow Path Only)

```python
# Only available if turnover=True
if batch.turnover:
    ew_turn, vw_turn = results['credit_spread'].get_turnover()
    print(f"Average VW turnover: {vw_turn.mean().mean():.1%}")
else:
    print("Turnover not computed (fast path used)")
```

### Characteristics (Slow Path Only)

```python
# Only available if chars is specified
if batch.chars:
    ew_chars, vw_chars = results['credit_spread'].get_characteristics()
    # ew_chars = {'duration': DataFrame(LOW, HIGH), ...}
    print(ew_chars['duration'].head())
else:
    print("Characteristics not computed")
```

---

## Fast Path vs Slow Path

### Decision Logic

```
┌─────────────────────────────────────────────────────────┐
│                    BatchWithinFirmSortFormation         │
│                                                         │
│   turnover=False AND chars=None?                        │
│              │                                          │
│       ┌──────┴──────┐                                   │
│       │             │                                   │
│      YES           NO                                   │
│       │             │                                   │
│       ▼             ▼                                   │
│  ┌─────────┐   ┌──────────────┐                        │
│  │  FAST   │   │    SLOW      │                        │
│  │  PATH   │   │    PATH      │                        │
│  │         │   │              │                        │
│  │ - Numba │   │ - Multiproc  │                        │
│  │ - Single│   │ - n_jobs     │                        │
│  │   proc  │   │   workers    │                        │
│  │ - 33x+  │   │ - Full       │                        │
│  │   faster│   │   features   │                        │
│  └─────────┘   └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### Feature Comparison

| Feature | Fast Path | Slow Path |
|---------|-----------|-----------|
| **When used** | `turnover=False` AND `chars=None` | `turnover=True` OR `chars` specified |
| **Method** | Vectorized numba kernels | Multiprocessing with `n_jobs` workers |
| **Speed** | 33x+ faster | Baseline |
| **Turnover** | ❌ Not computed | ✅ Computed |
| **Characteristics** | ❌ Not computed | ✅ Computed |
| **Rating filter** | ✅ Supported | ✅ Supported |
| **Subset filter** | ✅ Supported | ✅ Supported |
| **Memory** | Lower (single process) | Higher (multiple processes) |
| **Parallelism** | Within numba (prange) | Python multiprocessing |

### How to Check Which Path is Used

```python
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['sig1', 'sig2'],
    firm_id_col='PERMNO',
    turnover=False,
    verbose=True       # Will print which path is used
)
results = batch.fit()

# Output will show:
# "FAST BATCH PATH: Processing 2 signals..."   (fast path)
# or
# "SLOW BATCH PATH: Processing 2 signals..."   (slow path)
```

---

## Performance

### Benchmark Results

| Configuration | 10 Signals | Notes |
|---------------|------------|-------|
| Fast path (turnover=False) | **~0.3s** | Numba vectorized |
| Slow path, n_jobs=1 | ~10s | Sequential |
| Slow path, n_jobs=4 | ~3s | Parallel |

*Test data: 11,940 rows, 199 bonds, 50 firms, 60 dates*

### Large Data Performance

| Dataset Size | 10 Signals (Fast) | 10 Signals (Slow, n_jobs=4) |
|--------------|-------------------|----------------------------|
| 100K rows | ~0.5s | ~30s |
| 1M rows | ~2s | ~150s |
| 2.4M rows | ~5s | ~400s |

### Optimization Tips

1. **Use fast path when possible**: If you only need returns, set `turnover=False` and `chars=None`

2. **For slow path, tune n_jobs**:
   - Few signals: `n_jobs=1` (multiprocessing overhead not worth it)
   - Many signals: `n_jobs=4` or more

3. **Memory-constrained environments**: Use `n_jobs=1` or `n_jobs=2` to limit concurrent workers

---

## Comparison with StrategyFormation

### When to Use Each

| Use Case | Recommended Approach |
|----------|---------------------|
| Single signal | `StrategyFormation` + `WithinFirmSort` |
| Multiple signals, returns only | `BatchWithinFirmSortFormation` (fast path) |
| Multiple signals, with turnover/chars | `BatchWithinFirmSortFormation` (slow path) |
| Need full control over each signal | Loop over `StrategyFormation` |

### Code Comparison

**Single signal (StrategyFormation):**
```python
import PyBondLab as pbl

strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='credit_spread',
    firm_id_col='PERMNO',
)
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
).fit()

ew_ls, vw_ls = result.get_long_short()
```

**Multiple signals (BatchWithinFirmSortFormation):**
```python
from PyBondLab import BatchWithinFirmSortFormation

batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['credit_spread', 'yield_spread', 'duration'],
    firm_id_col='PERMNO',
    turnover=False,
)
results = batch.fit()

for sig in batch.signals:
    ew_ls, vw_ls = results[sig].get_long_short()
```

### Result Consistency

Results from `BatchWithinFirmSortFormation` are **numerically identical** to running individual `StrategyFormation` calls:

```python
# Batch approach
batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['signal1'],
    firm_id_col='PERMNO',
    turnover=False,
)
batch_results = batch.fit()
batch_ew, batch_vw = batch_results['signal1'].get_long_short()

# Individual approach
strategy = pbl.WithinFirmSort(holding_period=1, sort_var='signal1', firm_id_col='PERMNO')
sf = pbl.StrategyFormation(data, strategy, turnover=False)
sf_results = sf.fit()
sf_ew, sf_vw = sf_results.get_long_short()

# Compare (should be 0.0)
diff = (batch_vw - sf_vw).abs().max()
print(f"Max difference: {diff:.2e}")  # 0.00e+00
```

---

## Troubleshooting

### Common Issues

**1. "Signal 'xxx' not found in data columns"**
```python
# Check that signal columns exist
missing = [s for s in signals if s not in data.columns]
print(f"Missing: {missing}")
```

**2. "Data missing required columns"**
```python
# Use columns parameter to map your column names
batch = BatchWithinFirmSortFormation(
    ...,
    columns={'ID': 'cusip', 'VW': 'mktcap', 'RATING_NUM': 'rating'},
)
```

**3. Empty results for some signals**
```python
# Check if signal has valid (non-NaN) values
for sig in signals:
    n_valid = data[sig].notna().sum()
    print(f"{sig}: {n_valid} valid values")
```

**4. Slow performance**
```python
# Check if fast path can be used
if batch.turnover == False and batch.chars is None:
    print("Fast path will be used!")
else:
    print("Slow path - consider disabling turnover/chars for speed")
```

### Getting Help

- Check verbose output for progress and path information
- Review the [WithinFirmSort_README.md](WithinFirmSort_README.md) for strategy details
- Review the [CLAUDE.md](../CLAUDE.md) for technical implementation details

---

## Complete Example

```python
import pandas as pd
import numpy as np
from PyBondLab import BatchWithinFirmSortFormation

# Load your data
data = pd.read_parquet('bond_data.parquet')

# Define signals to test
signals = [
    'credit_spread', 'oas_spread', 'yield_spread',
    'duration_spread', 'liquidity', 'momentum_3m',
]

# Fast screening (returns only)
print("=== Fast Path: Factor Screening ===")
batch_fast = BatchWithinFirmSortFormation(
    data=data,
    signals=signals,
    firm_id_col='PERMNO',
    min_bonds_per_firm=2,
    turnover=False,
    rating='IG',                    # Investment grade only
    verbose=True,
)
results_fast = batch_fast.fit()

# View summary
print("\nFactor Screening Results:")
for sig in signals:
    ew_ls, vw_ls = results_fast[sig].get_long_short()
    sharpe = vw_ls.mean() / vw_ls.std() * 12**0.5
    mean_ret = vw_ls.mean() * 12
    print(f"  {sig:20s}: Mean={mean_ret:6.2%}, Sharpe={sharpe:.2f}")

# Detailed analysis of top factors
print("\n=== Slow Path: Detailed Analysis ===")
top_signals = ['credit_spread', 'yield_spread']  # Based on screening

batch_detail = BatchWithinFirmSortFormation(
    data=data,
    signals=top_signals,
    firm_id_col='PERMNO',
    min_bonds_per_firm=2,
    turnover=True,                  # Track turnover
    chars=['duration', 'spread'],   # Aggregate characteristics
    rating='IG',
    n_jobs=2,
    verbose=True,
)
results_detail = batch_detail.fit()

# Analyze credit_spread in detail
cs_result = results_detail['credit_spread']
ew_ls, vw_ls = cs_result.get_long_short()
ew_turn, vw_turn = cs_result.get_turnover()
ew_chars, vw_chars = cs_result.get_characteristics()

print(f"\nCredit Spread Analysis:")
print(f"  Annualized Return: {vw_ls.mean() * 12:.2%}")
print(f"  Annualized Vol:    {vw_ls.std() * 12**0.5:.2%}")
print(f"  Sharpe Ratio:      {vw_ls.mean() / vw_ls.std() * 12**0.5:.2f}")
print(f"  Avg Turnover:      {vw_turn.mean().mean():.1%}")
print(f"  Avg Duration HIGH: {vw_chars['duration']['HIGH'].mean():.2f}")
print(f"  Avg Duration LOW:  {vw_chars['duration']['LOW'].mean():.2f}")
```

---

## Summary

`BatchWithinFirmSortFormation` provides:

1. **Efficient batch processing** of multiple signals with within-firm sorting
2. **Automatic fast path** (33x+ speedup) when only returns are needed
3. **Flexible column mapping** for custom data formats
4. **Rating and subset filters** compatible with both paths
5. **Parallel processing** for slow path with turnover/characteristics
6. **Consistent API** matching individual `StrategyFormation` results

For single-signal usage or more control, see [WithinFirmSort_README.md](WithinFirmSort_README.md).
