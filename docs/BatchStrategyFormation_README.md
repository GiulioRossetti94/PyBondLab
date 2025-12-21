# BatchStrategyFormation User Guide

`BatchStrategyFormation` is a high-performance tool for running portfolio sorts across **multiple signals** efficiently. Instead of running `StrategyFormation` one signal at a time, `BatchStrategyFormation` processes all your signals in a single call with automatic parallelization.

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
   - [Memory Management for Large Datasets](#memory-management-for-large-datasets)
4. [Accessing Results](#accessing-results)
   - [Unified Panel Extraction with extract_panel](#unified-panel-extraction-with-extract_panel)
5. [Performance Optimization](#performance-optimization)
6. [When to Use Fast Path vs Slow Path](#when-to-use-fast-path-vs-slow-path)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from PyBondLab import BatchStrategyFormation

# Run portfolio sorts on multiple signals
batch = BatchStrategyFormation(
    data=data,
    signals=['momentum', 'value', 'size'],
    holding_period=1,
    num_portfolios=5,
    turnover=False,
)
results = batch.fit()

# Access results for each signal
ew_ls, vw_ls = results['momentum'].get_long_short()
print(f"Momentum Sharpe: {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
```

---

## Full API Reference

```python
BatchStrategyFormation(
    data: pd.DataFrame,
    signals: List[str],
    holding_period: int = 1,
    num_portfolios: int = 5,
    turnover: bool = True,
    chars: List[str] = None,
    rating: Union[str, Tuple[int, int]] = None,
    subset_filter: Dict[str, Tuple[float, float]] = None,
    banding: int = None,
    columns: Dict[str, str] = None,
    n_jobs: int = 1,
    signals_per_worker: int = 1,
    chunk_size: int = None,
    verbose: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | **required** | Bond panel data with date, ID, returns, weights, and signal columns |
| `signals` | List[str] | **required** | Column names to use as sorting signals |
| `holding_period` | int | 1 | Number of months to hold portfolios (1 = monthly rebalancing) |
| `num_portfolios` | int | 5 | Number of quantile portfolios (5 = quintiles, 10 = deciles) |
| `turnover` | bool | True | Whether to compute portfolio turnover statistics |
| `chars` | List[str] | None | Characteristic columns to aggregate at portfolio level |
| `rating` | str/tuple | None | Filter by credit rating: `'IG'`, `'NIG'`, `(min, max)` tuple, or `None` for all |
| `subset_filter` | Dict | None | Filter by characteristics: `{'col': (min, max)}` (e.g., `{'MATURITY': (1, 5)}`) |
| `banding` | int | None | Banding parameter to reduce turnover (1 or 2 typical) |
| `rebalance_frequency` | str/int | `'monthly'` | Rebalancing frequency: `'monthly'`, `'quarterly'`, `'semi-annual'`, `'annual'`, or int (months) |
| `rebalance_month` | int | 6 | Month for non-monthly rebalancing (1-12, e.g., 6 for June) |
| `columns` | Dict | None | Column name mapping (see [Custom Column Names](#custom-column-names)) |
| `n_jobs` | int | 1 | Number of parallel workers (-1 = all cores) |
| `signals_per_worker` | int | 1 | Signals per worker batch (2-4 recommended for large data) |
| `chunk_size` | int | None | Process signals in chunks to limit memory |
| `verbose` | bool | True | Print progress messages |

### Returns

`BatchResults` object with dictionary-like access to individual signal results.

---

## Examples

### Basic Usage

Process multiple signals with default settings:

```python
import pandas as pd
from PyBondLab import BatchStrategyFormation

# Your data should have these columns (at minimum):
# - date: datetime
# - ID: bond identifier
# - ret: monthly return
# - VW: value weight (market cap)
# - RATING_NUM: numeric rating (1-22)
# - signal columns: your sorting variables

# Run batch processing
batch = BatchStrategyFormation(
    data=data,
    signals=['momentum_3m', 'momentum_6m', 'momentum_12m', 'reversal', 'value'],
    holding_period=1,
    num_portfolios=5,
    turnover=True,
    verbose=True,
)
results = batch.fit()

# View summary across all signals
print(results.summary_df)
```

**Output:**
```
                ew_mean   vw_mean    ew_std    vw_std  ew_sharpe  vw_sharpe  n_periods  ew_turnover  vw_turnover
signal
momentum_3m      0.0234    0.0198    0.0456    0.0412       1.78       1.67        120         0.42         0.38
momentum_6m      0.0212    0.0187    0.0423    0.0398       1.74       1.63        120         0.38         0.35
momentum_12m     0.0189    0.0165    0.0398    0.0376       1.65       1.52        120         0.32         0.29
reversal        -0.0156   -0.0143    0.0387    0.0354      -1.40      -1.40        120         0.45         0.41
value            0.0087    0.0076    0.0312    0.0298       0.97       0.88        120         0.28         0.25
```

---

### Custom Column Names

If your data uses different column names than PyBondLab expects, use the `columns` parameter:

```python
# Your data has these columns:
# - date, cusip_id, ret_vw, mcap_e, spc_rat, mom3, mom6, val

batch = BatchStrategyFormation(
    data=data,
    signals=['mom3', 'mom6', 'val'],
    columns={
        'ID': 'cusip_id',        # Bond identifier
        'ret': 'ret_vw',         # Return column
        'VW': 'mcap_e',          # Value weight
        'RATING_NUM': 'spc_rat', # Rating column
    },
    holding_period=1,
    num_portfolios=5,
    turnover=False,
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

Only specify columns that differ from defaults.

---

### Fast Path (Returns Only)

When you only need long-short returns (no turnover, characteristics, or banding), the **fast batch path** is automatically used. This is significantly faster:

```python
# Fast path is used automatically when:
# - turnover=False
# - chars=None
# - banding=None
# - rating and subset_filter are now SUPPORTED in fast path!

batch = BatchStrategyFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3', 'sig4', 'sig5',
             'sig6', 'sig7', 'sig8', 'sig9', 'sig10'],
    holding_period=1,
    num_portfolios=5,
    turnover=False,   # Required for fast path
    chars=None,       # Required for fast path
    banding=None,     # Required for fast path
    rating='IG',      # Now works with fast path!
    subset_filter={'MATURITY': (1, 5)},  # Now works with fast path!
    verbose=True,
)
results = batch.fit()
# Prints: "FAST BATCH PATH: Processing 10 signals with numba with filters..."
```

**Performance comparison (25K rows, 10 signals):**
| Path | Time | Notes |
|------|------|-------|
| Slow path | 10.4s | Uses multiprocessing |
| Fast batch | 3.8s | Uses numba kernels |
| **Speedup** | **2.7x** | |

**Note:** Rating and subset_filter are applied at formation date only to avoid look-ahead bias. Bonds are excluded from ranking but their returns are still collected if they were ranked at a previous formation date.

---

### With Turnover and Characteristics

Track portfolio turnover and aggregate characteristics:

```python
batch = BatchStrategyFormation(
    data=data,
    signals=['momentum', 'value', 'quality'],
    holding_period=3,           # Quarterly rebalancing
    num_portfolios=5,
    turnover=True,              # Compute turnover
    chars=['duration', 'spread', 'rating'],  # Aggregate these at portfolio level
    banding=1,                  # Reduce turnover with banding
    verbose=True,
)
results = batch.fit()

# Access turnover for a specific signal
ew_turn, vw_turn = results['momentum'].get_turnover()
print(f"Momentum EW turnover: {ew_turn.mean().mean():.1%}")

# Access characteristics
ew_chars, vw_chars = results['momentum'].get_characteristics()
print(ew_chars['duration'])  # Portfolio-level duration by date
```

---

### Non-Staggered Rebalancing (Quarterly/Annual)

Use non-monthly rebalancing for lower turnover strategies:

```python
# Annual rebalancing in June
batch = BatchStrategyFormation(
    data=data,
    signals=['value', 'quality', 'momentum'],
    holding_period=12,
    num_portfolios=5,
    rebalance_frequency='annual',  # or 12
    rebalance_month=6,             # Rebalance in June
    turnover=True,
    chars=['duration', 'spread'],
    n_jobs=4,
)
results = batch.fit()

# Quarterly rebalancing
batch = BatchStrategyFormation(
    data=data,
    signals=['signal1', 'signal2'],
    holding_period=3,
    num_portfolios=5,
    rebalance_frequency='quarterly',  # or 3
    rebalance_month=6,                # June, September, December, March
    turnover=False,                   # Enables ultra-fast path
)
results = batch.fit()
```

**Performance (Phase 15 optimization):**

| Configuration | Fast Path | Notes |
|---------------|-----------|-------|
| Non-staggered, `turnover=False` | **~340x speedup** | Batch-level numba kernels |
| Non-staggered, `turnover=True` | **21-103x per signal** | Each worker uses fast path |

When `turnover=False`, `chars=None`, and `banding=None` with non-staggered rebalancing, an ultra-fast batch path processes all signals using numba kernels.

When turnover, chars, or banding are enabled, individual workers use the Phase 15b fast path, still achieving 21-103x speedup per signal.

---

### Parallel Processing

For large datasets, use parallel workers:

```python
# Use all CPU cores
batch = BatchStrategyFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3', ..., 'sig50'],
    holding_period=1,
    num_portfolios=5,
    turnover=True,
    n_jobs=-1,              # Use all CPU cores
    signals_per_worker=2,   # Process 2 signals per worker (reduces overhead)
    verbose=True,
)
results = batch.fit()
```

**Recommended `n_jobs` settings:**
| Scenario | `n_jobs` | Notes |
|----------|----------|-------|
| Small data (<100K rows) | 1-2 | Overhead may exceed benefit |
| Medium data (100K-1M rows) | 4 | Good balance |
| Large data (>1M rows) | -1 (all cores) | Maximum parallelism |
| Memory constrained | 2 | Limit concurrent workers |

---

### Memory Management for Large Datasets

For datasets with many signals, use chunking to limit memory:

```python
# Process 100 signals on a machine with limited RAM
batch = BatchStrategyFormation(
    data=data,  # 2M rows
    signals=[f'signal_{i}' for i in range(100)],
    holding_period=1,
    num_portfolios=5,
    turnover=True,
    n_jobs=4,
    signals_per_worker=2,   # Group signals to reduce overhead
    chunk_size=25,          # Process 25 signals at a time
    verbose=True,
)
results = batch.fit()
```

**Memory recommendations:**
| RAM | Rows | Signals | `n_jobs` | `chunk_size` |
|-----|------|---------|----------|--------------|
| 16GB | 500K | 50 | 4 | None |
| 16GB | 2M | 100 | 2 | 20 |
| 32GB | 2M | 200 | 4 | 50 |
| 64GB | 5M | 400 | 8 | 100 |

---

## Accessing Results

### Dictionary-like Access

```python
# Get results for a specific signal
momentum_result = results['momentum']

# Check available signals
print(list(results.keys()))
# ['momentum', 'value', 'size']

# Iterate over all results
for signal, result in results.items():
    ew_ls, vw_ls = result.get_long_short()
    print(f"{signal}: EW Sharpe = {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
```

### Long-Short Returns

```python
# Get long-short portfolio returns
ew_ls, vw_ls = results['momentum'].get_long_short()

# ew_ls is a pandas Series with DatetimeIndex
print(ew_ls.head())
# date
# 2010-01-31    0.0123
# 2010-02-28    0.0087
# 2010-03-31   -0.0045
# ...

# Compute statistics
print(f"Mean:   {ew_ls.mean() * 12:.2%}")      # Annualized mean
print(f"Std:    {ew_ls.std() * 12**0.5:.2%}")  # Annualized std
print(f"Sharpe: {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
```

### Summary DataFrame

```python
# Get summary statistics for all signals
summary = results.summary_df

# Columns include:
# - ew_mean, vw_mean: Annualized mean returns
# - ew_std, vw_std: Annualized volatility
# - ew_sharpe, vw_sharpe: Annualized Sharpe ratios
# - n_periods: Number of observations
# - ew_turnover, vw_turnover: Average turnover (if computed)

print(summary.sort_values('ew_sharpe', ascending=False))
```

### Factor Returns DataFrame

```python
# Get all factor returns as a DataFrame
factor_returns = results.get_factor_returns(weight_type='ew')

# factor_returns has signals as columns, dates as index
print(factor_returns.head())
#              momentum    value     size
# date
# 2010-01-31    0.0123   0.0045   0.0067
# 2010-02-28    0.0087   0.0023   0.0034
# ...

# Compute correlation matrix
print(factor_returns.corr())
```

### Turnover and Characteristics

```python
# Turnover (only available if turnover=True)
ew_turnover, vw_turnover = results['momentum'].get_turnover()

# Characteristics (only available if chars specified)
ew_chars, vw_chars = results['momentum'].get_characteristics()
```

---

### Unified Panel Extraction with `extract_panel`

For comprehensive analysis, use `extract_panel()` to extract all results into a single panel DataFrame:

```python
from PyBondLab import BatchStrategyFormation, extract_panel, NamingConfig

# Run batch strategy (turnover=True required for full panel data)
batch = BatchStrategyFormation(
    data=data,
    signals=['momentum', 'value', 'quality'],
    holding_period=1,
    num_portfolios=5,
    turnover=True,                  # Required for extract_panel
    chars=['duration', 'spread'],   # Optional
    verbose=True,
)
results = batch.fit()

# Extract unified panel
panel = extract_panel(results)

# Panel structure:
# - date: observation date
# - factor: signal name (e.g., 'momentum')
# - freq: holding period (1, 3, 6, etc.)
# - leg: 'ls' (long-short), 'l' (long), 's' (short)
# - weighting: 'ew' or 'vw'
# - return: portfolio return
# - turnover: turnover (if computed)
# - {char_name}: characteristic values (if computed)

print(panel.head())
#         date    factor  freq leg weighting    return  turnover  duration
# 2020-01-31  momentum     1  ls        ew  0.012300  0.450000  2.100000
# 2020-01-31  momentum     1  ls        vw  0.014500  0.420000  2.300000
# 2020-01-31  momentum     1   l        ew  0.023400  0.480000  5.200000
# ...
```

**With sign correction:**

```python
# Sign-correct negative factors (flip returns, swap legs, add * suffix)
panel = extract_panel(results, naming=NamingConfig(sign_correct=True))

# Factors with negative mean are flipped:
# - L-S returns multiplied by -1
# - Long and short legs swapped
# - Factor name gets '*' suffix (e.g., 'reversal*')
```

**Pivot to wide format:**

```python
# Get L-S returns in wide format (dates × factors)
ls_returns = panel[panel['leg'] == 'ls'].pivot_table(
    index='date',
    columns=['factor', 'weighting'],
    values='return'
)
print(ls_returns.head())
#            momentum           value            quality
#                  ew       vw      ew       vw       ew       vw
# date
# 2020-01-31  0.0123   0.0145  0.0045   0.0052   0.0087   0.0091
# ...

# Compute factor correlations
print(ls_returns['momentum']['ew'].corr(ls_returns['value']['ew']))
```

**Filter by leg or weighting:**

```python
# Get only long-short EW returns
ew_ls = panel[(panel['leg'] == 'ls') & (panel['weighting'] == 'ew')]

# Get only long leg data
long_leg = panel[panel['leg'] == 'l']

# Group by factor
factor_means = panel[panel['leg'] == 'ls'].groupby(['factor', 'weighting'])['return'].mean()
```

**Note:** `extract_panel` requires `turnover=True` because the fast batch path (used when `turnover=False`) only computes long-short returns, not individual leg returns.

---

## Performance Optimization

### Fast Path vs Slow Path

| Feature | Fast Path | Slow Path |
|---------|-----------|-----------|
| **When used** | `turnover=False`, `chars=None`, `banding=None` | `turnover=True` OR `chars` specified OR `banding` specified |
| **Rating/Subset Filter** | ✅ Supported | ✅ Supported |
| **Method** | Numba kernels (all signals at once) | Multiprocessing (one signal per worker) |
| **Speed** | 2.7-3x faster | Baseline |
| **Memory** | Lower (single process) | Higher (multiple processes) |
| **Parallelism** | Within numba (prange) | Python multiprocessing |

### Optimization Tips

1. **Use fast path when possible**: If you only need returns, set `turnover=False`

2. **Batch signals per worker**: For slow path with many signals:
   ```python
   signals_per_worker=2  # or 3-4 for very large data
   ```

3. **Use chunking for memory control**:
   ```python
   chunk_size=25  # Process 25 signals at a time
   ```

4. **Choose appropriate `n_jobs`**:
   - Small data: `n_jobs=1` (overhead exceeds benefit)
   - Large data: `n_jobs=-1` (use all cores)

5. **Profile your workload**: First run with `verbose=True` to see timing breakdown

---

## When to Use Fast Path vs Slow Path

### Use Fast Path When:

- You only need long-short portfolio returns
- No turnover analysis needed
- No characteristic aggregation needed
- No banding required
- Processing many signals (10+)
- Rating and subset_filter are fine (they work with fast path!)

```python
# Fast path example - factor screening with filters
batch = BatchStrategyFormation(
    data=data,
    signals=signal_columns,  # 50+ signals to screen
    holding_period=1,
    num_portfolios=5,
    turnover=False,
    rating='IG',                          # Works with fast path
    subset_filter={'MATURITY': (1, 5)},   # Works with fast path
    verbose=True,
)
```

### Use Slow Path When:

- Need turnover analysis
- Need to track portfolio characteristics
- Using banding to reduce turnover
- Need full portfolio breakdown (not just long-short)

```python
# Slow path example - detailed analysis
batch = BatchStrategyFormation(
    data=data,
    signals=['momentum', 'value'],
    holding_period=3,
    num_portfolios=5,
    turnover=True,
    chars=['duration', 'spread'],
    banding=1,
    rating='IG',
    n_jobs=4,
)
```

---

## Troubleshooting

### Common Issues

**1. "Signal columns not found in data"**
```python
# Check that signal columns exist
missing = [s for s in signals if s not in data.columns]
print(f"Missing: {missing}")
```

**2. "Data missing required columns"**
```python
# Use columns parameter to map your column names
batch = BatchStrategyFormation(
    ...,
    columns={'ID': 'cusip', 'VW': 'mktcap'},
)
```

**3. Memory errors with large data**
```python
# Reduce memory usage
batch = BatchStrategyFormation(
    ...,
    n_jobs=2,           # Fewer workers
    chunk_size=10,      # Smaller chunks
)
```

**4. Slow performance**
```python
# Check if fast path can be used
if turnover == False and chars is None and banding is None and rating is None:
    print("Fast path will be used!")
else:
    print("Slow path will be used - consider disabling turnover/chars for speed")
```

### Getting Help

- Check verbose output for timing breakdown
- Review the [CLAUDE.md](../CLAUDE.md) for technical details
- Run validation script: `python examples/validate_fast_batch.py`

---

## Complete Example

```python
import pandas as pd
import numpy as np
from PyBondLab import BatchStrategyFormation

# Load your data
data = pd.read_parquet('bond_data.parquet')

# Define signals to test
signals = [
    'momentum_3m', 'momentum_6m', 'momentum_12m',
    'reversal_1m', 'reversal_3m',
    'value', 'quality', 'low_vol',
]

# Fast screening (returns only)
print("=== Fast Path: Factor Screening ===")
batch_fast = BatchStrategyFormation(
    data=data,
    signals=signals,
    holding_period=1,
    num_portfolios=5,
    turnover=False,
    verbose=True,
)
results_fast = batch_fast.fit()

# View summary
print("\nFactor Screening Results:")
print(results_fast.summary_df.sort_values('ew_sharpe', ascending=False))

# Detailed analysis of top factors
print("\n=== Slow Path: Detailed Analysis ===")
top_signals = ['momentum_6m', 'value']  # Based on screening

batch_detail = BatchStrategyFormation(
    data=data,
    signals=top_signals,
    holding_period=3,
    num_portfolios=5,
    turnover=True,
    chars=['duration', 'spread', 'rating'],
    banding=1,
    n_jobs=4,
    verbose=True,
)
results_detail = batch_detail.fit()

# Analyze momentum_6m in detail
mom_result = results_detail['momentum_6m']
ew_ls, vw_ls = mom_result.get_long_short()
ew_turn, vw_turn = mom_result.get_turnover()
ew_chars, vw_chars = mom_result.get_characteristics()

print(f"\nMomentum 6M Analysis:")
print(f"  Annualized Return: {ew_ls.mean() * 12:.2%}")
print(f"  Annualized Vol:    {ew_ls.std() * 12**0.5:.2%}")
print(f"  Sharpe Ratio:      {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
print(f"  Avg Turnover:      {ew_turn.mean().mean():.1%}")
print(f"  Avg Duration:      {ew_chars['duration'].mean().mean():.2f}")
```

---

## Summary

`BatchStrategyFormation` provides:

1. **Efficient batch processing** of multiple signals
2. **Automatic fast path** when only returns are needed
3. **Flexible column mapping** for custom data formats
4. **Parallel processing** for large workloads
5. **Memory management** via chunking
6. **Full result access** matching `StrategyFormation` API

For most use cases, start with the fast path for screening, then switch to slow path for detailed analysis of promising factors.
