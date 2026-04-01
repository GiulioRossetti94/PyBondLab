# BatchAssayAnomaly User Guide

`BatchAssayAnomaly` is the fast public API for running anomaly assays across **multiple signals**. It applies the same speed-first specification-grid workflow as `assay_anomaly_fast`, but across many candidate signals.

Use it when:
- you have many signals
- you want the fast numba-based anomaly workflow
- you want comparable summary output across signals

Do not use it when you mainly need one signal or a slower, fuller inspection workflow:
- for one signal, prefer `assay_anomaly_fast`
- for the richer slow-path workflow, use `AssayAnomaly`
- treat `AssayAnomalyRunner` as advanced/internal

## Which Anomaly Tool Should I Use?

| API | Recommended use |
|---|---|
| `assay_anomaly_fast` | One signal, speed-first |
| `BatchAssayAnomaly` | Many signals, speed-first |
| `AssayAnomaly` | Slower but richer inspection workflow |
| `AssayAnomalyRunner` | Advanced/internal control |

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Full API Reference](#full-api-reference)
3. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Custom Column Names](#custom-column-names)
   - [Parallel Processing](#parallel-processing)
   - [Memory Management](#memory-management)
   - [Lambda Functions in Specs](#lambda-functions-in-specs)
4. [Accessing Results](#accessing-results)
   - [Individual Signal Results](#individual-signal-results)
   - [Summary DataFrame](#summary-dataframe)
   - [Aggregated Returns](#aggregated-returns)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Complete Example](#complete-example)

---

## Quick Start

The recommended use case is many signals on one shared specification grid:

```python
from PyBondLab import BatchAssayAnomaly

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

# Run batch anomaly assay on multiple signals
batch = BatchAssayAnomaly(
    data=data,
    signals=['cs', 'ytm', 'sze', 'ami'],
    specs=specs,
    n_jobs=-1,  # Use all CPU cores
    IDvar='cusip',
    DATEvar='date',
    RETvar='ret_vw',
    VWvar='mcap_s',
    RATINGvar='RATING_NUM',
)
results = batch.fit()

# View summary across all signals
print(results.summary_df)

# Access individual signal results
print(results['cs'].summary())
```

---

## Full API Reference

```python
BatchAssayAnomaly(
    data: pd.DataFrame,
    signals: List[str],
    specs: Dict[str, Any],
    holding_period: int = 1,
    dynamic_weights: bool = True,
    skip_invalid: bool = True,
    IDvar: str = None,
    DATEvar: str = None,
    RETvar: str = None,
    VWvar: str = None,
    RATINGvar: str = None,
    n_jobs: int = 1,
    signals_per_worker: int = 1,
    chunk_size: Optional[Union[int, str]] = None,
    verbose: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | **required** | Bond panel data with date, ID, returns, weights, and signal columns |
| `signals` | List[str] | **required** | Column names to use as sorting signals |
| `specs` | Dict | **required** | Specification grid dictionary (see below) |
| `holding_period` | int | 1 | Number of months to hold portfolios |
| `dynamic_weights` | bool | True | Use VW from d-1 (True) or formation date (False) |
| `skip_invalid` | bool | True | Skip invalid spec combinations (e.g., ig_only bp + NIG filter) |
| `IDvar` | str | 'ID' | Column name for bond identifier |
| `DATEvar` | str | 'date' | Column name for date |
| `RETvar` | str | 'ret' | Column name for returns |
| `VWvar` | str | 'VW' | Column name for value weights |
| `RATINGvar` | str | 'RATING_NUM' | Column name for numeric rating |
| `n_jobs` | int | 1 | Number of parallel workers (-1 = all cores, 1 = sequential) |
| `signals_per_worker` | int | 1 | Signals per worker batch (2-4 recommended for large data) |
| `chunk_size` | int/'auto' | None | Process signals in chunks to limit memory |
| `verbose` | bool | True | Print progress messages |

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

`BatchAssayResults` object with:
- `results`: OrderedDict mapping signal names to `AnomalyAssayResult` objects
- `summary_df`: DataFrame with key statistics for all signals
- Dictionary-like access: `results['signal_name']` → `AnomalyAssayResult`
- `timings`: Dict with processing times per signal
- `errors`: Dict with error messages for failed signals

Each individual signal result is still an `AnomalyAssayResult`. If you need the slower `AnomalyResults` panel produced by `AssayAnomaly`, use that API directly instead of `BatchAssayAnomaly`.

---

## Examples

### Basic Usage

Process multiple signals with default settings:

```python
import pandas as pd
from PyBondLab import BatchAssayAnomaly

# Define specification grid
specs = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        (3, 'terciles', None),
        (5, 'quintiles', None),
        (10, 'deciles', None),
    ],
    'rating_filters': {
        'all': None,
        'ig': (1, 10),
        'nig': (11, 22),
    },
    'bp_universes': {
        'all': None,
    },
    'maturity_filters': {
        'all': None,
        'short': (0, 5),
        'long': (10, 100),
    },
}

# Run batch processing
batch = BatchAssayAnomaly(
    data=data,
    signals=['cs', 'ytm', 'sze', 'ami', 'str', 'mom12_1'],
    specs=specs,
    holding_period=1,
    verbose=True,
)
results = batch.fit()

# View summary across all signals
print(results.summary_df)
```

**Output:**
```
          n_specs  n_sig_5pct  n_sig_1pct  pct_sig_5pct  pct_sig_1pct  mean_abs_t  max_abs_t          best_spec
signal
cs            108          72          45          66.7          41.7        2.34       4.56  VW_5p_quintiles_all_all_all
ytm           108          65          38          60.2          35.2        2.12       4.21  EW_5p_quintiles_all_ig_all
sze           108          45          22          41.7          20.4        1.67       3.45  VW_10p_deciles_all_all_short
...
```

---

### Custom Column Names

If your data uses different column names than PyBondLab expects:

```python
# Your data has these columns:
# - date, cusip_id, ret_vw, mcap_e, spc_rat, cs, ytm

batch = BatchAssayAnomaly(
    data=data,
    signals=['cs', 'ytm'],
    specs=specs,
    IDvar='cusip_id',       # Bond identifier
    DATEvar='date',         # Date column
    RETvar='ret_vw',        # Return column
    VWvar='mcap_e',         # Value weight
    RATINGvar='spc_rat',    # Rating column
)
results = batch.fit()
```

**Default column mapping:**
```python
{
    'DATEvar': 'date',
    'IDvar': 'ID',
    'RETvar': 'ret',
    'VWvar': 'VW',
    'RATINGvar': 'RATING_NUM',
}
```

Only specify columns that differ from defaults.

---

### Parallel Processing

For large datasets or many signals, use parallel workers:

```python
# Use all CPU cores
batch = BatchAssayAnomaly(
    data=data,
    signals=['sig1', 'sig2', 'sig3', ..., 'sig20'],
    specs=specs,
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

### Memory Management

For large datasets, use memory-aware auto-tuning:

```python
# Auto-tune parallel configuration based on available memory
batch = BatchAssayAnomaly(
    data=data,
    signals=[f'signal_{i}' for i in range(50)],
    specs=specs,
    n_jobs=-1,
    chunk_size='auto',  # Enable memory-aware auto-tuning
    verbose=True,
)
results = batch.fit()
```

Or manually specify chunk size:

```python
# Process 10 signals at a time to limit memory
batch = BatchAssayAnomaly(
    data=data,
    signals=signals_list,
    specs=specs,
    n_jobs=4,
    chunk_size=10,          # Process 10 signals at a time
    signals_per_worker=2,   # Group signals to reduce overhead
    verbose=True,
)
```

**Memory recommendations:**
| RAM | Rows | Signals | `n_jobs` | `chunk_size` |
|-----|------|---------|----------|--------------|
| 16GB | 500K | 20 | 4 | None |
| 16GB | 2M | 50 | 2 | 10 |
| 32GB | 2M | 100 | 4 | 25 |
| 64GB | 5M | 200 | 8 | 50 |

---

### Lambda Functions in Specs

**Important:** Lambda functions in `bp_universes` cannot be pickled, which prevents parallel processing. When detected, `BatchAssayAnomaly` automatically falls back to sequential processing:

```python
# This will trigger automatic fallback to sequential processing
specs_with_lambda = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [(5, 'quintiles', None)],
    'rating_filters': {'all': None, 'ig': (1, 10)},
    'bp_universes': {
        'all': None,
        'ig_only': lambda df: df['RATING_NUM'] <= 10,  # Lambda function!
    },
    'maturity_filters': {'all': None},
}

batch = BatchAssayAnomaly(
    data=data,
    signals=['cs', 'ytm'],
    specs=specs_with_lambda,
    n_jobs=-1,  # Will fallback to sequential
    verbose=True,
)
# Output: "Note: Falling back to sequential processing because bp_universes['ig_only'] contains a lambda function"
```

**To enable parallel processing**, define functions at module level:

```python
# Define at module level (not inside a function)
def ig_only_filter(df):
    return df['RATING_NUM'] <= 10

def nig_only_filter(df):
    return df['RATING_NUM'] > 10

# Now parallel processing works!
specs_picklable = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [(5, 'quintiles', None)],
    'rating_filters': {'all': None, 'ig': (1, 10)},
    'bp_universes': {
        'all': None,
        'ig_only': ig_only_filter,  # Module-level function - can be pickled!
        'nig_only': nig_only_filter,
    },
    'maturity_filters': {'all': None},
}

batch = BatchAssayAnomaly(
    data=data,
    signals=['cs', 'ytm'],
    specs=specs_picklable,
    n_jobs=-1,  # Parallel processing will work
)
```

---

## Accessing Results

### Individual Signal Results

```python
# Get results for a specific signal
cs_result = results['cs']

# Access returns DataFrame
returns_df = cs_result.returns_df
print(f"Shape: {returns_df.shape}")  # (n_dates, n_specs)

# Get summary statistics
summary = cs_result.summary(lag=0)  # lag=0 for OLS, lag>0 for HAC
print(summary.head())

# Check available signals
print(list(results.keys()))
# ['cs', 'ytm', 'sze', 'ami', 'str', 'mom12_1']

# Iterate over all results
for signal, result in results.items():
    summary = result.summary()
    n_sig = (summary['p_value'] < 0.05).sum()
    print(f"{signal}: {n_sig}/{len(summary)} significant at 5%")
```

### Summary DataFrame

```python
# Get summary statistics for all signals
summary_df = results.summary_df

# Columns:
# - n_specs: Number of specifications tested
# - n_sig_5pct: Number significant at 5% level
# - n_sig_1pct: Number significant at 1% level
# - pct_sig_5pct: Percentage significant at 5%
# - pct_sig_1pct: Percentage significant at 1%
# - mean_abs_t: Mean absolute t-statistic
# - max_abs_t: Maximum absolute t-statistic
# - best_spec: Specification with highest |t-stat|

print(summary_df.sort_values('mean_abs_t', ascending=False))
```

### Aggregated Returns

```python
# Get best-spec returns for all signals as a DataFrame
best_returns = results.get_all_returns()
# Returns DataFrame with signals as columns (best spec for each)
print(best_returns.head())
#             cs      ytm      sze      ami
# date
# 2015-01-31  0.012  0.008   0.005  -0.003
# 2015-02-28  0.015  0.011   0.007   0.002
# ...

# Get returns for a specific specification across all signals
specific_returns = results.get_all_returns(spec_id='EW_5p_quintiles_all_all_all')

# Compute correlations between factors
print(best_returns.corr())
```

### Timing and Errors

```python
# Check processing times
print(results.timings)
# {'cs': 5.2, 'ytm': 4.8, 'sze': 5.1, ..., 'total': 28.5}

# Check for errors
print(results.errors)
# {} (empty if no errors)

# Lists of successful and failed signals
print(results.successful_signals)
# ['cs', 'ytm', 'sze', 'ami', 'str', 'mom12_1']
print(results.failed_signals)
# []
```

---

## Performance Optimization

### Sequential vs Parallel

| Mode | When Used | Method |
|------|-----------|--------|
| Sequential | `n_jobs=1` or lambda functions in specs | Single process, no overhead |
| Parallel | `n_jobs > 1` and no lambda functions | ProcessPoolExecutor |

### Optimization Tips

1. **Use parallel processing for many signals**: When processing 4+ signals, parallel processing provides significant speedup.

2. **Batch signals per worker**: For large datasets, reduce overhead by processing multiple signals per worker:
   ```python
   signals_per_worker=2  # or 3-4 for very large data
   ```

3. **Use chunking for memory control**:
   ```python
   chunk_size=10  # Process 10 signals at a time
   ```

4. **Auto-tune with `chunk_size='auto'`**: Let the system determine optimal configuration based on available memory.

5. **First signal warmup**: The first signal is always processed sequentially for numba warmup, then remaining signals run in parallel.

### Performance Benchmarks

| Signals | Specs | n_jobs | Time | Notes |
|---------|-------|--------|------|-------|
| 6 | 280 | 1 | ~63s | Sequential baseline |
| 6 | 280 | 4 | ~35s | 1.8x speedup |
| 6 | 280 | -1 | ~30s | 2.1x speedup (8 cores) |

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
# Check required columns and use column mapping
required = ['date', 'cusip', 'ret_vw', 'mcap_s', 'RATING_NUM']
missing = [c for c in required if c not in data.columns]
print(f"Missing columns: {missing}")

# Map your column names
batch = BatchAssayAnomaly(
    ...,
    IDvar='cusip',
    RETvar='ret_vw',
    VWvar='mcap_s',
)
```

**3. Parallel processing falls back to sequential**
```python
# Check for lambda functions in specs
if 'bp_universes' in specs:
    for name, func in specs['bp_universes'].items():
        if callable(func) and hasattr(func, '__name__') and func.__name__ == '<lambda>':
            print(f"Lambda detected: bp_universes['{name}']")
            print("Replace with module-level function to enable parallel processing")
```

**4. Memory errors with large data**
```python
# Reduce memory usage
batch = BatchAssayAnomaly(
    ...,
    n_jobs=2,           # Fewer workers
    chunk_size=5,       # Smaller chunks
)
```

**5. Check progress and timing**
```python
# Enable verbose output
batch = BatchAssayAnomaly(..., verbose=True)
results = batch.fit()

# Check individual timings
for signal, elapsed in results.timings.items():
    if signal != 'total':
        print(f"{signal}: {elapsed:.2f}s")
```

---

## Complete Example

```python
import pandas as pd
import numpy as np
from PyBondLab import BatchAssayAnomaly

# Load data
data = pd.read_parquet('bond_data.parquet')

# Define comprehensive specification grid
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
        'nig': (11, 22),
    },
    'bp_universes': {
        'all': None,
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

# Run batch anomaly assay
print("Running batch anomaly assay...")
batch = BatchAssayAnomaly(
    data=data,
    signals=signals,
    specs=TIER2_SPECS,
    holding_period=1,
    dynamic_weights=True,
    skip_invalid=True,
    n_jobs=-1,
    verbose=True,
    IDvar='cusip',
    DATEvar='date',
    RETvar='ret_vw',
    VWvar='mcap_s',
    RATINGvar='RATING_NUM',
)
results = batch.fit()

# Print summary
print("\n" + "="*60)
print("SIGNAL ROBUSTNESS SUMMARY")
print("="*60)
print(results.summary_df.to_string())

# Analyze best signals
summary = results.summary_df
best_signals = summary.nlargest(3, 'mean_abs_t')
print("\nTop 3 Most Robust Signals:")
for signal in best_signals.index:
    row = summary.loc[signal]
    print(f"\n{signal}:")
    print(f"  Mean |t|: {row['mean_abs_t']:.2f}")
    print(f"  Max |t|: {row['max_abs_t']:.2f}")
    print(f"  Significant (5%): {row['pct_sig_5pct']:.1f}%")
    print(f"  Best spec: {row['best_spec']}")

# Get returns for best specification of each signal
best_returns = results.get_all_returns()
print("\nFactor Correlation Matrix:")
print(best_returns.corr().round(2))

# Export results for further analysis
for signal in signals:
    result = results[signal]
    result.returns_df.to_csv(f'factor_returns_{signal}.csv')
    result.summary(lag=0).to_csv(f'factor_summary_{signal}.csv')

print("\nResults exported!")
```

---

## Summary

`BatchAssayAnomaly` provides:

1. **Efficient batch processing** of multiple signals across specification grids
2. **Automatic parallelization** with memory-aware configuration
3. **Flexible column mapping** for custom data formats
4. **Automatic fallback** to sequential processing when specs contain lambda functions
5. **Comprehensive results** with individual signal access and aggregate statistics
6. **Memory management** via chunking for large workloads

For single-signal analysis, see `assay_anomaly_fast`. For factor construction without specification grids, see `BatchStrategyFormation`.
