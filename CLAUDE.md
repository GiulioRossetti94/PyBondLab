# PyBondLab Numba Optimization Project

## Project Goal
Dramatically speed up portfolio formation in PyBondLab using numba/prange, while maintaining exact numerical compatibility with the existing (slow but correct) implementation.

## Current Branch
`claude/numba-portfolio-optimization-WZSI1`

---

## ✅ Current Status (Dec 2025)

**Numba optimizations are ENABLED** and working with co-author's turnover fix.

**Active optimizations:**
- `numba_core.py` - All numba kernels active in `PyBondLab.py`
- `_accumulate_turnover_fast()` - Fast turnover path in `utils_turnover.py`
- `set_zero_for_holding_cohorts()` - Co-author's fix for holding cohort turnover (sets to 0, not NaN)

**Test results:** 12/12 tests pass in ~12.8s (3x faster than original ~38s)

---

## Performance Results (Current State)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total test suite time | 38.48s | 12.93s | **3x faster** |
| All 12 tests pass | YES | YES | Exact match (<1e-10) |

### Per-Test Performance (JIT Warmed)

| Test | Before | After | Speedup |
|------|--------|-------|---------|
| SingleSort hp=1, no turnover | 0.94s | 0.74s | 1.3x |
| SingleSort hp=1, turnover | 1.63s | 0.61s | **2.7x** |
| SingleSort hp=3, no turnover | 1.19s | 0.79s | 1.5x |
| SingleSort hp=3, turnover | 3.24s | 1.21s | **2.7x** |
| DoubleSort hp=1, turnover | 2.41s | 0.69s | **3.5x** |
| DoubleSort hp=3, turnover | 7.25s | 1.40s | **5.2x** |

---

## Optimization Status

### Completed Phases

| Phase | Description | Status | Files Changed |
|-------|-------------|--------|---------------|
| **Phase 1** | Enable existing numba functions | ✅ Complete | `precompute.py`, `utils_portfolio.py` |
| **Phase 2a** | Pre-extract data to numpy arrays | ✅ Complete | `PyBondLab.py` |
| **Phase 2b** | Vectorize return/weight computation | ✅ Complete | `PyBondLab.py`, `numba_core.py` |
| **Phase 4** | Batch turnover computation | ✅ Complete | `utils_turnover.py`, `numba_core.py` |

### Pending Phases

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| **Phase 3** | Parallelize main loop with prange | ⏳ Pending | Complex due to turnover state dependencies |

---

## Filter Optimization (Phase 10)

**Note:** Fast path is currently **DISABLED** for filtered strategies due to ID
intersection discrepancy. Slow path ensures numerical correctness. See "Known Issues".

### How Filters Work (Slow Path)

When filters are applied:
1. **Ranking** uses the filtered signal (e.g., `signal_trim` for Momentum)
2. **EA (Ex-Ante) returns** use original `ret` column
3. **EP (Ex-Post) returns** use filtered `ret_{adj}` column (e.g., `ret_trim`)
4. Both EA and EP use the **same ranking** - only return column differs

### Validation Results

All 12 filter configurations validated (slow path = source of truth):
- 12/13 tests PASS (all filtered cases match)
- 1/13 test has VW discrepancy (no-filter case, see Known Issues)

### Example Usage

```python
import PyBondLab as pbl

# Initialize Momentum strategy
mom = pbl.Momentum(holding_period=3, lookback_period=3, skip=1, num_portfolios=5)

# Run with trim filter (slow path used for correctness)
result = pbl.StrategyFormation(
    data,
    strategy=mom,
    filters={'adj': 'trim', 'level': 0.2},  # Trim returns > 20%
    turnover=False,
    verbose=True
).fit()

# Get EA and EP results (they differ when filters are applied!)
ew_ea, vw_ea = result.get_long_short()           # Ex-Ante (raw returns)
ew_ep, vw_ep = result.get_long_short_ex_post()   # Ex-Post (filtered returns)

print(f"EA mean: {ew_ea.mean():.6f}")
print(f"EP mean: {ew_ep.mean():.6f}")
print(f"EA-EP diff: {ew_ea.mean() - ew_ep.mean():.6f}")
```

### Verified Behavior

| Scenario | EA vs EP |
|----------|----------|
| No filter | EA = EP (identical) |
| Trim filter | EA ≠ EP (EA uses raw ret, EP uses trimmed ret) |
| Price filter | EA ≠ EP (EA uses raw ret, EP excludes filtered prices) |
| Bounce filter | EA ≠ EP (EA uses raw ret, EP excludes bounce-backs) |

### Test Script

```bash
python examples/data_uncertainty_baseline.py
```

This tests 13 filter configurations (trim, price, bounce, no_filter) with Momentum(3,3).

### SingleSort Test Script (Standard Signal)

```bash
python examples/data_uncertainty_singlesort.py
```

This tests fast vs slow path using a standard signal column (not a derived signal like Momentum).
Tests 28 configurations covering:
- **Holding periods**: hp=1 and hp=3
- **dynamic_weights**: True and False
- **Filters**: trim, price, bounce, no_filter

Key observations:
- For hp=1, both `dynamic_weights=True` and `False` produce identical results
- For hp=3, they produce different results (True uses VW from d-1, False uses VW from formation date)
- Fast path matches slow path within 1e-10 tolerance for all configurations
- Speedup: ~4x (after JIT warmup)

Command line options:
```bash
python examples/data_uncertainty_singlesort.py --hp 1         # Test only hp=1
python examples/data_uncertainty_singlesort.py --dw true      # Test only dynamic_weights=True
python examples/data_uncertainty_singlesort.py --no-validate  # Skip slow/fast comparison
```

---

## File Changes Summary

### New Files Created

| File | Purpose | Key Functions |
|------|---------|---------------|
| `PyBondLab/numba_core.py` | Numba-optimized core kernels | See detailed section below |
| `PyBondLab/pbl_test.py` | Baseline test script (source of truth) | `run_all_baseline_tests()`, `validate_against_baseline()` |
| `PyBondLab/baseline_results/` | Stored baseline results | `baseline_results.json`, `baseline_results.pkl` |
| `examples/test_fast_strategy.py` | Fast strategy path validation | `test_momentum_fast_vs_slow()`, `test_momentum_comprehensive()` |
| `examples/debug_wins.py` | Debug script for wins thresholds | Ex-ante vs global threshold comparison |
| `examples/debug_wins_full.py` | Debug script for wins factor comparison | Full slow vs fast path factor comparison |

### Modified Files

| File | Changes Made |
|------|--------------|
| `PyBondLab/PyBondLab.py` | Import numba kernels; replace pandas groupby in `_form_single_period` with numba functions |
| `PyBondLab/precompute.py` | Import optimized functions from `utils_optimized.py` |
| `PyBondLab/utils_portfolio.py` | Import optimized functions from `utils_optimized.py` |
| `PyBondLab/utils_turnover.py` | Add fast turnover path (enabled), import numba kernels |

### Unchanged Files (by design)

- `PyBondLab/FilterClass.py` - Must not change filter behavior
- `PyBondLab/StrategyClass.py` - Strategy definitions unchanged
- `PyBondLab/constants.py` - Constants unchanged
- `PyBondLab/utils.py` - Original functions preserved for fallback
- `PyBondLab/utils_optimized.py` - Pre-existing numba functions

---

## numba_core.py Architecture

This new module contains all numba-optimized kernels for portfolio formation.

### Core Computation Functions

```python
# Portfolio returns - replaces pandas groupby().mean() and weighted sum
compute_portfolio_returns_single(ranks, returns, weights, nport)
    -> (ew_returns, vw_returns)  # Arrays of length nport

# Portfolio weights - replaces pandas groupby().sum() for normalization
compute_portfolio_weights_single(ranks, value_weights, nport)
    -> (eweights, vweights, counts)  # Per-bond arrays

# Scaled weights for turnover - replaces pandas merge/divide operations
compute_scaled_weights_single(ranks, returns, eweights, vweights, counts,
                              ew_ptf_ret, vw_ptf_ret, nport)
    -> (ew_scaled, vw_scaled)  # Per-bond arrays

# Characteristics aggregation - replaces pandas groupby().mean()
compute_characteristics_single(ranks, weights, char_values, nport)
    -> (ew_char, vw_char)  # Arrays of length nport
```

### Batch Processing Functions (for future parallelization)

```python
# Process all periods in parallel (not yet integrated)
compute_all_portfolio_returns_batch(all_ranks, all_returns, all_weights,
                                    valid_mask, nport)
    -> (ew_ret_all, vw_ret_all)  # Shape (n_periods, nport)
```

### Turnover Functions (enabled - 3-5x speedup for turnover computation)

```python
# Batch turnover for all portfolios at once
compute_turnover_all_portfolios(ranks, positions, raw_ew, raw_vw,
                                prev_scaled_ew, prev_scaled_vw,
                                prev_sum_ew, prev_sum_vw,
                                prev_seen_ew, prev_seen_vw,
                                cohort, nport, n_assets)
    -> (turn_ew, turn_vw, new_seen_ew, new_seen_vw, curr_sum_ew, curr_sum_vw)

# Update previous weights for next period
# NOTE: Only zeros portfolios present in current data (bug fix)
update_prev_scaled_weights(scaled_ew, scaled_vw, ranks, positions,
                           prev_scaled_ew, prev_scaled_vw, nport)
```

### Strategy Signal Computation (Phase 12 - 163x speedup)

```python
# Panel-based momentum signal computation (parallel over bonds)
compute_momentum_signals_panel(logret_all, bond_starts, lookback, skip)
    -> signals  # Shape: (n_obs, 1), rolling cumulative return

# Panel-based LT-reversal signal computation (parallel over bonds)
compute_ltreversal_signals_panel(logret_all, bond_starts, lookback, skip)
    -> signals  # Shape: (n_obs, 1), mean of rolling returns

# Helper to find bond boundaries in sorted panel data
get_bond_boundaries(id_idx)
    -> bond_starts  # Array of indices where bond ID changes
```

---

## How the Optimizations Work

### Before (Slow - pandas groupby per period)

```python
# In _form_single_period - called ~1000+ times
sums = It1.groupby('ptf_rank')[VW].sum()           # Slow groupby
It1['weights'] = It1[VW] / It1['ptf_rank'].map(sums)
ptf_ret_ew = It1.groupby('ptf_rank')[ret_col].mean()  # Another groupby
ptf_ret_vw = (It1[ret_col] * It1['weights']).groupby(...).sum()  # Another
```

### After (Fast - numba kernels)

```python
# In _form_single_period - same call frequency but much faster
ranks_arr = It1['ptf_rank'].values.astype(np.float64)
returns_arr = It1[ret_col].values.astype(np.float64)
vw_arr = It1[VW].values.astype(np.float64)

# Single numba call replaces multiple groupby operations
eweights_arr, vweights_arr, counts_arr = compute_portfolio_weights_single(
    ranks_arr, vw_arr, tot_nport
)
ew_ret_arr, vw_ret_arr = compute_portfolio_returns_single(
    ranks_arr, returns_arr, vw_arr, tot_nport
)
```

---

## Test Suite (pbl_test.py)

### Running Tests

```bash
# Run full baseline test suite
python -m PyBondLab.pbl_test

# Or from Python
import sys
sys.path.insert(0, '.')
from PyBondLab.pbl_test import main
results = main()
```

### Validating Against Baseline

```python
from PyBondLab.pbl_test import (
    generate_synthetic_data, load_baseline_results,
    validate_against_baseline, RANDOM_SEED, N_DATES, N_BONDS
)

# Generate same data (same seed = same data)
data = generate_synthetic_data(n_dates=N_DATES, n_bonds=N_BONDS, seed=RANDOM_SEED)

# Load baseline
baseline = load_baseline_results()

# Validate - returns True if all pass
passed = validate_against_baseline(data, baseline)
```

### Test Configurations

| # | Sort Type | HP | Banding | Turnover | Chars |
|---|-----------|-----|---------|----------|-------|
| 1 | SingleSort | 1 | None | False | None |
| 2 | SingleSort | 1 | None | True | None |
| 3 | SingleSort | 1 | 1 | True | None |
| 4 | SingleSort | 1 | 2 | True | None |
| 5 | SingleSort | 3 | None | True | None |
| 6 | SingleSort | 3 | 1 | True | None |
| 7 | DoubleSort (uncond) | 1 | None | True | None |
| 8 | DoubleSort (cond) | 1 | None | True | None |
| 9 | DoubleSort (uncond) | 3 | 1 | True | None |
| 10 | DoubleSort (cond) | 3 | 1 | True | None |
| 11 | SingleSort | 1 | None | True | [char1, char2] |
| 12 | SingleSort | 3 | 1 | True | [char1, char2, char3] |

### Test Tolerance
- All values must match within `TOLERANCE = 1e-10`
- Returns, turnover, and characteristics all validated

---

## Known Issues & TODO

### Fast Path Requirements (RESOLVED)

The ultra-fast path (`_fit_fast_returns_only`) now correctly matches the slow path
for both `dynamic_weights=True` and `dynamic_weights=False`. The following issues
were fixed:

1. **Ranking algorithm mismatch**: Fast path used position-based ranking, slow path
   uses `np.percentile` thresholds. Fixed by rewriting `compute_ranks_all_dates_fast`
   to use percentile thresholds matching slow path.

2. **VW date intersection**: Fast path included bonds without valid VW at t-1.
   Fixed by adding `if np.isnan(weight): continue` checks in ultrafast functions.

3. **dynamic_weights support**: Fast path now supports both `dynamic_weights=True`
   and `False` by using a VW lookup table and conditionally selecting VW date.

**Current status:** Fast path matches slow path within 1e-17 tolerance for both
HP=1 and HP=3, with either `dynamic_weights` setting.

**Requirements for fast path:**
- `turnover=False`
- `chars=None`
- `banding_threshold=None`
- SingleSort only
- Monthly rebalancing
- No filters applied

---

## dynamic_weights Parameter

The `dynamic_weights` parameter controls which date's value weights (VW) are used
for VW portfolio return calculations.

### Behavior by Setting

| Setting | VW Source Date | Description |
|---------|---------------|-------------|
| `True` | `t+h-1` (day before return date) | VW from most recent available date |
| `False` | `t` (formation date) | VW from when portfolio was formed |

### Effect on Staggered Rebalancing (hp>1)

For `holding_period=1`, both settings use the same VW date (formation date = return date - 1).

For `holding_period>1` (staggered), the settings differ:

**Example: hp=3, return date = March**
- Cohort 0: Formed in February → uses VW from February (both settings same)
- Cohort 1: Formed in January → `True` uses VW from February, `False` uses VW from January
- Cohort 2: Formed in December → `True` uses VW from February, `False` uses VW from December

### Code Location

- **Config default**: `config.py:277` - `dynamic_weights: bool = False`
- **Baseline tests**: Use `dynamic_weights=True` (line 484 in `pbl_test.py`)
- **BatchStrategyFormation**: Hardcodes `dynamic_weights=True` (lines 136, 196, 480, 594)

### Usage

```python
from PyBondLab import StrategyFormation, SingleSort
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig

# Explicit dynamic_weights=True (recommended)
config = StrategyFormationConfig(
    data=DataConfig(),
    formation=FormationConfig(dynamic_weights=True)
)

sf = StrategyFormation(
    data=data,
    strategy=SingleSort(holding_period=3, sort_var='signal', num_portfolios=5),
    turnover=False,
    config=config
)
result = sf.fit()
```

### Parallelization (Phase 3) - Not Started

The main loop in `_form_cohort_portfolios` could be parallelized with `prange`,
but this is complex because:
- Turnover state has sequential dependencies (each period depends on previous)
- Banding creates dependencies between cohorts (lag_rank state)
- Without turnover, parallelization would be more straightforward

**Potential approaches:**
1. Parallelize only when `turnover=False`
2. Restructure to batch portfolio formation first, then sequential turnover
3. Accept current 3x speedup as sufficient

---

## Architecture Overview

### Key Files and Their Roles

| File | Purpose | Performance Critical |
|------|---------|---------------------|
| `PyBondLab/PyBondLab.py` | Main `StrategyFormation` class | **YES - Optimized** |
| `PyBondLab/numba_core.py` | **NEW** - Numba kernels | **YES - Core speedup** |
| `PyBondLab/precompute.py` | Precomputes ranks, weights | YES - Uses optimized imports |
| `PyBondLab/utils_turnover.py` | Turnover computation | YES - Fast path ready |
| `PyBondLab/utils_portfolio.py` | Portfolio utilities | YES - Uses optimized imports |
| `PyBondLab/utils_optimized.py` | Pre-existing numba functions | Already optimized |
| `PyBondLab/pbl_test.py` | **NEW** - Test suite | Validation only |

### Data Flow (with optimizations marked)

```
StrategyFormation.fit()
    → _validate_input()
    → _precompute_formation_data()  [Uses optimized utils]
        → compute_thresholds_optimized()  ← numba
        → assign_bond_bins_optimized()    ← numba
    → _form_cohort_portfolios()
        → For each (date, holding_period):
            → _form_single_period()  [OPTIMIZED]
                → intersect_id_optimized()           ← numba
                → compute_portfolio_weights_single() ← numba (NEW)
                → compute_portfolio_returns_single() ← numba (NEW)
                → compute_scaled_weights_single()    ← numba (NEW)
                → compute_characteristics_single()   ← numba (NEW)
                → accumulate_turnover()              ← numba (fast path enabled)
    → _finalize_results()
```

---

## Key Implementation Details

### Banding Parameter
- Type: `int` (commonly 1 or 2)
- Meaning: Number of portfolio ranks a bond must move to be reassigned
- Formula: `threshold = banding / nport`
- Location: `PyBondLab.py:1557-1597`

### Characteristics (chars)
- Parameter: `chars=["var1", "var2"]`
- Computes portfolio-level means (EW and VW)
- Now uses `compute_characteristics_single()` numba kernel

### dynamic_weights
- **Always True** - deprecated False option removed
- Uses weights from t+h-1 for return calculation at t+h

### Holding Period & Staggered Rebalancing
- `holding_period=1`: Monthly rebalance, no staggering
- `holding_period>1`: Staggered cohorts (e.g., hp=6 creates 6 cohorts)

---

## Quick Reference Commands

### Run Test Suite
```bash
python -m PyBondLab.pbl_test
```

### Quick Validation
```python
import sys
sys.path.insert(0, '.')

from PyBondLab.pbl_test import generate_synthetic_data, run_all_baseline_tests
import json

data = generate_synthetic_data()
results = run_all_baseline_tests(data, verbose=False)

with open('PyBondLab/baseline_results/baseline_results.json') as f:
    baseline = json.load(f)

for name, result in results.items():
    base = baseline[name]
    diff = abs(result.ew_ls_mean - base['ew_ls_mean'])
    print(f"{name}: diff={diff:.2e}")
```

### Git Workflow
```bash
git push -u origin claude/numba-portfolio-optimization-WZSI1
```

---

## Important Notes

1. **Never change FilterClass.py behavior** - filters must work identically
2. **dynamic_weights=True always** - don't optimize for False case
3. **Banding is integer** - not float, not boolean (commonly 1 or 2)
4. **Source of truth = baseline_results.pkl** - new code must match exactly
5. **Test with turnover=True** - this is the slow path to optimize
6. **Test holding_period=1 AND holding_period=3** - different code paths
7. **JIT warmup matters** - first run includes compilation time

---

---

## BatchStrategyFormation (Multi-Signal Processing)

### Overview

`BatchStrategyFormation` processes multiple signals in parallel using Python's `multiprocessing`.

### Full API

```python
from PyBondLab import BatchStrategyFormation

batch = BatchStrategyFormation(
    data=data,
    signals=['momentum', 'value', 'size', 'reversal'],
    holding_period=1,              # Rebalancing frequency (1=monthly)
    num_portfolios=5,              # Number of portfolio bins (quintiles)
    turnover=True,                 # Compute portfolio turnover
    chars=['char1', 'char2'],      # Characteristics to aggregate (optional)
    rating=None,                   # Rating filter (optional)
    banding=1,                     # Banding threshold (optional, int)
    n_jobs=4,                      # Number of parallel workers
    signals_per_worker=2,          # Process 2 signals per worker (reduces overhead)
    chunk_size=20,                 # Process 20 signals at a time (limits memory)
    verbose=True,                  # Show progress
)
results = batch.fit()

# Access results (same API as StrategyFormation)
results['momentum'].get_long_short()
results['momentum'].get_turnover()
results['momentum'].get_characteristics()  # If chars specified
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | required | Panel data with signals |
| `signals` | List[str] | required | Column names of signals to process |
| `holding_period` | int | 1 | Rebalancing frequency |
| `num_portfolios` | int | 5 | Number of portfolio bins |
| `turnover` | bool | True | Compute turnover statistics |
| `chars` | List[str] | None | Characteristics to aggregate at portfolio level |
| `rating` | str/tuple | None | Rating filter |
| `banding` | int | None | Banding threshold (1 or 2 typical) |
| `n_jobs` | int | 1 | Number of parallel workers |
| `signals_per_worker` | int | 1 | Signals per worker (2-4 recommended for large data) |
| `chunk_size` | int | None | Process in chunks to limit memory (recommended for 50+ signals) |
| `verbose` | bool | True | Show progress output |

### Example Script

```bash
python examples/batch_strategy_formation.py
```

### What Gets Parallelized

**Current implementation** (`batch.py`):
- Uses `ProcessPoolExecutor` with `n_jobs` workers
- Each worker runs a complete `StrategyFormation.fit()` for one signal
- First signal runs sequentially to extract "shared" precompute data

**What runs in parallel:**
```
Worker 1: signal_1 → StrategyFormation.fit() → result_1
Worker 2: signal_2 → StrategyFormation.fit() → result_2
Worker 3: signal_3 → StrategyFormation.fit() → result_3
Worker 4: signal_4 → StrategyFormation.fit() → result_4
```

### Current Performance

| Data Size | Signals | n_jobs=1 | n_jobs=4 | Speedup |
|-----------|---------|----------|----------|---------|
| 25K rows (test) | 10 | 7.4s | 2.9s | **2.5x** |
| 25K rows (test) | 20 | 13.0s | 4.8s | **2.7x** |
| 2M rows (real) | 10 | ~34s | ~17s | **~2x** |
| 2M rows (real) | 20 | ~81s | ~40s | **~2x** |

### Scaling Limitations & Memory Issues

**The Core Problem:**
```python
# In batch.py line 482-486:
worker_args = [
    (signal, self.data, ...)  # ENTIRE DataFrame copied to each worker!
    for signal in remaining_signals
]
```

**Memory impact with 2M row dataset:**
| Component | Size (est.) | With 4 workers |
|-----------|-------------|----------------|
| Data DataFrame | 500 MB | 2 GB (4 copies) |
| shared_precomp | 200 MB | 800 MB (4 copies) |
| Worker overhead | 50 MB | 200 MB |
| **Total** | **750 MB** | **3 GB** |

**Why it doesn't scale linearly:**
1. **Pickle serialization**: Each worker receives data via pickle (slow for large DataFrames)
2. **Memory pressure**: 4 copies of data = 4x RAM usage
3. **GIL not released**: Python code holds GIL; only numba kernels run truly parallel
4. **Startup overhead**: Each `ProcessPoolExecutor` task has fixed overhead

---

## Batch Memory Optimizations (Implemented)

The following optimizations have been implemented to reduce memory usage and improve performance:

### ✅ Option 1: Platform-Aware Start Method (DONE)
```python
# In batch.py - _get_start_method()
def _get_start_method() -> str:
    if platform.system() == 'Windows':
        return 'spawn'  # Windows only supports spawn
    else:
        return 'fork'   # Linux/macOS: copy-on-write memory sharing
```
- **Benefit**: Zero-copy data sharing on Linux/macOS
- **Cross-platform**: Automatically detects OS and uses best method

### ✅ Option 2: Minimal Data Transfer (DONE)
```python
# In batch.py - _get_minimal_data()
REQUIRED_COLUMNS = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']

# Only sends required columns + signal + chars to workers
minimal_data = data[required_cols].copy()
```
- **Benefit**: 50-80% reduction in data size per worker
- **Impact**: 15.6 MB → 3.3 MB per worker (79% reduction in test)

### ✅ Option 3: No shared_precomp Transfer (DONE)
```python
# Workers compute their own precompute data
# Avoids pickling large dict-of-DataFrames
shared_precomp = None  # Not passed to workers
```
- **Benefit**: Significant memory reduction
- **Trade-off**: ~5% compute overhead per worker (acceptable)

### ✅ Option 4: Signal Batching (DONE)
```python
# Process multiple signals per worker to reduce overhead
batch = BatchStrategyFormation(
    signals_per_worker=2,  # Each worker processes 2 signals
    ...
)
```
- **Benefit**: ~20-30% faster due to reduced worker startup/communication overhead
- **Trade-off**: Slightly higher per-worker memory

### ✅ Option 5: Chunked Processing (DONE)
```python
# Process signals in chunks to limit peak memory
batch = BatchStrategyFormation(
    chunk_size=20,  # Process 20 signals at a time
    ...
)
```
- **Benefit**: Limits concurrent memory usage for 100+ signals
- **Memory cleanup**: `gc.collect()` runs between chunks

### Recommended Settings by Use Case

| Scenario | n_jobs | signals_per_worker | chunk_size |
|----------|--------|-------------------|------------|
| Small (10 signals, <500K rows) | 4 | 1 | None |
| Medium (50 signals, 1M rows) | 4 | 2 | 20 |
| Large (100 signals, 2M rows) | 4 | 3 | 25 |
| Memory-constrained (16GB RAM) | 2 | 2 | 10 |

### Future Options (Not Yet Implemented)

**Option 6: Shared Memory Arrays (for datasets > 5M rows)**
```python
import multiprocessing.shared_memory as shm
# Convert DataFrame to numpy arrays in shared memory
```
- Requires significant refactoring

**Option 7: Memory-Mapped Files**
```python
data.to_parquet('/tmp/data.parquet')
# Workers: pd.read_parquet('/tmp/data.parquet', memory_map=True)
```
- Better for very large datasets that don't fit in RAM

---

## Multi-Signal Vectorized Functions (Experimental)

Located in `numba_core.py`, these attempt to process all signals at once:

```python
# Compute ranks for all signals in parallel
compute_ranks_multi_signal(signal_matrix, nport)  # shape: (n_bonds, n_signals)

# Compute returns for all signals
compute_portfolio_returns_multi_signal(ranks_matrix, returns, weights, nport)
```

**Important:** Do NOT use `fastmath=True` with these functions - it causes incorrect
rank assignments due to floating-point comparison reordering with NaN values.

**Status:** These don't provide speedup over sequential processing because:
- Sorting inside prange loops has overhead
- Memory allocation inside loops is expensive
- The real bottleneck is the full StrategyFormation pipeline, not just rank computation

---

## Future Optimization Opportunities

1. **Parallelize main loop** - Use `prange` when turnover is disabled
2. **Pre-allocate all arrays** - Avoid repeated allocations in the loop
3. **Vectorize ID intersection** - Currently still uses pandas operations
4. **Profile with larger data** - Current tests use 500 bonds, 60 dates
5. **Shared memory for very large datasets** - See "Future Options" in batch section

## Completed Optimizations Summary

- **Phase 1**: Enabled existing numba functions (1.05x speedup)
- **Phase 2**: Vectorized portfolio computation with numba kernels (1.35x speedup)
- **Phase 4**: Batch turnover computation (additional 2.2x speedup)
- **Phase 5**: BatchStrategyFormation for multi-signal processing (2-2.5x speedup with 4 workers)
- **Phase 6**: Batch memory optimizations (79% data reduction per worker)
  - Platform-aware start method (fork on Linux/macOS, spawn on Windows)
  - Minimal data transfer (only required columns)
  - No shared_precomp passed to workers
- **Phase 7**: Batch speed optimizations
  - Signal batching per worker (~20-30% faster with signals_per_worker=2)
  - Chunked processing with gc.collect() between chunks (memory control)
- **Phase 8**: Fast returns-only path (1.5-2x additional speedup when applicable)
  - Auto-detects when turnover=False, chars=None, banding=None
  - Processes all dates in parallel using prange
  - Numerically identical to slow path (< 1e-10 tolerance)
- **Phase 9**: Ultra-fast path bypassing pandas (5x speedup for large panels)
  - Bypasses `_precompute_data()` entirely for massive speedup
  - Vectorized rank computation across all dates using numba
  - `generate_synthetic_data_fast()` for large panel testing
- **Phase 10**: Filter optimization (38x speedup for filtered strategies)
  - Fast path now supports trim, price, bounce filters
  - EA uses original returns, EP uses filtered returns
  - Same ranking for both, only return column differs
- **Phase 11**: DataUncertaintyAnalysis blazing fast path (**75x speedup**)
  - Parallel numba kernels process ALL (date × filter) combinations via prange
  - `compute_ranks_all_filters`: Ranks for all filters at once
  - `compute_ls_returns_all_filters_hp1/staggered`: Returns for all filters in parallel
  - Filter-specific ranking with correct exclusion behavior
  - New price filter format: `[[left_levels], [right_levels]]`
- **Phase 12**: Fast Strategy Path for Momentum/LTreversal (**163x speedup**)
  - Panel-based numba kernels: `compute_momentum_signals_panel()`, `compute_ltreversal_signals_panel()`
  - Parallelizes signal computation over bonds (10,000+ parallel tasks)
  - Ex-ante winsorization using rolling historical percentiles (matches slow path exactly)
  - All filter types supported: baseline, trim, price, bounce, wins
- **Total**: ~3x faster single-signal, ~2.5x parallel speedup for batch, **5x for large panels, 75-163x for DataUncertaintyAnalysis**

---

## Fast Returns-Only Path (Phase 8 + 9)

### Overview

When only portfolio returns are needed (no turnover, characteristics, or banding),
the code automatically uses an ultra-fast path that:
1. Bypasses pandas precomputation entirely
2. Converts DataFrame to numpy arrays once
3. Computes ranks for ALL dates in parallel using numba
4. Computes returns for ALL dates in parallel using numba

### Conditions for Fast Path

The fast path is **automatically used** when ALL of these conditions are met:

| Condition | Required Value | Notes |
|-----------|---------------|-------|
| `turnover` | `False` | No turnover computation |
| `chars` | `None` | No characteristics tracking |
| `banding_threshold` | `None` | No transition bands |
| Strategy | `SingleSort` | Not DoubleSort |
| Rebalancing | `monthly` | Standard staggered rebalancing |

### Performance (Large Panels)

| Dataset | Ultra-Fast | Slow Path | Speedup |
|---------|-----------|-----------|---------|
| 3M rows (300×10K, balanced) | **1.16s** | 5.70s | **4.9x** |
| 2.5M rows (unbalanced) | **1.01s** | 5.67s | **5.6x** |
| 30K rows (60×500, test) | 0.24s | 0.41s | **1.7x** |

### How It Works

**Standard Path (Slow):**
```python
# 1. Precompute data (4s for 3M rows - pandas groupby per date)
precomp = self._precompute_data()

# 2. Per-date loops with pandas operations
for t in range(n_dates):
    ranks = precomp.ranks_map[date_t]
    returns = precomp.It1[date_t]
    # ... pandas operations
```

**Ultra-Fast Path:**
```python
# 1. Convert DataFrame to numpy arrays ONCE (0.1s)
date_idx = data['date'].map(date_to_idx).values
signal = data['signal'].values
returns = data['ret'].values

# 2. Compute ALL ranks in parallel using numba (0.3s)
ranks = compute_ranks_all_dates_fast(date_idx, signal, n_dates, nport)

# 3. Compute ALL returns in parallel using numba (0.5s)
ew_ret, vw_ret = compute_all_returns_ultrafast(...)
```

### Key Functions

```python
# In numba_core.py - Ultra-fast path (Phase 9)
compute_ranks_all_dates_fast(date_idx, signal, n_dates, nport)
    -> ranks  # Portfolio rank for each observation

build_vw_lookup_and_dynamic_weights(date_idx, id_idx, vw, n_dates, n_ids)
    -> dynamic_weights  # VW from previous period for each observation

compute_all_returns_ultrafast(ret_date_idx, ret_id_idx, returns, weights,
                              form_date_idx, form_id_idx, form_ranks,
                              n_dates, n_ids, nport)
    -> (ew_returns, vw_returns)  # Shape (n_dates, nport)

compute_staggered_returns_ultrafast(...)  # For h>1 with cohort averaging

# In PyBondLab.py
_can_use_fast_path()  # Auto-detect if fast path is possible
_fit_fast_returns_only()  # Main ultra-fast path implementation
```

### Usage Example

```python
from PyBondLab import StrategyFormation, SingleSort

# Fast path is used automatically
strategy = SingleSort(holding_period=1, sort_var='signal', num_portfolios=5)
sf = StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,           # Required for fast path
    chars=None,               # Required for fast path
    banding_threshold=None,   # Required for fast path
    verbose=True              # Will print "Using ULTRA-FAST returns-only path..."
)
result = sf.fit()

# Results are identical to slow path
print(result.ea.returns.ewls_df.mean())
```

### Integration with BatchStrategyFormation

The ultra-fast path **automatically works** with `BatchStrategyFormation` when conditions are met:

```python
from PyBondLab import BatchStrategyFormation

# Each worker uses ultra-fast path automatically!
batch = BatchStrategyFormation(
    data=data,
    signals=['signal1', 'signal2', 'signal3', ...],
    holding_period=1,
    num_portfolios=5,
    turnover=False,    # <-- Enables fast path
    chars=None,        # <-- Enables fast path
    banding=None,      # <-- Enables fast path
    n_jobs=4,
)
results = batch.fit()
```

**Performance with BatchStrategyFormation (3M rows, 10 signals):**
| Configuration | Time | Notes |
|---------------|------|-------|
| turnover=True (slow) | ~57s | Each worker uses slow path |
| turnover=False (fast) | ~12s | Each worker uses ultra-fast path |
| **Speedup** | **~5x** | Plus parallel speedup from n_jobs |

### Example Script

```bash
python examples/fast_returns_only.py
```

### When Fast Path is Disabled

```python
# Any of these will force the slow path:
sf = StrategyFormation(..., turnover=True)            # Needs turnover
sf = StrategyFormation(..., chars=['signal'])         # Needs characteristics
sf = StrategyFormation(..., banding_threshold=0.2)    # Needs banding
strategy = DoubleSort(...)                             # Not SingleSort
```

### Numerical Accuracy

The fast path produces **numerically identical** results to the slow path:
- All values match within `TOLERANCE = 1e-10`
- Dynamic weights from previous period are correctly applied
- Verified against baseline test suite (12/12 tests pass)

---

## Large Panel Data Generation

For testing with large panels, use `generate_synthetic_data_fast`:

```python
from PyBondLab.pbl_test import generate_synthetic_data_fast

# Balanced panel (full date × bond matrix)
data = generate_synthetic_data_fast(
    n_dates=300,
    n_bonds=10000,
    seed=42,
    n_chars=3,
    balanced_panel=True,   # Full panel, no missing observations
)
# Shape: (3000000, 11)

# Unbalanced panel (realistic missing data)
data = generate_synthetic_data_fast(
    n_dates=300,
    n_bonds=10000,
    seed=42,
    balanced_panel=False,
    pct_active_low=0.70,   # 70-95% of bonds active per date
    pct_active_high=0.95,
)
# Shape: ~(2500000, 11)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_dates` | required | Number of monthly periods |
| `n_bonds` | required | Total number of unique bonds |
| `seed` | 0 | Random seed for reproducibility |
| `n_chars` | 3 | Number of characteristic columns (char1, char2, ...) |
| `balanced_panel` | False | If True, full date × bond panel |
| `allow_nans` | True | Inject ~1-2% NaN values (realistic) |
| `pct_active_low` | 0.70 | Min % of bonds active per date (unbalanced only) |
| `pct_active_high` | 0.95 | Max % of bonds active per date (unbalanced only) |
| `id_as_category` | True | Convert ID to category dtype (saves memory) |
| `float_dtype` | np.float32 | Use float32 for smaller memory footprint |

---

## DataUncertaintyAnalysis (User-Friendly Wrapper)

### Overview

`DataUncertaintyAnalysis` is a high-level wrapper that simplifies running data uncertainty
analysis across multiple holding periods and filter configurations. It returns long-short
factor returns for EW/VW and EA/EP combinations.

### Quick Start

```python
from PyBondLab import DataUncertaintyAnalysis

# Run analysis on a pre-computed signal
results = DataUncertaintyAnalysis(
    data=data,
    signals=['my_signal'],           # Column name(s) in data
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5, -0.3],
        'price': [50, 200],
        'bounce': [0.05, -0.05],
        'wins': [(99, 'both'), (95, 'both')],
    },
    num_portfolios=5,
    n_jobs=4,
).fit()

# Access results
results.ew_ea                        # DataFrame: dates × configs
results.summary()                    # Summary stats with NW t-stats
```

### Full API

```python
DataUncertaintyAnalysis(
    data: pd.DataFrame,

    # Signal specification (one of these required)
    signals: List[str] = None,       # Column name(s) for pre-computed signals
    strategy: Strategy = None,       # Strategy object (Momentum, LTreversal)

    # Core parameters
    holding_periods: List[int] = [1, 3, 6],
    num_portfolios: int = 5,
    dynamic_weights: bool = True,

    # Filter configurations
    filters: Dict[str, List] = None,
    include_baseline: bool = True,

    # Rating configuration
    rating: str = None,              # Single rating: 'IG', 'NIG', or None
    ratings: List[str] = None,       # Multiple ratings: ['IG', 'NIG', None] for dimension

    # Optional
    n_jobs: int = 1,                 # Parallel workers
    verbose: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | required | Bond panel data |
| `signals` | List[str] | None | Column name(s) for pre-computed signals (uses fast path) |
| `strategy` | Strategy | None | Strategy object (Momentum, LTreversal) - uses slow path |
| `holding_periods` | List[int] | [1, 3, 6] | Holding periods to test |
| `num_portfolios` | int | 5 | Number of quantile buckets |
| `dynamic_weights` | bool | True | VW from d-1 (True) or formation date (False) |
| `filters` | Dict | None | Filter configurations (see below) |
| `include_baseline` | bool | True | Always include no-filter baseline |
| `rating` | str | None | Single rating filter: 'IG', 'NIG', or None |
| `ratings` | List | None | **NEW**: Rating as dimension: `['IG', 'NIG', None]` |
| `columns` | Dict | None | Column name mapping (see below) |
| `n_jobs` | int | 1 | Parallel workers (only used by slow path) |
| `verbose` | bool | True | Show progress output |

### Column Name Mapping

If your data uses different column names than PyBondLab expects, use the `columns` parameter:

```python
# PyBondLab expected names -> Your column names
columns = {
    'date': 'date',           # Date column (usually same)
    'ID': 'cusip_id',         # Bond identifier
    'ret': 'ret',             # Return column (usually same)
    'VW': 'mcap_e',           # Value weight column
    'RATING_NUM': 'spc_rat',  # Rating column
    'PRICE': 'prc_eom',       # Price column (for price filters)
}
```

**Required columns**: `date`, `ID`, `ret`, `VW`, `RATING_NUM`
**Optional columns**: `PRICE` (only needed for price filters)

**Memory optimization**: Only specify mappings for columns that differ from defaults.
The wrapper automatically subsets data to only required columns (typically 80%+ memory reduction).

### Filter Specification

```python
filters = {
    # Trim: exclude extreme returns
    # Positive = right tail, Negative = left tail, List = both tails
    'trim': [0.2, 0.5, -0.3, [-0.3, 0.3]],

    # Price: exclude bonds with extreme prices
    # REQUIRES nested format: [[left_levels], [right_levels]]
    # Left: exclude price < threshold, Right: exclude price > threshold
    # Generates: left configs, right configs, and all left×right "both" combinations
    'price': [[1, 2, 5], [125, 150, 200]],  # 3 left + 3 right + 9 both = 15 configs

    # Bounce: exclude reversal returns
    # Positive = right tail, Negative = left tail, List = both tails
    'bounce': [0.05, -0.05, 0.10, [-0.05, 0.05]],

    # Wins: winsorize extreme returns
    # Tuple of (percentile, location) where location = 'left', 'right', 'both'
    'wins': [(99, 'both'), (99, 'right'), (95, 'both'), (95, 'left')],
}
```

### Location Inference

For **trim** and **bounce** filters, the tail location is inferred from the level:

| Level Type | Location | Example |
|------------|----------|---------|
| Positive value | `right` | `0.2` → right tail |
| Negative value | `left` | `-0.3` → left tail |
| List `[low, high]` | `both` | `[-0.3, 0.3]` → both tails |

For **wins** filters, location is explicitly specified: `(99, 'both')`.

For **price** filters, location is determined by the nested format:
- `[[left], [right]]` where left levels exclude `price < threshold`, right levels exclude `price > threshold`
- Example: `[[1, 5], [150, 200]]` generates:
  - 2 left configs: `price_1_left`, `price_5_left`
  - 2 right configs: `price_150_right`, `price_200_right`
  - 4 both configs: `price_1_150_both`, `price_1_200_both`, `price_5_150_both`, `price_5_200_both`

### Rating as a Dimension

Use the `ratings` parameter to run analysis across multiple rating categories in parallel:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['mom3_1', 'mom6_1'],  # Multiple signals supported!
    holding_periods=[1, 3],
    filters={'trim': [0.2]},
    ratings=['IG', 'NIG', None],   # Run for all rating categories
).fit()

# Results: 2 signals × 3 ratings × 2 filters × 2 HPs = 24 configs
# Column names: mom3_1_hp1_baseline_IG, mom3_1_hp1_baseline_NIG, mom3_1_hp1_baseline
```

**Rating filtering behavior:**
- Rating is applied at **formation date** only (not a pre-filter)
- Bonds are ranked based on their rating at formation
- Returns are included regardless of rating changes during holding period
- IG: RATING_NUM 1-10, NIG: RATING_NUM 11-22

**Filter by rating in results:**
```python
# Filter for specific rating categories
ig_results = results.filter(rating='IG')
nig_results = results.filter(rating='NIG')
all_bonds = results.filter(rating=None)  # No rating restriction

# Summary includes rating column
print(results.summary()[['signal', 'hp', 'rating', 'filter_type', 'ew_ea_mean']])
```

### Results Object

```python
class DataUncertaintyResults:
    """Container for data uncertainty analysis results."""

    # Factor panels: DataFrame with dates as index, configs as columns
    @property
    def ew_ea(self) -> pd.DataFrame:
        """EW Ex-Ante long-short factors."""

    @property
    def vw_ea(self) -> pd.DataFrame:
        """VW Ex-Ante long-short factors."""

    @property
    def ew_ep(self) -> pd.DataFrame:
        """EW Ex-Post long-short factors."""

    @property
    def vw_ep(self) -> pd.DataFrame:
        """VW Ex-Post long-short factors."""

    @property
    def configs(self) -> pd.DataFrame:
        """Metadata for all configurations.

        Columns: column_name, signal, hp, filter_type, level, location
        """

    def summary(self) -> pd.DataFrame:
        """Summary statistics for all configurations.

        Returns DataFrame with:
        - signal, hp, rating, filter_type, level, location
        - ew_ea_mean, ew_ea_tstat (Newey-West)
        - vw_ea_mean, vw_ea_tstat
        - ew_ep_mean, ew_ep_tstat
        - vw_ep_mean, vw_ep_tstat
        - n_obs, sharpe (annualized)

        All means in % (×100).
        Newey-West lag = int(T^0.25).
        """

    def filter(self, signal=None, hp=None, filter_type=None,
               location=None, rating=None) -> 'DataUncertaintyResults':
        """Filter to subset of configurations.

        Parameters: signal, hp, filter_type, location, rating
        Use rating=None to filter for configs with no rating restriction.
        Returns new DataUncertaintyResults with filtered configs.
        """

    def to_excel(self, path: str):
        """Export results to Excel file."""
```

### Column Naming Convention

Factor DataFrame columns use the format:
```
{signal}_hp{hp}_{filter_type}_{level}[_{location}]
```

Examples:
- `momentum_hp1_baseline`
- `momentum_hp1_trim_0.2`
- `momentum_hp3_trim_-0.3_0.3`
- `momentum_hp1_wins_99_both`

### Summary Statistics

The `summary()` method returns:

| Column | Description |
|--------|-------------|
| `signal` | Signal name |
| `hp` | Holding period |
| `filter_type` | 'baseline', 'trim', 'price', 'bounce', 'wins' |
| `level` | Filter level value |
| `location` | Tail location: 'left', 'right', 'both', or None |
| `ew_ea_mean` | EW EA mean return (%) |
| `ew_ea_tstat` | EW EA Newey-West t-statistic |
| `vw_ea_mean` | VW EA mean return (%) |
| `vw_ea_tstat` | VW EA Newey-West t-statistic |
| `ew_ep_mean` | EW EP mean return (%) |
| `ew_ep_tstat` | EW EP Newey-West t-statistic |
| `vw_ep_mean` | VW EP mean return (%) |
| `vw_ep_tstat` | VW EP Newey-West t-statistic |
| `n_obs` | Number of observations (T) |
| `sharpe` | Annualized Sharpe ratio (EW EA) |

**Newey-West t-statistics**: Uses `statsmodels` with lag = int(T^0.25).

### Signal vs Strategy

**Pre-computed signal** (most common):
```python
# Signal column already exists in data
results = DataUncertaintyAnalysis(
    data=data,
    signals=['my_momentum', 'my_value'],  # Column names
    ...
)
```

**Strategy object** (for Momentum/LTreversal):
```python
# Strategy computes signal from returns; filters affect signal computation
# Note: holding_period and num_portfolios are NOT needed here - they come from
# DataUncertaintyAnalysis parameters (holding_periods, num_portfolios)
mom = pbl.Momentum(lookback_period=3, skip=1)
results = DataUncertaintyAnalysis(
    data=data,
    strategy=mom,
    holding_periods=[1, 3, 6],  # These override any strategy values
    num_portfolios=5,           # This overrides any strategy value
    ...
)
```

Note: When using a strategy object with filters, the signal is recomputed using
filtered returns (e.g., `ret_trim` for Momentum).

**Momentum/LTreversal API:**
- `lookback_period`, `skip`: Required - define the signal computation
- `holding_period`, `num_portfolios`: Optional - only needed for standalone `StrategyFormation` use

### Multiple Signals

```python
# Test multiple signals with same filter configurations
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum', 'value', 'size'],  # Multiple columns
    holding_periods=[1, 3],
    filters={'trim': [0.2, 0.5]},
).fit()

# Access by signal
results.filter(signal='momentum').ew_ea
results.filter(signal='value').summary()
```

### Parallelization

Parallelizes over (hp × filter × signal) configurations using `n_jobs` workers:

```python
# With 4 workers and 3 signals × 3 hp × 10 filters = 90 configs
results = DataUncertaintyAnalysis(
    data=data,
    signals=['sig1', 'sig2', 'sig3'],
    holding_periods=[1, 3, 6],
    filters={'trim': [0.2, 0.5], 'price': [50, 200], ...},
    n_jobs=4,  # Parallelize across configs
).fit()
```

### Example Usage

```python
import PyBondLab as pbl
from PyBondLab import DataUncertaintyAnalysis
from PyBondLab.pbl_test import generate_synthetic_data_fast

# Generate test data
data = generate_synthetic_data_fast(n_dates=120, n_bonds=500, seed=42)

# Run data uncertainty analysis
results = DataUncertaintyAnalysis(
    data=data,
    signals=['signal1'],
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5, -0.3, [-0.3, 0.3]],
        'price': [50, 200, 500],
        'bounce': [0.05, -0.05],
        'wins': [(99, 'both'), (95, 'both')],
    },
    num_portfolios=5,
    dynamic_weights=True,
    n_jobs=4,
    verbose=True,
).fit()

# View summary statistics
print(results.summary())

# Access specific factor panels
ew_ea_factors = results.ew_ea  # DataFrame: dates × configs
vw_ep_factors = results.vw_ep

# Filter to specific configurations
hp1_results = results.filter(hp=1)
trim_results = results.filter(filter_type='trim')
right_tail = results.filter(location='right')
ig_results = results.filter(rating='IG')
all_bonds = results.filter(rating=None)  # No rating restriction

# Export to Excel
results.to_excel('data_uncertainty_results.xlsx')

# Example with ratings as a dimension
results_ratings = DataUncertaintyAnalysis(
    data=data,
    signals=['signal1', 'signal2'],    # Multiple signals - uses fast path
    holding_periods=[1, 3],
    filters={'trim': [0.2]},
    ratings=['IG', 'NIG', None],       # Run for all rating categories
    verbose=True,
).fit()

# This produces: 2 signals × 3 ratings × 2 filters × 2 HPs = 24 configs
```

### Example Script

```bash
python examples/data_uncertainty_analysis.py
```

### Fast Path Optimization (Phase 11)

When running DataUncertaintyAnalysis with pre-computed signals (not strategy objects),
an optimized fast path is automatically used that provides significant speedup.

**Performance (Blazing Fast - Parallel Numba):**
| Dataset | Fast Path | Slow Path | Speedup |
|---------|-----------|-----------|---------|
| 60 dates × 500 bonds, 20 configs | **0.43s** | 31.9s | **75x** |
| 120 dates × 1000 bonds, 50 configs | **1.2s** | ~80s | **~65x** |

**How It Works:**

The blazing fast path uses parallel numba kernels to process ALL (date × filter) combinations
simultaneously:

1. Converting DataFrame to numpy arrays once
2. Building filter masks for ALL filter configurations at once
3. Computing ranks for ALL (date × filter) combinations in parallel using `prange`
4. Building rank lookup tables for ALL filters in parallel
5. Computing portfolio returns for ALL (date × filter × hp) combinations in parallel
6. Handling ID intersection correctly (bonds must exist at formation, return, AND VW dates)

**Key Numba Kernels (in `numba_core.py`):**

```python
# Compute ranks for ALL filters at once (parallel over date × filter)
compute_ranks_all_filters(date_idx, signal, filter_masks, n_dates, nport, n_filters)
    -> ranks_all  # Shape: (n_obs, n_filters)

# Build rank lookup tables for ALL filters
build_rank_lookups_all_filters(date_idx, id_idx, ranks_all, n_dates, n_ids, n_filters)
    -> rank_lookups  # Shape: (n_dates, n_ids, n_filters)

# HP=1: Compute returns for ALL filters in parallel
compute_ls_returns_all_filters_hp1(form_date_idx, form_id_idx, ...)
    -> (ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls)  # Shape: (n_dates, n_filters)

# HP>1: Staggered rebalancing for ALL filters in parallel
compute_ls_returns_all_filters_staggered(...)
    -> (ew_ea_ls, vw_ea_ls, ew_ep_ls, vw_ep_ls)  # Shape: (n_dates, n_filters)
```

**Filter Handling:**

| Filter Type | Ranking Behavior | EA Returns | EP Returns |
|-------------|-----------------|------------|------------|
| `baseline` | Rank by valid signal | Original `ret` | Original `ret` |
| `trim` | Exclude NaN filtered ret from ranking | Original `ret` | Filtered `ret_trim` |
| `price` | Exclude NaN filtered ret from ranking | Original `ret` | Filtered `ret_price` |
| `bounce` | Exclude NaN filtered ret from ranking | Original `ret` | Filtered `ret_bounce` |
| `wins` | Rank by valid signal (clips, doesn't exclude) | Original `ret` | **NaN** (not available) |

**Key Implementation Details:**

1. **Parallel (date × filter) processing**: Each (date, filter) combination is processed
   independently using `prange`, enabling massive parallelization.

2. **Filter exclusion**: For trim/price/bounce filters, observations with NaN filtered returns
   are completely excluded from portfolio formation (not ranked, not included in returns).

3. **ID intersection**: Fast path replicates the slow path's `intersect_id` behavior:
   - Bonds must have valid signal at formation date (for ranking)
   - Bonds must have valid VW at d-1 (for weighting)
   - Bonds must have valid return at return date

4. **Wins filter EP**: The slow path returns NaN for wins EP because winsorization clips
   extreme values rather than creating a separate `ret_wins` column. Fast path matches this.

5. **`fastmath=True` disabled**: The numba kernel `compute_portfolio_returns_single` cannot
   use `fastmath=True` because it causes incorrect NaN comparisons, leading to NaN results.

**Code Location:**
- Fast path implementation: `PyBondLab/data_uncertainty.py` (`_fit_fast()`)
- Blazing fast kernels: `PyBondLab/numba_core.py`:
  - `compute_ranks_all_filters()` - Parallel rank computation
  - `build_rank_lookups_all_filters()` - Parallel lookup table building
  - `compute_ls_returns_all_filters_hp1()` - HP=1 parallel returns
  - `compute_ls_returns_all_filters_staggered()` - HP>1 parallel returns

**When Fast Path is Used:**

Fast path is automatically used when:
- `signals` parameter is provided (pre-computed signals)
- `strategy` parameter is NOT provided
- `use_fast_path=True` (default)

To force slow path for validation: `DataUncertaintyAnalysis(..., use_fast_path=False)`

**Fast Strategy Path (Phase 12):**

When a `strategy` object is provided (Momentum, LTreversal), a specialized fast strategy path
is now used that computes signals using parallel numba kernels:

```python
mom = pbl.Momentum(lookback_period=3, skip=1)
results = DataUncertaintyAnalysis(
    data=data,
    strategy=mom,           # Uses fast strategy path automatically
    holding_periods=[1, 3],
    filters={'trim': [0.2], 'wins': [(99, 'both')]},
    use_fast_path=True,     # Default - enables fast strategy path
).fit()
```

---

## Fast Strategy Path (Phase 12)

### Overview

When using `DataUncertaintyAnalysis` with a `Momentum` or `LTreversal` strategy object,
a specialized fast path computes signals using parallel numba kernels instead of pandas.

### Performance

| Dataset | Fast Path | Slow Path | Speedup |
|---------|-----------|-----------|---------|
| 60 dates × 300 bonds, 12 filters | **0.08s** | 13.1s | **163x** |
| 60 dates × 500 bonds, 12 filters | **0.10s** | ~15s | **~150x** |

### Key Numba Kernels

```python
# In numba_core.py - Panel-based signal computation
compute_momentum_signals_panel(logret_all, bond_starts, lookback, skip)
    -> signals  # Shape: (n_obs, 1)

compute_ltreversal_signals_panel(logret_all, bond_starts, lookback, skip)
    -> signals  # Shape: (n_obs, 1)

get_bond_boundaries(id_idx)
    -> bond_starts  # Array of indices where bond ID changes
```

### How It Works

1. **Sort by (ID, date)**: Data is sorted once for bond-wise processing
2. **Find bond boundaries**: `get_bond_boundaries()` identifies where each bond's data starts
3. **Parallel signal computation**: Each bond's signal is computed independently using `prange`
4. **Filter-specific signal computation**:
   - Baseline/trim/price/bounce: Signal from original returns
   - Wins: Signal from **ex-ante winsorized** returns (historical thresholds)

### Ex-Ante Winsorization for Wins Filter

For wins filters with strategy objects, the signal must be computed from winsorized returns.
The fast path uses **ex-ante (rolling historical) thresholds** to match the slow path exactly:

```python
# For each date t, thresholds come from returns BEFORE date t
def _compute_ex_ante_wins(ret, date_idx, n_dates, level, location):
    wins_ret = ret.copy()
    for d in range(n_dates):
        hist_mask = date_idx < d  # Only historical data
        hist_ret = ret[hist_mask]
        lb = np.nanpercentile(hist_ret, 100 - level)
        ub = np.nanpercentile(hist_ret, level)
        # Apply thresholds to current date
        curr_mask = date_idx == d
        wins_ret[curr_mask] = np.clip(wins_ret[curr_mask], lb, ub)
    return wins_ret
```

This ensures:
- **99 percentile wins**: Exact match (diff=0.00e+00)
- **95 percentile wins**: Exact match (diff=0.00e+00)

### Signal Computation Behavior by Filter Type

| Filter Type | Signal Computed From | EA Returns | EP Returns |
|-------------|---------------------|------------|------------|
| `baseline` | Original returns | Original `ret` | Original `ret` |
| `trim` | Original returns | Original `ret` | Filtered `ret_trim` |
| `price` | Original returns | Original `ret` | Filtered `ret_price` |
| `bounce` | Original returns | Original `ret` | Filtered `ret_bounce` |
| `wins` | **Ex-ante winsorized** returns | Original `ret` | Winsorized `ret` |

### Test Script

```bash
python examples/test_fast_strategy.py
```

This validates:
1. Numba signal computation matches pandas (< 1e-10 tolerance)
2. Fast path matches slow path for all filter types
3. Wins filter uses ex-ante thresholds correctly

---
