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

## File Changes Summary

### New Files Created

| File | Purpose | Key Functions |
|------|---------|---------------|
| `PyBondLab/numba_core.py` | Numba-optimized core kernels | See detailed section below |
| `PyBondLab/pbl_test.py` | Baseline test script (source of truth) | `run_all_baseline_tests()`, `validate_against_baseline()` |
| `PyBondLab/baseline_results/` | Stored baseline results | `baseline_results.json`, `baseline_results.pkl` |

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
- **Total**: ~3x faster single-signal, ~2.5x parallel speedup for batch
