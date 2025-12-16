# PyBondLab Numba Optimization Project

## Project Goal
Dramatically speed up portfolio formation in PyBondLab using numba/prange, while maintaining exact numerical compatibility with the existing (slow but correct) implementation.

## Current Branch
`claude/numba-portfolio-optimization-WZSI1`

---

## Performance Results (Current State)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total test suite time | 38.48s | 28.46s | **1.35x faster** |
| All 12 tests pass | YES | YES | Exact match (<1e-10) |

### Per-Test Performance (JIT Warmed)

| Test | Before | After | Speedup |
|------|--------|-------|---------|
| SingleSort hp=1, no turnover | 0.94s | 0.57s | 1.6x |
| SingleSort hp=1, turnover | 1.63s | 0.82s | 2.0x |
| SingleSort hp=3, turnover | 3.24s | 2.01s | 1.6x |
| DoubleSort hp=1, turnover | 2.41s | 2.20s | 1.1x |
| DoubleSort hp=3, turnover | 7.25s | 6.19s | 1.2x |

---

## Optimization Status

### Completed Phases

| Phase | Description | Status | Files Changed |
|-------|-------------|--------|---------------|
| **Phase 1** | Enable existing numba functions | ✅ Complete | `precompute.py`, `utils_portfolio.py` |
| **Phase 2a** | Pre-extract data to numpy arrays | ✅ Complete | `PyBondLab.py` |
| **Phase 2b** | Vectorize return/weight computation | ✅ Complete | `PyBondLab.py`, `numba_core.py` |

### Pending Phases

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| **Phase 3** | Parallelize main loop with prange | ⏳ Pending | Complex due to banding dependencies |
| **Phase 4** | Batch turnover computation | ⚠️ Infrastructure ready | Disabled - needs debugging for double sorts |

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
| `PyBondLab/utils_turnover.py` | Add fast turnover path (disabled), import numba kernels |

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

### Turnover Functions (infrastructure ready, disabled)

```python
# Batch turnover for all portfolios at once
compute_turnover_all_portfolios(ranks, positions, raw_ew, raw_vw,
                                prev_scaled_ew, prev_scaled_vw,
                                prev_sum_ew, prev_sum_vw,
                                prev_seen_ew, prev_seen_vw,
                                cohort, nport, n_assets)
    -> (turn_ew, turn_vw, new_seen_ew, new_seen_vw, curr_sum_ew, curr_sum_vw)

# Update previous weights for next period
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

### Batch Turnover (Phase 4) - Disabled

The fast turnover path in `utils_turnover.py` is disabled because it produces
slightly different results (~0.006 difference) for double sorts (unconditional).

**Location:** `utils_turnover.py:372-376`
```python
# Fast path disabled - needs debugging for double sorts
# TODO: Fix numerical differences in compute_turnover_all_portfolios
# if NUMBA_TURNOVER_AVAILABLE and state.logger is None:
#     _accumulate_turnover_fast(...)
```

**To debug:**
1. The `_accumulate_turnover_fast` function exists and is ready
2. The numba kernel `compute_turnover_all_portfolios` exists
3. Issue is likely in how positions/indices map between arrays
4. Single sorts work fine; issue appears in double sorts (25 portfolios)

### Parallelization (Phase 3) - Not Started

The main loop in `_form_cohort_portfolios` could be parallelized with `prange`,
but this is complex because:
- Banding creates dependencies between cohorts (lag_rank state)
- Turnover state needs careful handling across parallel iterations
- Without banding, parallelization is straightforward

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
                → accumulate_turnover()              ← original (fast path disabled)
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

## Future Optimization Opportunities

1. **Enable batch turnover** - Debug and enable `_accumulate_turnover_fast`
2. **Parallelize main loop** - Use `prange` when banding is disabled
3. **Pre-allocate all arrays** - Avoid repeated allocations in the loop
4. **Vectorize ID intersection** - Currently still uses pandas operations
5. **Profile with larger data** - Current tests use 500 bonds, 60 dates
