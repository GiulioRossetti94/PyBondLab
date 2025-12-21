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
| **Phase 15a** | Non-staggered rebalancing fast path | ✅ Complete | **100x speedup** achieved! |
| **Phase 15** | Non-staggered integration | ✅ Complete | BatchStrategyFormation (~340x), DataUncertaintyAnalysis integrated |
| **Phase 15b** | Non-staggered with turnover/chars/banding | ✅ Complete | **21-103x speedup**, all 6 tests PASS |
| **Phase 16** | Optimize WithinFirmSort | ✅ Complete | 16g (33x speedup) + 16h (chars) + 16i + 16j ✅ |
| **Phase 17** | Non-staggered rebalancing bug fix | ✅ Complete | Fixed: Returns now computed EVERY month |

---

## Phase 15: Non-Staggered Rebalancing Optimization

### Overview

Optimize `rebalance_frequency > 1` (quarterly, semi-annual, annual) for blazing fast performance.
Non-staggered rebalancing is **much simpler** than monthly (staggered) because:
- Single portfolio per rebalancing period (no cohort averaging)
- Direct returns (no cohort overlap)
- Simple turnover (compare successive rebalancing dates)

### ✅ Phase 15a Results (COMPLETE)

**Achieved ~100x speedup** (far exceeding the 5-7x target!):

| Configuration | Slow Path | Fast Path | Speedup |
|---------------|-----------|-----------|---------|
| Annual (freq=12, month=6) | 3.69s | 0.020s | **182x** |
| Annual (freq=12, month=1) | 1.56s | 0.018s | **87x** |
| Semi-annual (freq=6, month=6) | 1.45s | 0.016s | **89x** |
| Quarterly (freq=3, month=6) | 1.49s | 0.018s | **85x** |

**All 7 tests PASS with exact numerical match (diff=0.00e+00).**

### Implementation

#### Phase 15a: Fast Path (No Turnover/Chars/Banding) - COMPLETE

The fast path is automatically used when:
- `rebalance_frequency != 'monthly'` (quarterly, semi-annual, annual)
- `turnover=False`
- `chars=None`
- `banding=None`
- `SingleSort` only (no DoubleSort)

Key numba kernels in `numba_core.py`:
- `compute_ranks_at_rebal_dates()` - Parallel rank computation at rebal dates only
- `build_rank_lookup_nonstaggered()` - Build rank lookup table
- `compute_nonstaggered_returns_fast()` - Parallel return computation
- `build_vw_lookup_table()` - Build VW lookup table
- `compute_nonstaggered_ls_returns()` - Compute long-short returns

#### Phase 15b: Full Path (With Turnover/Chars/Banding) - COMPLETE

**Status: ALL FEATURES COMPLETE - 6/6 tests PASS**

New numba kernels added to `numba_core.py`:
- `compute_nonstaggered_weights_at_rebal()` - Compute weights at rebalancing date
- `scale_weights_by_returns()` - Scale weights by cumulative returns
- `compute_turnover_single_rebal()` - Compute turnover between rebalancings
- `compute_nonstaggered_chars_single()` - Aggregate characteristics
- `apply_banding_single_rebal()` - Apply banding logic
- `build_ret_lookup()` - Build returns lookup table
- `compute_nonstaggered_full_fast()` - Main entry point (combines all features)

**Validation Results (All PASS):**
| Feature | Status | Speedup | Notes |
|---------|--------|---------|-------|
| Returns only | ✅ PASS | **21x** | Validates against slow path |
| Turnover | ✅ PASS | **103x** | Exact numerical match |
| Characteristics | ✅ PASS | **57x** | Exact numerical match |
| Banding | ✅ PASS | **67x** | Validates against slow path |
| Turnover+chars | ✅ PASS | **93x** | All features combined |
| All features | ✅ PASS | **86x** | Turnover+chars+banding |

**Key fixes implemented:**
- Turnover: Added liquidation turnover at last date (matches slow path's `finalize_turnover()`)
- Characteristics: Fixed bond filtering to match slow path's `It1` (bonds with valid returns only)
- Characteristics: Corrected `char_date` logic based on `dynamic_weights` setting

### Timeline Behavior

```
Annual Rebalancing (June, HP=12):
================================
Year 1:
  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
  ───  ───  ───  ───  ───  [F1] ─R1─ ─R2─ ─R3─ ─R4─ ─R5─ ─R6─

Year 2:
  Jan  Feb  Mar  Apr  May  Jun  ...
  ─R7─ ─R8─ ─R9─ R10─ R11─ R12─ [F2]

Legend:
  [Fn] = Formation date (portfolio created)
  ─Rn─ = Return collected for month n
```

### ✅ Integration Points (COMPLETE)

1. **BatchStrategyFormation**: `rebalance_frequency` and `rebalance_month` parameters added
   - Ultra-fast numba path enabled for non-staggered with turnover=False (~340x speedup!)
   - 4/4 validation tests pass (annual, semi-annual, quarterly)

2. **DataUncertaintyAnalysis**: `rebalance_frequency` and `rebalance_month` parameters added
   - Slow path integration complete (3/3 tests pass)
   - Fast path disabled for non-staggered (TODO: Phase 15b)

3. **StrategyFormation**: Fast path already integrated in Phase 15a (~100x speedup)

### Validation Scripts

| Script | Description | Results |
|--------|-------------|---------|
| `examples/validate_nonstaggered_fast_path.py` | StrategyFormation fast vs slow | 7/7 PASS, ~100x speedup |
| `examples/validate_batch_nonstaggered.py` | BatchStrategyFormation fast vs slow | 4/4 PASS, ~340x speedup |
| `examples/validate_dua_nonstaggered.py` | DataUncertaintyAnalysis slow path | 3/3 PASS |
| `examples/validate_phase15b.py` | Phase 15b full features | 2/6 PASS (banding works, turnover/chars WIP) |

Documentation: `docs/NonStaggeredRebalancing_README.md`

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

## Fast Batch Path (Phase 13)

### Overview

When `BatchStrategyFormation` is called with `turnover=False`, `chars=None`, `banding=None`, and `rating=None`,
an ultra-fast path is automatically used that processes ALL signals in parallel using numba kernels.

This bypasses the multiprocessing overhead entirely and computes ranks + returns for all signals at once.

### Performance

| Dataset | Signals | Slow Path | Fast Batch | Speedup |
|---------|---------|-----------|------------|---------|
| 25K rows, HP=1 | 10 | 10.4s | 3.8s | **2.7x** |
| 25K rows, HP=3 | 10 | 4.3s | 1.5s | **2.9x** |

**Note:** Speedup improves with more signals since the fast path amortizes setup cost.

### When Fast Batch Path is Used

Fast path is **automatically enabled** when ALL of these conditions are met:

| Condition | Required Value |
|-----------|---------------|
| `turnover` | `False` |
| `chars` | `None` |
| `banding` | `None` |
| `rating` | `None` |

### Key Numba Kernels

```python
# In numba_core.py - Multi-signal batch processing
compute_ranks_all_signals(date_idx, signals, n_dates, nport, n_signals)
    -> ranks_all  # Shape: (n_obs, n_signals)

build_rank_lookups_all_signals(date_idx, id_idx, ranks_all, n_dates, n_ids, n_signals)
    -> rank_lookups  # Shape: (n_dates * n_ids, n_signals)

compute_ls_returns_all_signals_hp1(...)  # HP=1
    -> (ew_ls, vw_ls)  # Shape: (n_dates, n_signals)

compute_ls_returns_all_signals_staggered(...)  # HP>1
    -> (ew_ls, vw_ls)  # Shape: (n_dates, n_signals)
```

### Usage Example

```python
from PyBondLab import BatchStrategyFormation

# Fast path is used automatically when conditions are met
batch = BatchStrategyFormation(
    data=data,
    signals=['signal1', 'signal2', 'signal3', ...],  # Can be 100+ signals
    holding_period=1,
    num_portfolios=5,
    turnover=False,   # Required for fast path
    chars=None,       # Required for fast path
    banding=None,     # Required for fast path
    rating=None,      # Required for fast path
    n_jobs=4,         # Ignored when fast path is used
    verbose=True,
)
results = batch.fit()  # Prints "FAST BATCH PATH: Processing N signals..."

# Access results (same API)
ew_ls, vw_ls = results['signal1'].get_long_short()
```

### Validation Script

```bash
python examples/validate_fast_batch.py
python examples/validate_fast_batch.py --hp 3 --n_signals 20
```

Validation confirms fast batch matches slow path within machine epsilon (< 1e-17).

---

## Filter Support for BatchStrategyFormation (Phase 14) - COMPLETE

### Overview

Phase 14 extends `BatchStrategyFormation` to support `rating` and `subset_filter` parameters
in the ultra-fast numba path, while avoiding look-ahead bias.

### Current State

| Parameter | Slow Path | Fast Path | Status |
|-----------|-----------|-----------|--------|
| `rating='IG'` or `'NIG'` | ✅ Works | ✅ Works | **Complete** |
| `rating=(min, max)` tuple | ✅ Works | ✅ Works | **Complete** |
| `subset_filter={...}` | ✅ Works | ✅ Works | **Complete** |

**All 14 validation tests pass with machine epsilon tolerance (<1e-17).**

### The Look-Ahead Bias Problem

**Naive filtering introduces bias:**
```python
# WRONG: This filters based on characteristics at ALL dates
data = data[data['RATING_NUM'] <= 10]  # Look-ahead bias!
```

**Example scenario:**
```
Date t   (formation): Bond has RATING_NUM=10 (IG) → Should be ranked
Date t+1 (return):    Bond has RATING_NUM=11 (downgraded) → Return should be collected!
```

With naive filtering, the (t+1, bond) row is removed entirely, losing the return observation.
This uses future information (the downgrade) to exclude returns - look-ahead bias.

### Correct Approach: Filter at Formation Only

**The key insight:** Filters should be applied only when computing ranks (formation date),
not when collecting returns. The fast path already has this structure:

```python
# Return computation looks up rank from FORMATION date
for i in range(n_obs):
    d = date_idx[i]           # Return date (t+1)
    bond = id_idx[i]

    form_d = d - 1            # Formation date (t)
    rank = rank_lookups[form_d, bond, s]  # Was bond ranked at formation?

    if rank > 0:              # Bond was in a portfolio at formation
        # Include this return (regardless of current characteristics)
```

### Implementation: Set Signal to NaN for Filtered Observations

Instead of removing rows, we set the signal to NaN for observations that don't pass the filter:

```python
def _fit_fast_batch(self) -> BatchResults:
    data = self.data  # DON'T pre-filter rows!

    # Build filter mask (True = passes filter at this date)
    filter_mask = np.ones(len(data), dtype=np.bool_)

    if self.rating is not None:
        rating_vals = data['RATING_NUM'].values
        if self.rating == 'IG':
            filter_mask &= (rating_vals <= 10)
        elif self.rating == 'NIG':
            filter_mask &= (rating_vals > 10)
        elif isinstance(self.rating, tuple):
            min_r, max_r = self.rating
            filter_mask &= (rating_vals >= min_r) & (rating_vals <= max_r)

    if self.subset_filter is not None:
        for col, (min_val, max_val) in self.subset_filter.items():
            col_vals = data[col].values
            filter_mask &= (col_vals >= min_val) & (col_vals <= max_val)

    # Set signals to NaN for filtered-out observations
    # Existing rank code already skips NaN → these bonds won't be ranked
    signals_matrix = np.empty((len(data), n_signals), dtype=np.float64)
    for s_idx, signal in enumerate(self.signals):
        sig_vals = data[signal].values.astype(np.float64)
        sig_vals[~filter_mask] = np.nan  # Filter out at formation
        signals_matrix[:, s_idx] = sig_vals

    # Rank computation: only ranks non-NaN signals (filtered bonds excluded)
    ranks_all = compute_ranks_all_signals(...)

    # Return computation: uses FULL returns array (no filtering)
    ret = data['ret'].values.astype(np.float64)  # ALL returns
```

### Why This Works

| Step | What Happens | Look-Ahead Bias? |
|------|--------------|------------------|
| Formation (t) | Check if bond passes filter → if yes, rank it | No - decision at t |
| Rank lookup | Look up rank assigned at formation date | No - uses t info |
| Return (t+1) | Collect return if bond was ranked | No - includes downgrades |

**Example with downgrade scenario:**
```
Date t:   RATING_NUM=10, passes IG filter → ranked, assigned to portfolio 3
Date t+1: RATING_NUM=11 (downgraded)
          - Rank lookup uses formation date (t) → rank = 3
          - Return IS collected (bond was in portfolio at formation)
```

### Implementation Plan

**Step 1: Add `subset_filter` to slow path**
- Add parameter to `BatchStrategyFormation.__init__`
- Thread through to worker functions
- Pass to `DataConfig(subset_filter=...)`

**Step 2: Enable fast path with filters**
- Remove `if self.rating is not None: return False` check
- Build filter mask in `_fit_fast_batch`
- Apply mask to signals (set to NaN) before ranking
- Keep returns/VW arrays unfiltered

**Step 3: Validation**
Compare fast batch vs slow SingleSort:
- `rating=(1, 10)` with HP=1 and HP=3
- `rating=(11, 22)` (NIG) with HP=1
- `subset_filter={'char1': (min, max)}` test case
- Verify differences < 1e-10

### New API

```python
from PyBondLab import BatchStrategyFormation

batch = BatchStrategyFormation(
    data=data,
    signals=['signal1', 'signal2'],
    holding_period=3,
    num_portfolios=5,
    turnover=False,

    # NEW: Filter support (now works with fast path!)
    rating=(1, 10),                    # IG bonds only (or 'IG', 'NIG')
    subset_filter={
        'MATURITY': (1, 5),            # Maturity 1-5 years
        'DURATION': (2, 8),            # Duration 2-8 years
    },

    verbose=True,
)
results = batch.fit()  # Uses fast path with filters!
```

### Expected Performance

| Configuration | Time (est.) | Notes |
|---------------|-------------|-------|
| No filter (fast path) | ~1.0s | Current fast path |
| With rating filter (fast path) | ~1.1s | +0.1s for filter mask |
| With subset_filter (fast path) | ~1.1s | +0.1s for filter mask |
| With filter (slow path) | ~15-30s | Current fallback |

### Validation Script

```bash
python examples/validate_fast_batch_filters.py
```

Tests:
- `rating=(1, 10)` vs `rating='IG'` (should be identical)
- Fast batch with rating vs slow SingleSort with rating
- Fast batch with subset_filter vs slow SingleSort with subset_filter
- HP=1 and HP=3 for all configurations

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
- **Phase 13**: Fast Batch Path for BatchStrategyFormation (**2.7-3x speedup**)
  - Processes ALL signals in parallel using numba kernels
  - `compute_ranks_all_signals()`: Ranks for all signals at once
  - `compute_ls_returns_all_signals_hp1/staggered()`: Returns for all signals
  - Bypasses multiprocessing overhead when turnover=False, chars=None, banding=None
  - Exact match with slow path (< 1e-17 tolerance)
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
    rating: Union[str, Tuple[int, int]] = None,  # 'IG', 'NIG', None, or (min, max) tuple
    ratings: List[...] = None,       # Multiple ratings as dimension: ['IG', 'NIG', (1, 10), None]

    # Subset filter (characteristic-based filtering)
    subset_filter: Dict[str, Tuple[float, float]] = None,  # e.g., {'MATURITY': (1, 5)}

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
| `rating` | str/tuple | None | Single rating filter: 'IG', 'NIG', None, or tuple `(min, max)` |
| `ratings` | List | None | Rating as dimension: `['IG', 'NIG', (1, 10), None]` |
| `subset_filter` | Dict | None | **NEW**: Characteristic filter: `{'col': (min, max)}` |
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

# Example with rating tuple (custom range)
results_custom_rating = DataUncertaintyAnalysis(
    data=data,
    signals=['signal1'],
    holding_periods=[1, 3],
    rating=(7, 10),                    # Custom rating range (BBB+ to BBB-)
    verbose=True,
).fit()

# Example with subset_filter (characteristic-based filtering)
results_filtered = DataUncertaintyAnalysis(
    data=data,
    signals=['signal1'],
    holding_periods=[1, 3],
    subset_filter={
        'MATURITY': (1, 5),            # Maturity 1-5 years
    },
    verbose=True,
).fit()

# Example with combined rating and subset_filter
results_combined = DataUncertaintyAnalysis(
    data=data,
    signals=['signal1'],
    holding_periods=[1, 3],
    rating='IG',                       # Investment grade only
    subset_filter={
        'DURATION': (2, 8),            # Duration 2-8 years
    },
    verbose=True,
).fit()
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

## Phase 16: WithinFirmSort Optimization

### Overview

Optimize the `WithinFirmSort` strategy for faster portfolio formation. This strategy is fundamentally different from SingleSort/DoubleSort because it sorts bonds **within each firm**, isolating within-firm bond dispersion from cross-firm differences.

### Bug Fixes (Completed)

#### Fix 1: Fast Path Exclusion

**Issue**: WithinFirmSort was incorrectly using the fast path when `turnover=False`.

**Root cause**: `_can_use_fast_path()` only checked for DoubleSort, not WithinFirmSort. Since WithinFirmSort is not a DoubleSort, it passed all checks and incorrectly used `_fit_fast_returns_only()`.

**Fix**: Added check in `_can_use_fast_path()` (PyBondLab.py:1391-1394):
```python
# WithinFirmSort requires special handling (within-firm grouping, rating bins, etc.)
is_within_firm = getattr(self.strategy, "__strategy_name__", "") == "Within-Firm Sort"
if is_within_firm:
    return False
```

#### Fix 2: Result Indexing (Return Date)

**Issue**: Results in `port_idx` were indexed by FORMATION date instead of RETURN date.

**Root cause**: `_form_single_period()` stored `port_idx[date_t]` where `date_t` is the formation date.
For consistency with SingleSort/DoubleSort and correct factor return alignment, results should be
indexed by the return date (`date_t1`).

**Fix**: Modified `_form_single_period()` signature to accept `date_t1` parameter and changed:
```python
# Before: self.port_idx[date_t] = weights_df
# After:
self.port_idx[date_t1] = weights_df  # Index by return date
```

Both calling sites (`_form_cohort_portfolios`, `_form_nonstaggered_portfolio`) updated to pass `date_t1`.

#### Fix 3: HP>1 Disabled

**Issue**: HP>1 (staggered rebalancing) had fundamental bugs in cohort averaging.

**Root cause**: In `_form_cohort_portfolios`, `port_idx[date_t]` was overwritten in the horizon loop,
so only the LAST horizon's data was retained. For HP=3, this means only returns from `form_date + 3`
were used, instead of averaging across all 3 cohorts.

**Fix**: Disabled HP>1 for WithinFirmSort until proper cohort averaging is implemented:
```python
# In StrategyClass.py WithinFirmSort.__init__():
if holding_period > 1:
    raise ValueError(
        f"WithinFirmSort currently only supports holding_period=1. "
        f"Got holding_period={holding_period}. "
        f"HP>1 staggered rebalancing has known bugs and is disabled."
    )
```

### Current Baseline Performance

| Configuration | Time | Notes |
|---------------|------|-------|
| HP=1, no turnover | ~1.0s | Test data: 11,940 rows |
| HP=1, with turnover | ~0.9s | 199 bonds, 50 firms, 60 dates |

**Note**: HP>1 is currently disabled (raises ValueError).

### Feature Support

| Feature | Supported | Notes |
|---------|-----------|-------|
| **Turnover** | ✅ YES | Uses standard PyBondLab machinery |
| **HP>1 (Staggered)** | ❌ DISABLED | Cohort averaging bug - raises ValueError |
| **Chars** | ⏳ TODO | Will implement with Option B aggregation |
| **Banding** | ❌ NO | Not applicable - only HIGH/LOW portfolios |

**Why no banding?** WithinFirmSort only has 2 portfolios (HIGH and LOW). Banding prevents
reassignment when rank changes by less than `banding/nport`. With nport=2, banding=1 would
require a change of 0.5 (i.e., moving from one portfolio to the other), which is always
the case when rank changes. Therefore banding is meaningless for WithinFirmSort.

**Why HP>1 disabled?** The current cohort loop overwrites `port_idx` on each iteration,
so only the last horizon's data is retained for aggregation. Proper fix requires restructuring
to store all cohort data and average correctly.

**Chars aggregation (Option B):** For chars, we need to average characteristics at the
formation month using the same hierarchical aggregation as returns:
1. Within each firm: Compute VW-average char for HIGH and LOW portfolios
2. Across firms: Cap-weight the firm-level chars within each rating tercile
3. Across ratings: Simple average across rating terciles

### WithinFirmSort Architecture

#### How It Differs from SingleSort

| Aspect | SingleSort | WithinFirmSort |
|--------|-----------|----------------|
| **Grouping** | None (cross-sectional) | Date × Rating Tercile × Firm |
| **Percentiles** | Global (20/40/60/80) | Within-firm (33.3/66.7) |
| **Portfolios** | N portfolios | 2 (high/low only) |
| **Return Aggregation** | Simple VW average | Firm-cap-weighted → Rating-averaged |

#### Key Files

| File | Purpose |
|------|---------|
| `StrategyClass.py` | `WithinFirmSort` class definition |
| `utils_within_firm.py` | Core computation functions |
| `precompute.py:370-413` | Integration with precomputation |
| `PyBondLab.py:2185-2300` | Integration with aggregation |

#### Key Functions

1. **`compute_within_firm_portfolios()`** - Creates rating terciles, groups by (date, rating_terc, firm), assigns bonds to high/low
2. **`compute_within_firm_assignments_numba()`** - Numba-compiled core for percentile thresholds and assignment
3. **`compute_within_firm_returns_aggregation()`** - Hierarchical aggregation: firm-cap-weighted → rating-averaged

### Optimization Plan

#### Phase 16a: Profile and Identify Bottlenecks

**Status: ✅ COMPLETE**

Profiling revealed aggregation as the major bottleneck (78% of time):
- `compute_within_firm_returns_aggregation()`: **6.49s (78%)**
- `_precompute_data()`: 1.15s (14%)
- `_form_cohort_portfolios()`: 0.63s (8%)
- Portfolio assignment: 0.02s (<1%)

The pandas-based nested loop aggregation (60 dates × 3 ratings × ~20 firms) was the culprit.

#### Phase 16b: Vectorize Portfolio Assignment

**Status: Pending** (Low priority - only 0.02s)

Current implementation in `compute_within_firm_portfolios()`:
- Creates pandas groupby for (date, rating_terc, firm)
- Calls numba kernel per group
- Already fast (~0.02s), not a bottleneck

#### Phase 16c: Vectorize Return Aggregation

**Status: ✅ COMPLETE**

Added `compute_within_firm_aggregation_fast()` numba kernel that:
1. Single-pass accumulation over all (date, rating_terc, firm, portfolio) groups
2. Vectorized firm-level H-L factor computation
3. Vectorized cap-weighted aggregation across firms
4. Vectorized averaging across rating terciles
5. **Properly computes BOTH EW and VW** (bug fix: EW was incorrectly copied from VW)

**EW vs VW Bug Fix:**
- **Issue**: EW and VW long-short returns were identical (EW was copied from VW)
- **Root cause**: Code explicitly set `ewls_series = vwls_series` (PyBondLab.py:2226-2230)
- **Fix**: Modified kernel to return 6 arrays (ew_ls, vw_ls, ew_high, ew_low, vw_high, vw_low)
- Now properly:
  - EW: EW returns within firm → equal-weight across firms → avg across ratings
  - VW: VW returns within firm → cap-weight across firms → avg across ratings

**Performance:**
- Aggregation kernel: **0.0001s** (was 2.8s = **40,000x speedup**)
- Including prep: 0.008s (= **350x speedup**)
- Overall: **3-4x speedup** (bottleneck moved elsewhere)

#### Phase 16d: Create Ultra-Fast Path for WithinFirmSort

**Status: ❌ FAILED - REVERTED**

An attempt was made to create an ultra-fast path for WithinFirmSort similar to SingleSort.
The implementation was reverted due to critical issues.

**What Was Attempted:**
1. Add `compute_within_firm_assignments_all_dates()` numba kernel for parallel portfolio assignment
2. Add `_can_use_withinfirm_fast_path()` to detect when fast path is applicable
3. Add `_fit_withinfirm_fast()` for ultra-fast WithinFirmSort execution
4. Bypass `_precompute_data()` entirely and work directly with numpy arrays

**Critical Issues Discovered:**

1. **Severe Performance Regression:**
   - Fast path took 105+ seconds on real data (2.4M rows, 50k cusips)
   - Slow path was ~15 seconds
   - Root cause: Python loops iterating over (n_dates × n_bonds) to collect returns
   - The approach was fundamentally wrong - needed numba vectorization, not Python loops

2. **Incorrect Result Indexing:**
   - Fast path indexed results by FORMATION date
   - This does NOT match SingleSort/DoubleSort which index by RETURN date
   - WithinFirmSort should align with standard strategies for consistency

3. **Potential HP>1 Bug in Slow Path (NEEDS INVESTIGATION):**
   - For HP>1, `port_idx` dictionary is overwritten in the loop over horizons `h`
   - Only the LAST horizon's data (h=hp-1) is retained in `port_idx`
   - This means `compute_within_firm_returns_aggregation()` only sees one horizon's returns
   - **Expected behavior**: For HP=3, each return should be average of 3 cohorts
   - **Actual behavior**: Only uses returns from `form_date + hp` (last horizon)
   - This may be a BUG in the slow path that needs fixing BEFORE fast path work

**Evidence of HP>1 Issue:**
```python
# In _form_cohort_portfolios, for HP=3:
for h in range(self.hor):  # h = 0, 1, 2
    # ... calls _form_single_period ...
    # At end of loop:
    self.port_idx[date_t] = weights_df  # OVERWRITES each iteration!
# Result: port_idx[date_t] only contains h=2 data (return at form_date + 3)
```

Tested with synthetic data where returns differ by date:
- Formation date 0: port_idx stores return from date 3 (not average of dates 1, 2, 3)
- Formation date 1: port_idx stores return from date 4 (not average of dates 2, 3, 4)

**What Needs To Be Done:**

1. **FIRST: Fix HP>1 behavior in slow path (if it's a bug)**
   - Investigate whether `port_idx` overwrite is intentional or a bug
   - If bug: Modify to store all horizon returns and average properly
   - If intentional: Document the rationale clearly

2. **THEN: Implement fast path correctly**
   - Use vectorized numba kernels (not Python loops)
   - Index results by RETURN date (matching SingleSort/DoubleSort)
   - For HP>1, implement proper cohort averaging
   - Target: Match slow path results exactly, then optimize

3. **Design Considerations for Correct Fast Path:**
   - Build rank lookup table: `(form_date, bond_id) -> (rank, rating_terc, firm_idx, vw)`
   - For each return date, look up ranks from all contributing formation dates
   - Collect returns and aggregate with proper cohort averaging
   - Use prange parallelization for the return collection loop

**Reverted Commit:** `a88ad94` was reverted to `34f291c`

#### Phase 16e: Implement Chars Support

**Status: Pending**

Add characteristics aggregation using hierarchical aggregation (same as returns):
1. At each formation date, compute VW-average char for HIGH and LOW within each firm
2. Aggregate across firms using cap-weighting within each rating tercile
3. Average across rating terciles
4. Output: DataFrame with columns `['LOW', 'HIGH']` for each characteristic

**Implementation Steps:**
1. Slow path first (validate correctness)
2. Fast path with numba kernel `compute_withinfirm_chars_all_dates()`

#### Phase 16f: Turnover Optimization

**Status: Pending**

Optimize turnover computation for WithinFirmSort:
1. Profile current implementation
2. Identify bottlenecks
3. Optimize with numba if needed

---

## Phase 16g: WithinFirmSort Fast Path + BatchWithinFirmSortFormation

### Overview

Create an ultra-fast path for WithinFirmSort (HP=1, no turnover, no chars) and a new
`BatchWithinFirmSortFormation` class for processing multiple signals in parallel.

**Status: ✅ COMPLETE**

### Performance Results

| Configuration | Before | After | Speedup |
|---------------|--------|-------|---------|
| StrategyFormation HP=1, no turnover | ~1.0s | **0.03s** | **33x** |
| BatchWithinFirmSortFormation fast path | N/A | Works | Exact match |
| BatchWithinFirmSortFormation slow path (turnover) | N/A | Works | Uses multiproc |

**All validation tests pass with exact numerical match (diff=0.00e+00).**

### Architecture

```
BatchWithinFirmSortFormation
│
├── Fast path (numba vectorized):
│   ├── Conditions: turnover=False AND chars=None
│   ├── Pre-compute rating terciles (ONCE)
│   ├── Pre-compute firm groupings (ONCE)
│   ├── Compute HIGH/LOW for ALL signals at once (numba)
│   └── Aggregate returns for ALL signals (numba)
│
└── Slow path (multiprocessing):
    ├── Conditions: turnover=True OR chars is not None
    ├── Uses ProcessPoolExecutor with n_jobs workers
    ├── Each worker: StrategyFormation(WithinFirmSort, signal=X)
    └── Collect and merge results
```

### Key Insight

For WithinFirmSort with multiple signals:
- **Rating terciles**: Computed ONCE (depends on `RATING_NUM`, not signal)
- **Firm groupings**: Computed ONCE (depends on `PERMNO`, not signal)
- **HIGH/LOW assignment**: Computed N times (one per signal) - CAN BE VECTORIZED
- **Return aggregation**: Computed N times (one per signal) - CAN BE VECTORIZED

### Implementation Plan

#### Step 1: Fast Path in StrategyFormation (Phase 16g.1)

Add `_can_use_withinfirm_fast_path()` and `_fit_withinfirm_fast()`:

```python
def _can_use_withinfirm_fast_path(self):
    """Check if WithinFirmSort fast path can be used."""
    if self.strategy.__strategy_name__ != "Within-Firm Sort":
        return False
    if self.turnover:
        return False
    if self.chars:
        return False
    if self.hor != 1:
        return False
    return True
```

**Key numba kernels to add:**
- `compute_withinfirm_assignments_all_dates()` - Parallel assignment across all dates
- `compute_withinfirm_returns_all_dates()` - Parallel return aggregation

#### Step 2: BatchWithinFirmSortFormation Fast Path (Phase 16g.2)

New class for batch processing multiple signals with numba vectorization:

```python
class BatchWithinFirmSortFormation:
    """
    Batch processing for WithinFirmSort with multiple signals.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signals : list of str
        Column names to use as sorting signals
    firm_id_col : str, default='PERMNO'
        Firm identifier column
    rating_bins : list, optional
        Rating bin edges (default: [-inf, 7, 10, inf])
    min_bonds_per_firm : int, default=2
        Minimum bonds per firm-date-rating group
    turnover : bool, default=False
        Compute turnover (uses slow path)
    chars : list of str, optional
        Characteristics to aggregate (uses slow path)
    rating : str or tuple, optional
        Rating filter ('IG', 'NIG', or (min, max))
    subset_filter : dict, optional
        Characteristic-based filters
    n_jobs : int, default=1
        Parallel workers (for slow path)
    verbose : bool, default=True
        Show progress
    """
```

**Fast batch path (turnover=False, chars=None):**
```python
def _fit_fast_batch_withinfirm(self):
    # 1. Pre-compute rating terciles (ONCE)
    # 2. Pre-compute firm groupings (ONCE)
    # 3. Compute HIGH/LOW for ALL signals at once
    # 4. Build rank lookups for ALL signals
    # 5. Aggregate returns for ALL signals in parallel
```

**Key numba kernels:**
- `compute_withinfirm_assignments_all_signals()` - Vectorized assignment
- `compute_withinfirm_returns_all_signals()` - Vectorized aggregation

#### Step 3: BatchWithinFirmSortFormation Slow Path (Phase 16g.3)

When `turnover=True` or `chars` is set, use multiprocessing:

```python
def _fit_slow_batch_withinfirm(self):
    """Slow path using multiprocessing for turnover/chars."""
    from concurrent.futures import ProcessPoolExecutor

    def process_signal(signal):
        strategy = WithinFirmSort(
            holding_period=1,
            sort_var=signal,
            firm_id_col=self.firm_id_col,
            rating_bins=self.rating_bins,
            min_bonds_per_firm=self.min_bonds_per_firm,
        )
        sf = StrategyFormation(
            data=self.data,
            strategy=strategy,
            turnover=self.turnover,
            chars=self.chars,
        )
        return signal, sf.fit()

    with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        results = dict(executor.map(process_signal, self.signals))

    return results
```

### Validation Strategy

Batch results must match standalone StrategyFormation exactly:

```python
# Validation: batch results should match standalone
batch = BatchWithinFirmSortFormation(data, signals=['sig1', 'sig2'], ...)
batch_results = batch.fit()

# Compare with standalone
for sig in signals:
    standalone = StrategyFormation(data, WithinFirmSort(sort_var=sig), ...).fit()
    assert batch_results[sig].ew_ls == standalone.ew_ls  # Must match exactly
```

### Target Performance

| Configuration | Current | Target | Speedup |
|---------------|---------|--------|---------|
| 1 signal, HP=1, no turnover | ~1.0s | <0.2s | **5x** |
| 10 signals, HP=1, no turnover | ~10s | <0.5s | **20x** |
| 50 signals, HP=1, no turnover | ~50s | <1.5s | **30x+** |
| 10 signals, with turnover (slow) | ~10s | ~3s | **3x** (multiproc) |

### File Changes

| File | Changes |
|------|---------|
| `PyBondLab/PyBondLab.py` | Add `_can_use_withinfirm_fast_path()`, `_fit_withinfirm_fast()` |
| `PyBondLab/numba_core.py` | Add WithinFirmSort numba kernels |
| `PyBondLab/batch_withinfirm.py` | **NEW** - `BatchWithinFirmSortFormation` class |
| `PyBondLab/__init__.py` | Export `BatchWithinFirmSortFormation` |
| `examples/validate_batch_withinfirm.py` | **NEW** - Validation script |

---

## Phase 16h: Chars Support for WithinFirmSort (HP=1)

### Overview

Add characteristics aggregation using the same hierarchical structure as returns.

**Status: ✅ COMPLETE**

### Key Point: Chars at Formation Date

**IMPORTANT**: Characteristics are computed at **formation date (t)**, not return date (t+1).
This is standard PyBondLab behavior and must be preserved.

```
Formation date t:
  1. For each (date_t, rating_terc, firm):
     - Bonds assigned to HIGH → compute VW-weighted average of char
     - Bonds assigned to LOW → compute VW-weighted average of char
  2. Across firms (within rating_terc): cap-weight firm-level chars
  3. Across rating_terc: simple average

Char values come from formation date data (It0/It1m), not return date.
port_idx is indexed by return date (t+1), so chars need to use date lookup to formation date (t).
```

### Aggregation Logic

For each characteristic at each formation date:
1. **Within-firm**: Compute VW-average char for HIGH and LOW portfolios
2. **Across firms**: Cap-weight the firm-level chars within each rating tercile
3. **Across ratings**: Simple average across rating terciles

### Output Format

```python
# For single signal (via StrategyFormation):
ew_chars, vw_chars = result.get_characteristics()
# Returns tuple of dicts:
# ew_chars = {'char1': DataFrame(LOW, HIGH), 'char2': DataFrame(LOW, HIGH), ...}
# vw_chars = {'char1': DataFrame(LOW, HIGH), 'char2': DataFrame(LOW, HIGH), ...}

# For batch (multiple signals via BatchWithinFirmSortFormation):
batch_results['signal1'].get_characteristics()
# Same format as above (tuple of dicts)
```

### Implementation Details

**Numba kernel added** (`numba_core.py`):
```python
compute_within_firm_chars_aggregation(
    date_idx, id_idx, firm_idx, rating_terc, ptf_rank, char_values, vw, n_dates, n_firms
) -> (ew_low, ew_high, vw_low, vw_high)
```

**Key implementation points**:
1. Uses hierarchical aggregation matching returns structure
2. port_idx indexed by return date (t+1), so chars lookup uses formation date (t = t+1 - 1)
3. Date mapping created: `{return_date: formation_date}` for char value lookup
4. Merge chars from raw data at formation date, then aggregate using numba kernel

### Validation Results

All tests pass:
```
EW Chars keys: ['char1', 'char2']
VW Chars keys: ['char1', 'char2']

char1:
  EW shape: (15, 2), VW shape: (15, 2)
  EW non-NaN: LOW=14, HIGH=14
  VW non-NaN: LOW=14, HIGH=14
```

### Batch Support

When `chars` is set, `BatchWithinFirmSortFormation` automatically uses the slow path
with multiprocessing. Each worker runs `StrategyFormation(chars=...)` for one signal.

---

## Phase 16i: Turnover Status (HP=1)

### Overview

Verify turnover works correctly for WithinFirmSort.

**Status: ✅ ALREADY OPTIMIZED**

### Current State

Turnover for WithinFirmSort uses the **same machinery as SingleSort/DoubleSort**,
which was optimized in Phase 4 with numba kernels (`_accumulate_turnover_fast`).

**No additional optimization work is needed.**

### Verification

Turnover is computed at **bond level** (not hierarchical):
- Each bond has individual weights (eweights, vweights)
- Turnover = sum(|current_weight - previous_weight|) / 2
- Standard PyBondLab turnover tracking applies

### Code Reference

```python
# In PyBondLab.py, WithinFirmSort uses same turnover as SingleSort:
if self.turnover and not result['weights_df'].empty:
    self.turnover_manager.accumulate(
        self.turnover_state,
        self.cohort,
        tot_nport,
        t_idx,
        result['weights_df'],
        result['weights_scaled_df']
    )

# This calls _accumulate_turnover_fast() which uses numba kernels
```

### Batch Support

When `turnover=True` in `BatchWithinFirmSortFormation`:
- Uses slow path with multiprocessing
- Each worker runs `StrategyFormation(turnover=True)` for one signal
- Turnover computed using optimized Phase 4 kernels

No additional work needed for Phase 16i.

### Performance Results (Phase 16c)

**Small Test Data (11,940 rows, 199 bonds, 50 firms, 60 dates):**

| Configuration | Before | After | Achieved Speedup |
|---------------|--------|-------|------------------|
| HP=1, no turnover | 3.8s | **0.94s** | **4.0x** ✅ |
| HP=1, with turnover | 3.7s | **0.94s** | **3.9x** ✅ |

**Note**: HP>1 is now disabled due to cohort averaging bugs (see Fix 3 above).

**Large Data Profiling (matching user's data: 2.4M rows, ~50k cusips, 2.9k firms, 272 dates):**

| Component | Time | % of Total |
|-----------|------|------------|
| `_precompute_data()` | **10.06s** | 58% |
| `compute_within_firm_portfolios()` (272 calls) | **6.09s** | 35% |
| Aggregation (fast kernel) | 1.39s | 8% |
| **Total** | **17.5s** | 100% |

**Key Insight**: After Phase 16c, the bottleneck shifted to `_precompute_data()` and
per-date portfolio assignment calls. Phase 16g will bypass both by computing everything
directly from numpy arrays.

### Target Performance (Future Phases)

| Configuration | Current | Target | Status |
|---------------|---------|--------|--------|
| HP=1, no turnover, no chars | 0.94s | <0.2s | Phase 16g |
| HP=1, with chars | TBD | <0.3s | Phase 16h |
| HP=1, turnover + chars | TBD | TBD | Phase 16i |

### Validation Script

```bash
python examples/validate_withinfirmsort.py
```

Tests:
1. Basic execution
2. turnover=True vs turnover=False consistency (bug fix validation)
3. HP=3 staggered rebalancing
4. Difference from SingleSort (confirms within-firm logic is applied)

### Documentation

See `docs/WithinFirmSort_README.md` for detailed documentation on:
- Methodology
- Usage examples
- Architecture
- Comparison with standard sorting

---

## Phase 16j: Align BatchWithinFirmSortFormation with BatchStrategyFormation

### Overview

Align `BatchWithinFirmSortFormation` with `BatchStrategyFormation` to provide consistent
user experience: column mapping, progress bars, memory optimization, and summary output.

**Status: ✅ Complete**

### Target Output

```
Columns renamed: cusip->ID, ret_vwx_bgn->ret, mcap_e->VW, spc_rat->RATING_NUM
Processing 11 signals with 10 worker(s)...
  Platform: Windows, using 'spawn' start method
  Running first signal (warmup)...
  First signal done: 9.74s
  Processing 10 remaining signals in parallel...
  Data size: 1247.8MB → 48.1MB per worker (96% reduction)
Parallel processing: 100%|██████████| 10/10 [00:13<00:00,  1.37s/it]

============================================================
BATCH PROCESSING COMPLETE
============================================================
Total signals:    11
Successful:       11
Failed:           0
Workers used:     10
Total time:       25.04s
Avg time/signal:  9.46s
Effective speedup: 4.2x
============================================================
```

### Implementation Phases

#### Phase 16j.1: Column Mapping
- Add `columns` parameter to `BatchWithinFirmSortFormation.__init__`
- Rename columns at start (same logic as `BatchStrategyFormation`)
- Report: `Columns renamed: cusip->ID, ...`

#### Phase 16j.2: Warmup + Progress
- Run first signal sequentially (warmup JIT compilation)
- Add tqdm progress bar for remaining signals
- Track timing per signal

#### Phase 16j.3: Memory Optimization
- Implement `_get_minimal_data()` for WithinFirmSort
- Required columns: `date`, `ID`, `ret`, `VW`, `RATING_NUM`, `PERMNO` (firm ID), + signals + chars
- Report data size reduction: `Data size: X MB → Y MB per worker (Z% reduction)`

#### Phase 16j.4: Platform-Aware Start Method
- Use `spawn` on Windows, `fork` on Linux/macOS
- Report: `Platform: Windows, using 'spawn' start method`

#### Phase 16j.5: Summary Output
- Match summary format from `BatchStrategyFormation`
- Include: total signals, success/fail, workers, time, speedup

#### Phase 16j.6: Shared Base Class
- Extract common functionality into `BaseBatchFormation`
- Both `BatchStrategyFormation` and `BatchWithinFirmSortFormation` inherit from it
- Common code: column mapping, warmup, progress, memory optimization, summary output

### Key Constraint

**Keep numba fast path** for `turnover=False` in `BatchWithinFirmSortFormation`:
- Fast path: numba vectorized processing when `turnover=False` and `chars=None`
- Slow path: multiprocessing when `turnover=True` or `chars` is set

### Files to Modify

| File | Changes |
|------|---------|
| `PyBondLab/batch_base.py` | **NEW** - `BaseBatchFormation` class with shared functionality |
| `PyBondLab/batch.py` | Inherit from `BaseBatchFormation`, remove duplicated code |
| `PyBondLab/batch_withinfirm.py` | Inherit from `BaseBatchFormation`, add new features |
| `PyBondLab/__init__.py` | Export `BaseBatchFormation` (optional) |

### Validation

After implementation, both classes should produce identical output format:

```python
# BatchStrategyFormation
batch1 = BatchStrategyFormation(data, columns={...}, signals=[...], n_jobs=4)
results1 = batch1.fit()  # Shows progress, summary

# BatchWithinFirmSortFormation
batch2 = BatchWithinFirmSortFormation(data, columns={...}, signals=[...], n_jobs=4)
results2 = batch2.fit()  # Shows SAME progress format, summary format
```

---

## Phase 17: Non-Staggered Rebalancing Bug Fix

### Overview

**CRITICAL BUG (FIXED)**: When using non-staggered rebalancing (`rebalance_frequency != 'monthly'`),
returns/turnover/chars were only computed at rebalancing dates + 1, instead of EVERY month.

**Status: ✅ Complete**

### Fix Summary

**Phase 17a (Fast Path)**: Fixed `compute_nonstaggered_full_fast()` in `numba_core.py`
- Changed from limiting to `hp` months to iterating until next rebalancing date
- Now properly collects returns for ALL months between rebalancing dates

**Phase 17b (Slow Path)**: Fixed `_form_nonstaggered_portfolio()` in `PyBondLab.py`
- Passes `rebal_dates_idx` to function
- Iterates through ALL months until next rebalancing (not just `self.hor` times)
- Fixes DoubleSort with non-staggered rebalancing

**Validation Results:**
- SingleSort quarterly: 9/9 expected dates ✅
- DoubleSort quarterly: 9/9 expected dates ✅
- All 12 baseline tests pass ✅

### Bug Description

With quarterly rebalancing (rebalance_frequency='quarterly') and HP=1:
- **Expected**: Returns for ALL 11 months (months 2-12)
- **Actual**: Returns only for 3 months (months 4, 7, 10) - only at rebalancing dates + 1

```
Quarterly Rebalancing Timeline (BUG):
=====================================

Month:   1     2     3     4     5     6     7     8     9    10    11    12
         |                 |                 |                 |
         [R1]              [R2]              [R3]              [R4]
               X     X     ✓     X     X     ✓     X     X     ✓     X     X

Where:
  [Rn] = Rebalancing date (portfolio formed)
  ✓    = Return computed (BUG: only one per quarter)
  X    = Return MISSING (should be computed!)
```

**Expected Behavior:**
```
Month:   1     2     3     4     5     6     7     8     9    10    11    12
         |                 |                 |                 |
         [R1]              [R2]              [R3]              [R4]
               ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓     ✓

All months should have returns. Portfolio composition stays fixed between
rebalancing dates. Weights renormalized if bonds drop out.
```

### Root Cause

**Fast Path (`compute_nonstaggered_full_fast` in `numba_core.py`):**
```python
# Line 4612-4614 - BUG: hp limits return collection
if d > form_d + hp:
    continue
```
With `hp=1`, this only collects 1 month of returns per rebalancing date.

**Slow Path (`_form_nonstaggered_portfolio` in `PyBondLab.py`):**
```python
# Line 2034 - BUG: hor (holding_period) limits return collection
for h in range(self.hor):
    t1_idx = rebal_idx + h + 1
```
With `hor=1`, this loop only runs once per rebalancing.

### Correct Behavior

1. **Returns computed EVERY month** after first rebalancing date
   - At each month (rebalancing OR non-rebalancing), treat it as a pseudo-formation period
   - Check intersection of valid bonds at pseudo-formation date (t) and return date (t+1)
   - Only include bonds that exist at BOTH dates in return calculation
   - This matches the monthly rebalancing `intersect_id()` logic

2. **Portfolio composition fixed** until next rebalancing (ranks stay the same)
   - Ranks assigned at rebalancing date persist until next rebalancing
   - We do NOT add new bonds between rebalancing dates
   - But the set of bonds in each portfolio can SHRINK if bonds drop out

3. **Weights renormalized** when bonds drop out
   - At each month, renormalize weights based on which bonds are still present
   - EW: weight = 1/n_remaining_bonds (not original n_bonds)
   - VW: weight = VW_i / sum(VW_j for remaining bonds)
   - Weights always sum to 1 within each portfolio

4. **Turnover between rebalancing dates**
   - **NOT always 0!**
   - If bonds drop out, there IS "latent turnover" from weight renormalization
   - Only 0 for perfectly balanced panels with no dropouts
   - This matches real-world behavior: dropping a position induces turnover

5. **Chars computed** using current bond universe (renormalized like weights)

### Pseudo-Formation Logic (Critical)

```
Quarterly Rebalancing Example:
==============================

December (REAL rebalancing):
  1. Compute ranks from signal
  2. Assign bonds to portfolios 1-5
  3. Check intersection with January returns
  4. Compute weights for bonds in intersection
  5. Collect January returns

January (PSEUDO-formation, no rebalancing):
  1. Ranks stay fixed from December
  2. Check intersection: bonds at Jan with valid Feb returns
  3. Some bonds may have dropped out!
  4. Renormalize weights for remaining bonds
  5. Collect February returns
  6. Turnover from weight changes (if any bonds dropped)

February (PSEUDO-formation, no rebalancing):
  1. Ranks stay fixed from December
  2. Check intersection: bonds at Feb with valid Mar returns
  3. Renormalize weights
  4. Collect March returns
  5. Turnover from weight changes

March (REAL rebalancing):
  1. Compute NEW ranks from signal
  2. Assign bonds to portfolios (may differ from December)
  3. Check intersection with April returns
  4. Compute weights
  5. Collect April returns
  6. Turnover from rank changes + weight changes
```

### Implementation Plan

#### Phase 17a: Fix Fast Path (numba kernel)

**File**: `numba_core.py`

**Changes to `compute_nonstaggered_full_fast()`:**

1. Replace `hp` check with next-rebalancing check:
   ```python
   # OLD (BUG):
   if d > form_d + hp:
       continue

   # NEW (FIX):
   # Find next rebalancing date
   next_rebal_d = n_dates  # default: no more rebalancing
   for r in range(n_rebal):
       if rebal_date_indices[r] > form_d:
           next_rebal_d = rebal_date_indices[r]
           break

   # Skip if return date is at or after next rebalancing
   if d >= next_rebal_d:
       continue
   ```

2. Add pseudo-formation intersection logic:
   ```python
   # For each return date d, treat (d-1) as pseudo-formation date
   # Check intersection: bonds with valid data at (d-1) AND valid returns at d
   # This matches the monthly rebalancing intersect_id() behavior
   for i in range(n_obs):
       if date_idx[i] == d:
           bond = id_idx[i]
           rank = rank_lookup[form_d, bond]  # Rank from REAL formation date
           if np.isnan(rank):
               continue
           # Bond must have valid VW at pseudo-formation (d-1)
           if dynamic_weights:
               weight = vw_lookup[d - 1, bond]
           else:
               weight = vw_lookup[form_d, bond]
           if np.isnan(weight):
               continue
           # Include this bond in return calculation
   ```

3. Renormalize weights for bond dropouts:
   ```python
   # After collecting bonds for return date d:
   # - Some bonds from formation may be missing at d (dropped out)
   # - Renormalize weights so they sum to 1 within each portfolio
   for p in range(nport):
       if ptf_vw_sum[p] > 0:
           # EW: already handled by counting remaining bonds
           # VW: divide by sum of VW for remaining bonds
   ```

4. Compute turnover at EVERY date (not just rebalancing):
   ```python
   # Turnover occurs even between rebalancing dates if bonds drop out
   # The weight renormalization induces "latent turnover"
   # Use same turnover logic as monthly rebalancing
   ```

#### Phase 17b: Fix Slow Path

**File**: `PyBondLab.py`

**Changes to `_form_nonstaggered_portfolio()`:**

Replace `for h in range(self.hor):` loop with proper date iteration:

```python
def _form_nonstaggered_portfolio(self, rebal_idx, precomp, ...):
    """Form portfolio for one rebalancing period (non-staggered)."""

    # Find next rebalancing date index
    next_rebal_idx = len(self.datelist)  # default: end of data
    for idx in rebal_dates_idx:
        if idx > rebal_idx:
            next_rebal_idx = idx
            break

    # Collect returns for ALL months until next rebalancing
    for t1_idx in range(rebal_idx + 1, next_rebal_idx + 1):
        if t1_idx >= len(self.datelist):
            break

        # Pseudo-formation date is t1_idx - 1
        pseudo_form_idx = t1_idx - 1
        pseudo_form_date = self.datelist[pseudo_form_idx]

        # Get data at pseudo-formation and return dates
        # Use intersect_id() to find bonds present at BOTH dates
        It0_pseudo = precomp.It0.get(pseudo_form_date, pd.DataFrame())
        It1 = precomp.It1.get(date_t1, pd.DataFrame())
        It1m = precomp.It1m.get(pseudo_form_date, pd.DataFrame())
        It0_pseudo, It1, It1m = intersect_id(It0_pseudo, It1, It1m, self.dynamic_weights)

        # Map ranks from ORIGINAL formation date (not pseudo-formation)
        It1['ptf_rank'] = It1[ColumnNames.ID].map(
            precomp.ranks_map.get(self.datelist[rebal_idx], pd.Series())
        )

        # ... existing return/weight computation code ...
        # Turnover is computed at EVERY date (latent turnover from dropouts)
```

#### Phase 17c: Weight Renormalization

When bonds drop out between rebalancing dates, weights must be renormalized:

**Example:**
```
Formation (month 3):
  Portfolio 1: Bond A (40%), Bond B (30%), Bond C (30%)

Month 4 (Bond B drops out):
  Portfolio 1: Bond A (40%/70% = 57.1%), Bond C (30%/70% = 42.9%)

Month 5 (all bonds present):
  Portfolio 1: Bond A (40%), Bond B (30%), Bond C (30%)  # Original weights restored
```

**For EW portfolios:**
- Original weight = 1/n_bonds_in_portfolio
- After dropout: weight = 1/n_remaining_bonds

**For VW portfolios:**
- Original weight = VW_i / sum(VW_j for j in portfolio)
- After dropout: weight = VW_i / sum(VW_j for j in remaining bonds)

**Key insight**: We need to track which bonds were assigned to each portfolio at formation,
then at each return date, compute weights only for bonds still present.

#### Phase 17d: Validation

Create validation script `examples/validate_phase17.py` that tests:

1. **Balanced panel (no dropouts):**
   - All months should have returns (except first formation month)
   - Weights remain constant (no bonds dropping out)
   - Turnover = 0 between rebalancing dates (only for balanced panel!)

2. **Unbalanced panel (10% random dropouts):**
   - All months should have returns
   - Weights renormalized when bonds drop out
   - Verify renormalization is correct (weights sum to 1)
   - Turnover > 0 even between rebalancing dates (latent turnover)

3. **Multiple frequencies:**
   - Quarterly (freq=3)
   - Semi-annual (freq=6)
   - Annual (freq=12)

4. **With turnover/chars:**
   - Turnover computed at EVERY date (not just rebalancing)
   - Chars computed with renormalized weights at each date

5. **Fast vs slow path comparison:**
   - Results must match exactly (< 1e-10 tolerance)

### Key Technical Details

**Rank persistence:**
- Ranks assigned at formation date stay fixed until next rebalancing
- Already stored in `rank_lookup[form_d, bond_id]`

**Weight sources:**
- `dynamic_weights=True`: Use VW from d-1 for date d returns
- `dynamic_weights=False`: Use VW from formation date
- Either way, weights are renormalized based on which bonds are present at d

**Turnover semantics:**
- Turnover computed at EVERY date, same as monthly rebalancing
- Between rebalancing dates: turnover from weight changes when bonds drop out
- Turnover = 0 ONLY if perfectly balanced panel with no dropouts

### Files to Modify

| File | Changes |
|------|---------|
| `PyBondLab/numba_core.py` | Fix `compute_nonstaggered_full_fast()` - date iteration, weight renorm |
| `PyBondLab/PyBondLab.py` | Fix `_form_nonstaggered_portfolio()` - date iteration, weight renorm |
| `examples/validate_phase17.py` | **NEW** - Phase 17 validation script |
| `examples/validate_nonstaggered_bug.py` | Update expected behavior after fix |

### Expected Results After Fix

**Quarterly Rebalancing (12 months):**
| Metric | Before (Bug) | After (Fix) |
|--------|--------------|-------------|
| Return dates | 3 | **11** |
| Turnover dates | 3 | **11** (computed at every date) |
| Chars dates | 3 | **11** |

**Turnover behavior by panel type:**
| Panel Type | Turnover Between Rebalancing |
|------------|------------------------------|
| Balanced (no dropouts) | 0 (weights unchanged) |
| Unbalanced (dropouts) | > 0 (latent turnover from weight changes) |

### Validation Script Location

`examples/validate_nonstaggered_bug.py` - already created, confirms the bug exists.

After Phase 17 implementation, this script should report "0 bugs detected".

---

## Phase 18: NamingConfig - Factor Naming Quality of Life

### Overview

Add a naming configuration system for factor outputs. This provides:
- Consistent, readable factor names (lowercase, signal-based)
- Rating suffixes (`_ig`, `_hy`)
- Sign correction with `*` suffix (independent for EW/VW)
- Within-firm suffix (`_wf`)
- Optional weighting prefix (`ew_`, `vw_`)
- Factor-level turnover computation

**Status: 🚧 In Progress**

### Design Decisions

| Feature | Implementation |
|---------|---------------|
| Naming style | Lowercase by default |
| Rating suffixes | `_ig` (investment grade), `_hy` (high yield) |
| Sign correction | `*` suffix (EW and VW independent) |
| Within-firm suffix | `_wf` |
| Weighting prefix | Optional `ew_`, `vw_` |
| Base name | Use signal name (e.g., `cs` instead of `EWEA_ALL_1`) |
| DoubleSort separator | `_` (e.g., `cs_duration`) |
| WithinFirmSort portfolios | `_low`, `_high` |
| Factor turnover | `get_turnover(level='factor')` returns `(P_N + P_1) / 2` |
| Backward compatibility | Old names without `naming=` parameter |

### NamingConfig Dataclass

**File:** `PyBondLab/naming.py` (new file)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class NamingConfig:
    """Configuration for factor naming conventions.

    Attributes
    ----------
    lowercase : bool, default=True
        Use lowercase names (e.g., 'cs' instead of 'CS').
    sign_correct : bool, default=False
        Flip negative factors and add '*' suffix.
        Applied independently for EW and VW.
    use_signal_name : bool, default=True
        Use signal column name as base (e.g., 'cs' instead of 'EWEA_ALL_1').
    weighting_prefix : bool, default=False
        Add 'ew_' or 'vw_' prefix to factor names.
    include_rating_suffix : bool, default=True
        Add '_ig' or '_hy' suffix for rating-filtered strategies.
    include_wf_suffix : bool, default=True
        Add '_wf' suffix for WithinFirmSort strategies.
    doublesort_sep : str, default='_'
        Separator for DoubleSort factor names (e.g., 'cs_duration').

    Examples
    --------
    Default config (recommended):
    >>> cfg = NamingConfig()
    >>> # SingleSort: 'cs', 'cs_ig', 'cs*'
    >>> # DoubleSort: 'cs_duration'
    >>> # WithinFirmSort: 'cs_wf', portfolios: 'cs_low', 'cs_high'

    With weighting prefix:
    >>> cfg = NamingConfig(weighting_prefix=True)
    >>> # 'ew_cs', 'vw_cs', 'ew_cs_ig', 'vw_cs*'

    Without signal name (legacy style):
    >>> cfg = NamingConfig(use_signal_name=False)
    >>> # 'factor', 'factor_ig', 'factor*'
    """
    lowercase: bool = True
    sign_correct: bool = False
    use_signal_name: bool = True
    weighting_prefix: bool = False
    include_rating_suffix: bool = True
    include_wf_suffix: bool = True
    doublesort_sep: str = '_'


def make_factor_name(
    signal_name: str,
    config: NamingConfig,
    *,
    weighting: Optional[str] = None,  # 'ew' or 'vw'
    rating: Optional[str] = None,     # 'ig' or 'hy'
    is_within_firm: bool = False,
    sign_corrected: bool = False,
    second_signal: Optional[str] = None,  # For DoubleSort
) -> str:
    """Generate factor name based on config.

    Parameters
    ----------
    signal_name : str
        Base signal name (e.g., 'CS', 'duration').
    config : NamingConfig
        Naming configuration.
    weighting : str, optional
        'ew' or 'vw' for weighting prefix.
    rating : str, optional
        'ig' or 'hy' for rating suffix.
    is_within_firm : bool
        Add '_wf' suffix for WithinFirmSort.
    sign_corrected : bool
        Add '*' suffix if sign was flipped.
    second_signal : str, optional
        Second signal for DoubleSort (e.g., 'duration').

    Returns
    -------
    str
        Formatted factor name.
    """
    # Build base name
    if config.use_signal_name:
        name = signal_name
    else:
        name = 'factor'

    # Apply case
    if config.lowercase:
        name = name.lower()
        if second_signal:
            second_signal = second_signal.lower()

    # DoubleSort: add second signal
    if second_signal:
        name = f"{name}{config.doublesort_sep}{second_signal}"

    # WithinFirmSort suffix
    if is_within_firm and config.include_wf_suffix:
        name = f"{name}_wf"

    # Rating suffix
    if rating and config.include_rating_suffix:
        name = f"{name}_{rating}"

    # Weighting prefix
    if weighting and config.weighting_prefix:
        name = f"{weighting}_{name}"

    # Sign correction suffix (added last)
    if sign_corrected:
        name = f"{name}*"

    return name


def make_portfolio_name(
    signal_name: str,
    portfolio_num: int,
    num_portfolios: int,
    config: NamingConfig,
    *,
    second_signal: Optional[str] = None,
    second_portfolio_num: Optional[int] = None,
    is_within_firm: bool = False,
) -> str:
    """Generate portfolio name (e.g., 'cs1', 'cs5', 'cs_low', 'cs_high').

    For WithinFirmSort: portfolio_num 1 = LOW, 2 = HIGH.
    """
    base = signal_name.lower() if config.lowercase else signal_name

    if is_within_firm:
        # WithinFirmSort: use _low, _high
        suffix = '_low' if portfolio_num == 1 else '_high'
        return f"{base}{suffix}"

    if second_signal:
        # DoubleSort: cs1_dur1, cs5_dur5
        sec = second_signal.lower() if config.lowercase else second_signal
        return f"{base}{portfolio_num}{config.doublesort_sep}{sec}{second_portfolio_num}"

    # SingleSort: cs1, cs2, ..., cs5
    return f"{base}{portfolio_num}"
```

### Updated StrategyResults Getter Methods

**File:** `PyBondLab/StrategyResultsClass.py`

```python
class StrategyResults:
    def get_long_short(
        self,
        naming: Optional[NamingConfig] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Get long-short factor returns.

        Parameters
        ----------
        naming : NamingConfig, optional
            If provided, rename output series using naming conventions.
            If None, use legacy names (backward compatible).

        Returns
        -------
        ew_ls : pd.Series
            Equal-weighted long-short returns.
        vw_ls : pd.Series
            Value-weighted long-short returns.
        """
        ew_ls = self.ea.returns.ewls_df.copy()
        vw_ls = self.ea.returns.vwls_df.copy()

        if naming is not None:
            # Apply sign correction if enabled
            ew_sign_corrected = False
            vw_sign_corrected = False

            if naming.sign_correct:
                if ew_ls.mean() < 0:
                    ew_ls = -ew_ls
                    ew_sign_corrected = True
                if vw_ls.mean() < 0:
                    vw_ls = -vw_ls
                    vw_sign_corrected = True

            # Generate names
            ew_name = make_factor_name(
                self._signal_name,
                naming,
                weighting='ew',
                rating=self._rating_str,
                is_within_firm=self._is_within_firm,
                sign_corrected=ew_sign_corrected,
            )
            vw_name = make_factor_name(
                self._signal_name,
                naming,
                weighting='vw',
                rating=self._rating_str,
                is_within_firm=self._is_within_firm,
                sign_corrected=vw_sign_corrected,
            )

            ew_ls.name = ew_name
            vw_ls.name = vw_name

        return ew_ls, vw_ls

    def get_turnover(
        self,
        level: str = 'portfolio',
        naming: Optional[NamingConfig] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
        """Get turnover statistics.

        Parameters
        ----------
        level : str, default='portfolio'
            'portfolio' - Return turnover per portfolio (DataFrame).
            'factor' - Return factor turnover: (P_N + P_1) / 2 (Series).
        naming : NamingConfig, optional
            If provided, rename output using naming conventions.

        Returns
        -------
        ew_turnover, vw_turnover : tuple
            Turnover statistics (DataFrames for 'portfolio', Series for 'factor').
        """
        ew_turn = self.ea.turnover.ewturn_df.copy()
        vw_turn = self.ea.turnover.vwturn_df.copy()

        if level == 'factor':
            # Factor turnover = (P_N + P_1) / 2 = average of long and short legs
            nport = ew_turn.shape[1]
            ew_factor = (ew_turn.iloc[:, 0] + ew_turn.iloc[:, nport - 1]) / 2
            vw_factor = (vw_turn.iloc[:, 0] + vw_turn.iloc[:, nport - 1]) / 2

            if naming is not None:
                ew_name = make_factor_name(
                    self._signal_name, naming,
                    weighting='ew', rating=self._rating_str,
                    is_within_firm=self._is_within_firm,
                )
                vw_name = make_factor_name(
                    self._signal_name, naming,
                    weighting='vw', rating=self._rating_str,
                    is_within_firm=self._is_within_firm,
                )
                ew_factor.name = f"{ew_name}_turnover"
                vw_factor.name = f"{vw_name}_turnover"

            return ew_factor, vw_factor

        # Portfolio-level turnover (default)
        if naming is not None:
            # Rename columns: 1, 2, ..., N -> cs1, cs2, ..., csN
            ew_cols = [
                make_portfolio_name(self._signal_name, i + 1, len(ew_turn.columns), naming)
                for i in range(len(ew_turn.columns))
            ]
            vw_cols = [
                make_portfolio_name(self._signal_name, i + 1, len(vw_turn.columns), naming)
                for i in range(len(vw_turn.columns))
            ]
            ew_turn.columns = ew_cols
            vw_turn.columns = vw_cols

        return ew_turn, vw_turn
```

### Implementation Steps

#### Step 1: Create naming.py module
- Create `PyBondLab/naming.py` with `NamingConfig` dataclass
- Add `make_factor_name()` and `make_portfolio_name()` functions
- Add unit tests

#### Step 2: Update StrategyResults
- Add `_signal_name`, `_rating_str`, `_is_within_firm` attributes
- Update `get_long_short()` to accept `naming` parameter
- Update `get_turnover()` to accept `level` and `naming` parameters
- Update `get_characteristics()` to accept `naming` parameter

#### Step 3: Update StrategyFormation
- Pass signal name, rating, strategy type to StrategyResults
- Ensure backward compatibility (no naming = legacy behavior)

#### Step 4: Update BatchStrategyFormation
- Results dict uses signal names as keys (already done)
- Each result has proper metadata for naming

#### Step 5: Validation Script
- Test all naming configurations
- Verify sign correction works independently for EW/VW
- Verify factor turnover computation
- Verify backward compatibility

### Naming Examples

**SingleSort:**
```python
result = StrategyFormation(data, SingleSort(sort_var='cs'), rating='IG').fit()
cfg = NamingConfig()

ew_ls, vw_ls = result.get_long_short(naming=cfg)
# ew_ls.name = 'cs_ig'
# vw_ls.name = 'cs_ig'

cfg_prefix = NamingConfig(weighting_prefix=True)
ew_ls, vw_ls = result.get_long_short(naming=cfg_prefix)
# ew_ls.name = 'ew_cs_ig'
# vw_ls.name = 'vw_cs_ig'

cfg_sign = NamingConfig(sign_correct=True)
ew_ls, vw_ls = result.get_long_short(naming=cfg_sign)
# If EW mean < 0 and VW mean > 0:
# ew_ls.name = 'cs_ig*'  (flipped)
# vw_ls.name = 'cs_ig'   (not flipped)
```

**DoubleSort:**
```python
result = StrategyFormation(data, DoubleSort(sort_var='cs', cond_var='duration')).fit()
cfg = NamingConfig()

ew_ls, vw_ls = result.get_long_short(naming=cfg)
# ew_ls.name = 'cs_duration'
# vw_ls.name = 'cs_duration'
```

**WithinFirmSort:**
```python
result = StrategyFormation(data, WithinFirmSort(sort_var='cs')).fit()
cfg = NamingConfig()

ew_ls, vw_ls = result.get_long_short(naming=cfg)
# ew_ls.name = 'cs_wf'
# vw_ls.name = 'cs_wf'

# Portfolio names
ew_turn, vw_turn = result.get_turnover(naming=cfg)
# ew_turn.columns = ['cs_low', 'cs_high']
```

**Factor Turnover:**
```python
result = StrategyFormation(data, SingleSort(sort_var='cs')).fit()
cfg = NamingConfig()

ew_turn, vw_turn = result.get_turnover(level='factor', naming=cfg)
# ew_turn.name = 'cs_turnover'
# Returns: (P_5 + P_1) / 2 for quintile portfolios
```

### Files to Create/Modify

| File | Changes |
|------|---------|
| `PyBondLab/naming.py` | **NEW** - NamingConfig dataclass and utility functions |
| `PyBondLab/StrategyResultsClass.py` | Update getter methods with `naming` parameter |
| `PyBondLab/PyBondLab.py` | Pass metadata to StrategyResults |
| `PyBondLab/__init__.py` | Export `NamingConfig` |
| `examples/validate_naming.py` | **NEW** - Validation script |

### Validation Script

```bash
python examples/validate_naming.py
```

Tests:
1. Basic naming (lowercase, signal-based)
2. Rating suffix (_ig, _hy)
3. Sign correction (EW and VW independent)
4. Weighting prefix (ew_, vw_)
5. DoubleSort separator
6. WithinFirmSort suffix and portfolios
7. Factor turnover computation
8. Backward compatibility (no naming = legacy names)

---
