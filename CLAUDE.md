# PyBondLab Numba Optimization Project

## Project Goal
Dramatically speed up portfolio formation in PyBondLab using numba/prange, while maintaining exact numerical compatibility with the existing (slow but correct) implementation.

## Branch
`claude/optimize-portfolio-numba-JZmnm`

---

## Architecture Overview

### Key Files and Their Roles

| File | Purpose | Performance Critical |
|------|---------|---------------------|
| `PyBondLab/PyBondLab.py` | Main `StrategyFormation` class, orchestrates portfolio formation | **YES - Main bottleneck** |
| `PyBondLab/precompute.py` | `PrecomputeBuilder` - precomputes ranks, weights, formation data | **YES** |
| `PyBondLab/utils_turnover.py` | `TurnoverManager`, `TurnoverState`, turnover computation | **YES - Very slow** |
| `PyBondLab/utils_portfolio.py` | Portfolio weight/return calculations, banding logic | **YES** |
| `PyBondLab/utils.py` | Core sorting utilities (thresholds, bins, double sorts) | YES |
| `PyBondLab/utils_optimized.py` | **EXISTING numba versions** (not fully utilized!) | Already optimized |
| `PyBondLab/StrategyClass.py` | Strategy definitions (SingleSort, DoubleSort, Momentum) | No |
| `PyBondLab/FilterClass.py` | Data filtering (trim, winsorize, price, bounce) | No |
| `PyBondLab/constants.py` | Column names, defaults, rating bounds | No |

### Data Flow
```
User calls StrategyFormation.fit()
    → _validate_input()
    → _precompute_formation_data()  [PrecomputeBuilder]
        → For each date: compute thresholds, assign bins, store ranks
    → _form_cohort_portfolios() OR _form_nonstaggered_portfolio()
        → For each (date, holding_period):
            → _form_single_period()  [MAIN BOTTLENECK]
                → intersect_id()
                → map ranks
                → compute weights (groupby)
                → compute returns (groupby)
                → if turnover: accumulate_turnover()
    → _finalize_results()
```

---

## Identified Performance Bottlenecks

### 1. Main Loop in `_form_cohort_portfolios` (PyBondLab.py:1269-1337)
- **Problem:** Serial Python for-loop over dates × holding periods
- **Scale:** 192 dates × 6 holding periods = 1,152 iterations
- **Solution:** Restructure for numba prange parallelization

### 2. `_form_single_period` Method (PyBondLab.py:1402-1536)
- **Problem:** Heavy pandas operations per call (groupby, map, merge)
- **Called:** Once per (date, holding_period) combination
- **Key operations:**
  - `intersect_id()` - DataFrame filtering
  - `It1['ID'].map(ranks_map)` - Series lookup
  - `groupby().size()`, `groupby().sum()` - Weight computation
  - `groupby().mean()` - Return aggregation
- **Solution:** Pre-extract to numpy arrays, vectorize across all periods

### 3. Turnover Computation (utils_turnover.py:269-472)
- **Problem:** Called for every portfolio at every timestep
- **Scale:** 10 portfolios × 192 dates × 6 cohorts = ~11,520 inner loop calls
- **Current state:** Has `_sum_min_prev_raw` numba function, but outer loop is Python
- **Solution:** Batch portfolio loop, vectorize weight updates

### 4. Unused Optimizations
- **`utils_optimized.py` has numba versions that are NOT being used:**
  - `compute_thresholds_optimized()`
  - `assign_bond_bins_numba()`
  - `double_sort_uncond_numba()`
  - `double_sort_cond_numba()`
- **Quick win:** Switch imports to use these

---

## Optimization Plan

### Phase 1: Enable Existing Numba Functions
- Switch `utils.py` imports to use `utils_optimized.py` versions
- Expected speedup: 2-5x for precomputation phase

### Phase 2: Vectorize Core Operations
- Pre-extract all data to numpy arrays before main loop
- Create 2D arrays: `ranks[date_idx, bond_idx]`, `weights[date_idx, bond_idx]`
- Replace per-period groupby with numpy advanced indexing

### Phase 3: Parallelize Main Loop
- Restructure `_form_cohort_portfolios` for numba prange
- Each iteration must be independent (no shared state except output arrays)

### Phase 4: Batch Turnover Computation
- Pre-allocate all turnover state arrays
- Convert portfolio loop to numba-compiled batch function
- Vectorize weight state updates

---

## Test Cases for `pbl_test.py`

The test script establishes **SOURCE OF TRUTH** results from the original (slow) code.
Future optimized code MUST match these results exactly.

### Required Test Configurations

| # | Sort Type | holding_period | banding | turnover | chars |
|---|-----------|----------------|---------|----------|-------|
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

### Values to Capture for Each Test
1. **Factor means:** EW and VW long-short portfolio returns (mean across time)
2. **Turnover:** Mean EW and VW turnover per portfolio (when enabled)
3. **Characteristics:** Mean characteristic value per portfolio (when chars provided)
4. **Portfolio counts:** Number of bonds per portfolio (sanity check)

### Test Tolerance
- Returns: exact match (< 1e-10 difference)
- Turnover: exact match (< 1e-10 difference)
- Characteristics: exact match (< 1e-10 difference)

---

## Key Implementation Details

### Banding Parameter
- Type: `int` (commonly 1 or 2)
- Meaning: Number of portfolio ranks a bond must move to be reassigned
- Logic location: `PyBondLab.py:1557-1597` and `utils_portfolio.py:376-413`
- Formula: `threshold = banding / nport` (e.g., banding=1, nport=10 → threshold=0.1)

### Characteristics (chars)
- Parameter: `chars=["var1", "var2"]` - list of column names
- Computes portfolio-level means (EW and VW) for each characteristic
- Logic: `utils_portfolio.py:262-307` `compute_portfolio_characteristics()`

### dynamic_weights
- **Always True** - deprecated False option, never use
- Means: Use weights from t+h-1 for return calculation at t+h

### Holding Period & Staggered Rebalancing
- `holding_period=1`: No staggering, monthly rebalance
- `holding_period>1`: Staggered cohorts
  - Example: `holding_period=6` creates 6 overlapping cohorts
  - Each cohort formed in different month, held for 6 months
  - Final return = average across cohorts

### Double Sort Methods
- **Unconditional:** Sort on var1, sort on var2 independently, combine ranks
- **Conditional:** Sort on var1, then within each var1 bin, sort on var2
- Strategy classes: `DoubleSort` in `StrategyClass.py:66-108`

---

## Column Name Reference (from constants.py)

```python
# Core columns
DATE = 'date'
ID = 'ID'
RETURN = 'ret'
RATING = 'RATING_NUM'
VALUE_WEIGHT = 'VW'

# Derived columns
PORTFOLIO_RANK = 'ptf_rank'
EQ_WEIGHTS = 'eweights'
VAL_WEIGHTS = 'vweights'
COUNT = 'count'
```

---

## Synthetic Data Generation

For testing, generate synthetic bond panel with:
- ~200-500 bonds
- ~36-60 months
- Required columns: date, ID, ret, RATING_NUM, VW
- Add 2-3 random characteristic columns for chars testing
- Include some NaN values (realistic)
- Bonds should enter/exit (varying availability per date)

Example from `test_synthetic_validation.py:25-107`

---

## Files to Create/Modify

### New Files
- `PyBondLab/pbl_test.py` - Baseline test script (source of truth)
- `PyBondLab/numba_core.py` (future) - Numba-optimized core functions

### Files to Modify (future optimization)
- `PyBondLab/PyBondLab.py` - Main class, add optimized code path
- `PyBondLab/precompute.py` - Switch to numba utils
- `PyBondLab/utils_turnover.py` - Vectorize turnover loop

---

## Important Notes

1. **Never change FilterClass.py behavior** - filters must work identically
2. **dynamic_weights=True always** - don't optimize for False case
3. **Banding is integer** - not float, not boolean
4. **Source of truth = original slow code** - new code must match exactly
5. **Test with turnover=True** - this is the slow path we must optimize
6. **Test holding_period=1 AND holding_period=3** - different code paths

---

## Quick Reference: Running Tests

```python
# From PyBondLab-Dev directory
import sys
sys.path.insert(0, '.')

# Import local version (not installed package)
from PyBondLab import PyBondLab as pbl

# Or import specific classes
from PyBondLab.PyBondLab import StrategyFormation
from PyBondLab.StrategyClass import SingleSort, DoubleSort
```

---

## Git Workflow
```bash
# Always push to feature branch
git push -u origin claude/optimize-portfolio-numba-JZmnm
```
