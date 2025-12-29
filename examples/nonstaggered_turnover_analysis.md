# Non-Staggered Turnover Analysis

## Problem Statement

When using non-staggered rebalancing (quarterly, semi-annual, annual), the current implementation computes turnover at EVERY month, not just at rebalancing dates. This includes "would-be turnover" between rebalancing dates, even though no actual trading occurs.

## Current Behavior

### What Happens Now

1. **At each month** (including between rebalancing dates):
   - Portfolio weights are computed fresh (1/n for EW)
   - These fresh weights are compared to previous scaled weights
   - Turnover is computed as: `sum(|current_weight - prev_scaled_weight|) / 2`

2. **Problem**: This computes "what turnover WOULD BE if we rebalanced every month", not actual turnover.

### Example (Quarterly Rebalancing)

```
Month 1 (rebal): Form portfolio, weights = [1/n, ..., 1/n], turnover = computed
Month 2 (hold):  Fresh weights = [1/n, ..., 1/n], prev_scaled varies → turnover > 0 !
Month 3 (hold):  Fresh weights = [1/n, ..., 1/n], prev_scaled varies → turnover > 0 !
Month 4 (rebal): Fresh weights = [1/n, ..., 1/n], turnover = computed (correct)
```

Between months 2-3, turnover should be 0 because we're NOT trading - just holding.

## Comparison with Staggered Rebalancing

For staggered rebalancing (HP=3), we correctly set turnover to 0 for holding cohorts:

```python
# In set_zero_for_holding_cohorts():
if cohort != rebalancing_cohort:
    # Cohort is holding, not rebalancing
    state.ew_turn_ea[tau, cohort, k] = 0.0  # Set to 0, not NaN
```

This ensures holding periods have zero turnover, matching the intuition that "no trading = no turnover".

## Weight Scaling Question

User asked: For quarterly rebalancing with returns 1%, 5%, -3%, should the weight evolve as:
```
w_end = 0.20 * 1.01 * 1.05 * 0.97 = 0.2057
```

### Current Implementation

Current code uses **single-period scaling**:
```python
scaled_weight = original_weight * (1 + bond_return) / (1 + portfolio_return)
```

This is computed fresh each month from formation date weights, not cumulatively.

### Analysis

For turnover computation, two approaches are valid:

**Option A: Zero Turnover Between Rebalancing (Simple)**
- Set turnover to 0 between rebalancing dates
- Only compute turnover at actual rebalancing dates
- Matches staggered holding cohort behavior
- Simple, consistent, and what user requested

**Option B: Cumulative Weight Scaling (Complex)**
- Track cumulative weights: `w_t = w_0 * Π(1+r_i) / Π(1+R_p)`
- Compute "drift turnover" from bond dropouts only
- More complex, requires significant code changes

**Recommendation**: Option A - Zero turnover between rebalancing dates.

## Proposed Solution

Add logic to `compute_nonstaggered_turnover_state()` to detect if current date is a rebalancing date. If not, set turnover to 0 (like staggered holding cohorts).

### Implementation

1. Track rebalancing dates in TurnoverState or pass them to `compute()`
2. In `compute_nonstaggered_turnover_state()`:
   ```python
   if not is_rebalancing_date(tau):
       # Between rebalancing dates - no trading
       for k in range(tot_nport):
           state.ew_turn_ea[tau, cohort, k] = 0.0
           state.vw_turn_ea[tau, cohort, k] = 0.0
       return
   ```
3. Only compute actual turnover at rebalancing dates

### API Addition (Optional)

Add parameter to control behavior:
```python
StrategyFormation(
    ...,
    rebalance_frequency='quarterly',
    turnover_between_rebal=False,  # NEW: Default False (zero turnover between)
)
```

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Turnover between rebal | Computed (wrong) | Zero (correct) |
| Turnover at rebal | Computed | Computed |
| Weight scaling | Single-period | Single-period (unchanged) |
| Consistency with staggered | No | Yes |
