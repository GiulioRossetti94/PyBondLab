# Staggered Rebalancing Turnover Computation

## Overview

This document explains how turnover is computed for staggered (overlapping cohort) portfolio rebalancing in PyBondLab, including a critical bug fix implemented in December 2025.

---

## Table of Contents

1. [Staggered Rebalancing Basics](#staggered-rebalancing-basics)
2. [Turnover Formula](#turnover-formula)
3. [Cohort-Level vs Aggregate Turnover](#cohort-level-vs-aggregate-turnover)
4. [The Bug: Horizon Loop Overwrites](#the-bug-horizon-loop-overwrites)
5. [The Fix: Only Accumulate for h=0](#the-fix-only-accumulate-for-h0)
6. [Key Assumptions](#key-assumptions)
7. [Turnover and Characteristics Alignment (shift(1))](#turnover-and-characteristics-alignment-shift1)
8. [Detailed Examples](#detailed-examples)
9. [Discussion: Bond Dropouts](#discussion-bond-dropouts)
10. [Code Reference](#code-reference)

---

## Staggered Rebalancing Basics

### What is Staggered Rebalancing?

For holding period HP > 1, PyBondLab uses **staggered rebalancing** where multiple overlapping cohorts exist simultaneously. Each cohort:
- Is formed at a different time
- Holds for HP months before rebalancing
- Contributes 1/HP of the final portfolio return

### Timeline Diagram (HP=3)

```
Month:    0     1     2     3     4     5     6     7     8     9
          |     |     |     |     |     |     |     |     |     |
Cohort 0: [F]---H-----H----[R]---H-----H----[R]---H-----H----[R]
Cohort 1:       [F]---H-----H----[R]---H-----H----[R]---H-----H
Cohort 2:             [F]---H-----H----[R]---H-----H----[R]---H

Legend:
  [F] = Formation (portfolio created, first time)
  [R] = Rebalancing (portfolio reformed)
  H   = Holding (no trading, portfolio unchanged)
```

### Cohort Assignment

At each formation date `t`, the cohort is determined by:
```python
cohort = t % HP
```

For HP=3:
- t=0: cohort 0 (formation)
- t=1: cohort 1 (formation)
- t=2: cohort 2 (formation)
- t=3: cohort 0 (rebalancing!)
- t=4: cohort 1 (rebalancing!)
- t=5: cohort 2 (rebalancing!)
- ...

---

## Turnover Formula

### Per-Portfolio Turnover

For a single portfolio at rebalancing:

```
Turnover = sum(|current_weight_i - previous_weight_i|) / 2
```

This is implemented as:
```python
turnover = prev_sum + curr_sum - 2 * sum_min
```

Where:
- `prev_sum` = sum of previous (scaled) weights = ~1.0
- `curr_sum` = sum of current weights = 1.0
- `sum_min` = sum of min(prev_weight_i, curr_weight_i) for each bond

### Turnover Bounds

| Scenario | sum_min | Turnover |
|----------|---------|----------|
| No change (identical portfolios) | 1.0 | 0.0 |
| Complete change (no overlap) | 0.0 | 2.0 |
| Partial change | 0 < x < 1 | 0 < t < 2 |

### Scaled Weights

Previous weights are **scaled** to account for returns during the holding period:

```python
scaled_weight_i = (1 + bond_return_i) / (1 + portfolio_return) * original_weight_i
```

This ensures the comparison is "fair" - if a bond appreciated, its weight increased even without trading.

---

## Cohort-Level vs Aggregate Turnover

### At Each Formation Date

At formation date `t`:
1. **Rebalancing cohort** (cohort = t % HP): Computes actual turnover
2. **Holding cohorts**: Set turnover to **0.0** (no trading)
3. **Non-existent cohorts**: Remain as **NaN** (not yet formed)

### Aggregate Turnover

The final reported turnover is the **average across cohorts**:

```python
aggregate_turnover = nanmean([cohort_0_turnover, cohort_1_turnover, cohort_2_turnover])
```

### Example (HP=3, Complete Turnover)

At t=3 (cohort 0 rebalances):
```
Cohort 0: 2.0  (complete turnover)
Cohort 1: 0.0  (holding)
Cohort 2: 0.0  (holding)

Aggregate = (2.0 + 0.0 + 0.0) / 3 = 0.67
```

This makes intuitive sense: only 1/3 of the portfolio is trading at any given time.

---

## The Bug: Horizon Loop Overwrites

### The Problem

In the original code, turnover accumulation was **inside the horizon loop**:

```python
def _form_cohort_portfolios(self, t_idx, ...):
    for h in range(self.hor):  # h = 0, 1, 2 for HP=3
        t1_idx = t_idx + h + 1  # Return date

        result = self._form_single_period(...)

        # BUG: Called for EVERY h!
        if self.turnover and not result['weights_df'].empty:
            self.turnover_manager.accumulate(
                self.turnover_state,
                self.cohort,
                tot_nport,
                t_idx,  # Same tau for all h!
                result['weights_df'],
                result['weights_scaled_df']
            )
```

### Why This Was Wrong

For HP=3 at formation date t=0:

| Iteration | h | Return Date | What Happens |
|-----------|---|-------------|--------------|
| 1st | 0 | t=1 | First time: `prev_seen=False` → No turnover, set `prev_seen=True` |
| 2nd | 1 | t=2 | `prev_seen=True` now! → Computes turnover (h=1 vs h=0 weights) |
| 3rd | 2 | t=3 | `prev_seen=True` → Computes turnover (h=2 vs h=1 weights), **overwrites** |

The **final stored value** compared h=2 weights with h=1 weights - which are nearly identical since they're from the **same formation month**!

### Visual Diagram of the Bug

```
Formation Date t=0, HP=3:
=========================

Horizon Loop:
  h=0: weights_df for return at t=1
       └─> accumulate(tau=0, cohort=0)
           └─> prev_seen[0] = False
               └─> No turnover (first time)
               └─> Set prev_seen[0] = True
               └─> Store scaled weights

  h=1: weights_df for return at t=2 (slightly different intersection)
       └─> accumulate(tau=0, cohort=0)
           └─> prev_seen[0] = True now!
               └─> Compute turnover: h=1 weights vs h=0 scaled weights
               └─> turnover ≈ 0.01 (nearly identical!)
               └─> Store at ew_turn_ea[0, 0, :] ← OVERWRITES

  h=2: weights_df for return at t=3 (slightly different intersection)
       └─> accumulate(tau=0, cohort=0)
           └─> prev_seen[0] = True
               └─> Compute turnover: h=2 weights vs h=1 scaled weights
               └─> turnover ≈ 0.01 (nearly identical!)
               └─> Store at ew_turn_ea[0, 0, :] ← OVERWRITES AGAIN

Final stored value: ~0.01 (comparing same-month weights)
Should have been: NaN (first formation, no previous portfolio)
```

### Impact at Actual Rebalancing

The bug persisted at actual rebalancing dates (e.g., t=3 for cohort 0):

```
Rebalancing Date t=3, Cohort 0:
===============================

  h=0: Should compute actual turnover (~2.0 for complete change)
       └─> Compares t=3 weights with t=0 scaled weights
       └─> turnover = 2.0 (correct!)

  h=1: OVERWRITES with small value
       └─> Compares t=3,h=1 weights with t=3,h=0 scaled weights
       └─> turnover ≈ 0.01 (same month!)

  h=2: OVERWRITES again
       └─> Compares t=3,h=2 weights with t=3,h=1 scaled weights
       └─> turnover ≈ 0.01 (same month!)

Final stored value: ~0.01 (wrong!)
Should have been: ~2.0 (actual rebalancing turnover)
```

### Measured Impact

| HP | Expected Turnover | Actual (Bug) | Ratio |
|----|-------------------|--------------|-------|
| 1 | 2.0 | 1.69 | 0.85 |
| 2 | 1.0 | **0.05** | **0.05** |
| 3 | 0.67 | **0.07** | **0.10** |

HP > 1 was **10-20x too low**!

---

## The Fix: Only Accumulate for h=0

### The Solution

Add `h == 0` condition to only accumulate turnover for the first horizon:

```python
def _form_cohort_portfolios(self, t_idx, ...):
    for h in range(self.hor):
        t1_idx = t_idx + h + 1

        result = self._form_single_period(...)

        # FIX: Only accumulate for h=0
        if self.turnover and h == 0 and not result['weights_df'].empty:
            self.turnover_manager.accumulate(...)
```

### Why This Works

| Formation Date | h=0 Behavior |
|----------------|--------------|
| First formation (t=0) | `prev_seen=False` → No turnover, set `prev_seen=True` |
| Rebalancing (t=HP) | `prev_seen=True` → Compute actual turnover vs previous formation |

The key insight: **turnover measures change between consecutive formation dates for the same cohort**, not between horizons within a single formation.

### Corrected Results

| HP | Expected | After Fix | Ratio |
|----|----------|-----------|-------|
| 1 | 2.0 | 1.69 | 0.85 |
| 2 | 1.0 | 0.77 | 0.77 |
| 3 | 0.67 | 0.57 | 0.85 |

All HPs now show **consistent ratios** (~0.77-0.85), matching expectations given random noise and edge effects.

---

## Key Assumptions

### Assumption 1: Holding Cohorts Have Zero Turnover

**Statement**: For cohorts that are NOT rebalancing at time t, we set turnover = 0.0.

**Rationale**: A holding cohort does not trade. By definition:
- Portfolio composition stays fixed (same bonds, same target weights)
- No buy/sell orders are executed
- Turnover = 0 by the trading definition

**Code**:
```python
def set_zero_for_holding_cohorts(self, state, rebalancing_cohort, tot_nport, tau):
    for cohort in range(hor):
        if cohort == rebalancing_cohort:
            continue  # Skip rebalancing cohort
        if cohort > tau:
            continue  # Cohort doesn't exist yet (NaN)

        # Set to 0.0 for holding cohorts
        for k in range(tot_nport):
            if state.prev_seen_ew[cohort, k]:
                state.ew_turn_ea[tau, cohort, k] = 0.0
```

**Implication**: This assumes a **perfectly balanced panel** where no bonds drop out. See [Discussion: Bond Dropouts](#discussion-bond-dropouts) below.

### Assumption 2: Scaled Weights Reflect Price Changes

**Statement**: Between rebalancing dates, weights are scaled by returns to reflect market price changes.

**Rationale**: If you hold a portfolio without trading:
- Bonds that appreciate have higher dollar values
- Their portfolio weight naturally increases
- Comparison with "fresh" weights should account for this drift

**Formula**:
```
scaled_weight_i = (1 + r_i) / (1 + r_portfolio) * original_weight_i
```

### Assumption 3: First Formation Has No Turnover

**Statement**: The first time a cohort is formed (e.g., cohort 0 at t=0), there's no turnover because there's no previous portfolio to compare with.

**Implementation**: Tracked via `prev_seen` flags:
- Initially `False` for all cohorts
- Set to `True` after first formation
- Turnover only computed when `prev_seen=True`

---

## Turnover and Characteristics Alignment (shift(1))

### The Alignment Problem

Before the shift(1) fix, there was a subtle indexing mismatch:

| DataFrame | Index Represents | Value Represents |
|-----------|------------------|------------------|
| Returns | Return date t | Return earned from portfolio formed at t-1 |
| Turnover (old) | Date t | Cost incurred at formation t-1 |

For net-of-cost calculations, users needed to manually align:
```python
# Old (confusing): Manual lag required
net_return[t] = gross_return[t] - cost * turnover[t-1]  # Awkward!
```

### The Fix: shift(1)

After applying `.shift(1)` to turnover and characteristics:

| DataFrame | Index Represents | Value Represents |
|-----------|------------------|------------------|
| Returns | Return date t | Return from portfolio formed at t-1 |
| Turnover | Return date t | Cost to enter portfolio that generates return[t] |
| Chars | Return date t | Characteristics of portfolio that generates return[t] |

Now net-of-cost is straightforward:
```python
# New (intuitive): Direct alignment
net_return[t] = gross_return[t] - cost * turnover[t]
```

### First Row is NaN (Warmup Period)

After shift(1), the first row of turnover and characteristics is NaN:

```
Returns:     [ret_jan, ret_feb, ret_mar, ...]
Turnover:    [NaN,     turn_feb, turn_mar, ...]
Chars:       [NaN,     char_feb, char_mar, ...]
```

**Why NaN?**
- At the first return date (Jan), there's no previous formation to compare
- No turnover was incurred because no prior portfolio existed
- Characteristics of "no portfolio" is undefined

**Important**: For HP>1 staggered rebalancing, the first NaN applies to ALL cohorts. The warmup period represents when the overlapping cohort structure is still being initialized.

### Net-of-Cost Calculation Example

```python
from PyBondLab import StrategyFormation, SingleSort

# Run strategy with turnover
result = StrategyFormation(
    data=data,
    strategy=SingleSort(holding_period=1, sort_var='signal', num_portfolios=5),
    turnover=True,
).fit()

# Get aligned data
ew_ls, vw_ls = result.get_long_short()
ew_turn, vw_turn = result.get_turnover()

# Compute factor-level turnover: average of long and short portfolios
n_port = ew_turn.shape[1]
factor_turnover = (ew_turn.iloc[:, 0] + ew_turn.iloc[:, n_port - 1]) / 2

# Set transaction cost (e.g., 20 bps per unit turnover)
cost_per_unit = 0.002

# Net-of-cost returns (aligned dates only)
common_dates = ew_ls.index.intersection(factor_turnover.index)
net_returns = ew_ls.loc[common_dates] - cost_per_unit * factor_turnover.loc[common_dates]

# Note: First common date will have NaN turnover → NaN net return
# Usable net returns start from second date onwards
```

### Same Logic for Characteristics

Characteristics are also shifted by 1, so:
- `chars[t]` = characteristics of the portfolio that generates `return[t]`
- First row is NaN (warmup period)

This enables direct factor exposure analysis:
```python
ew_chars, vw_chars = result.get_characteristics()
duration_chars = ew_chars['duration']

# Duration[t] = duration exposure of portfolio generating return[t]
# Can directly regress: return[t] ~ duration[t] + ...
```

### Verification Script

```bash
python examples/validate_shift_alignment.py
```

This validates:
- First turnover row is NaN
- First chars row is NaN
- BatchStrategyFormation produces consistent alignment
- Net-of-cost calculation works correctly

---

## Detailed Examples

### Example 1: Complete Turnover (Balanced Panel, HP=3)

**Setup**:
- 50 bonds, all present at all dates
- Signal reverses every 3 months (so each cohort sees different signal at rebalancing)
- 5 quintile portfolios

**Timeline**:
```
Month 0: Cohort 0 forms (ascending signal: bond 0 → P1, bond 49 → P5)
Month 1: Cohort 1 forms (ascending signal)
Month 2: Cohort 2 forms (ascending signal)
Month 3: Cohort 0 REBALANCES (descending signal: bond 0 → P5, bond 49 → P1)
         Complete portfolio reversal → turnover = 2.0
```

**Raw Turnover Array at t=3**:
```
t=3:
  cohort 0: [2.0, 2.0, 0.01, 2.0, 2.0]  # Rebalancing (complete turnover)
  cohort 1: [0.0, 0.0, 0.0, 0.0, 0.0]   # Holding
  cohort 2: [0.0, 0.0, 0.0, 0.0, 0.0]   # Holding
```

Note: Middle portfolio (P3) has ~0.01 turnover because "middle" bonds don't change much.

**Aggregate Turnover**:
```
average = (2.0 + 0.0 + 0.0) / 3 = 0.67
```

### Example 2: No Turnover (Signal Unchanged)

**Setup**:
- Same signal every month (no reversal)
- Same bonds in same portfolios at each rebalancing

**Raw Turnover Array at t=3**:
```
t=3:
  cohort 0: [0.02, 0.02, 0.02, 0.02, 0.02]  # Tiny (only from weight scaling)
  cohort 1: [0.0, 0.0, 0.0, 0.0, 0.0]
  cohort 2: [0.0, 0.0, 0.0, 0.0, 0.0]
```

The ~0.02 is from scaled weight drift due to returns, not actual portfolio change.

### Example 3: Partial Turnover

**Setup**:
- 20% of bonds change portfolios at each rebalancing
- 80% stay in same portfolio

**Expected**:
- 20% of weight exits old portfolio: 0.20
- 20% of weight enters new portfolio: 0.20
- turnover = (0.20 + 0.20) / 2 = 0.20 per portfolio (approximate)

**With HP=3**:
```
aggregate_turnover ≈ 0.20 / 3 ≈ 0.07
```

---

## Discussion: Bond Dropouts

### The Question

> Is it "fair" to force holding cohort turnover to 0 when bonds can drop out of the sample?

### Current Behavior

When a bond drops out of the sample (e.g., matures, defaults, or is removed):
1. It's no longer in `weights_df` at the next return date
2. The remaining bonds' weights are **renormalized** to sum to 1
3. But for **holding cohorts**, we still set turnover = 0

### The Problem

If bond X drops out of Portfolio 1:
- Original weights: A=0.25, B=0.25, X=0.25, C=0.25
- After dropout: A=0.33, B=0.33, C=0.33 (renormalized)

This is a **latent weight change** even without active trading. The portfolio's risk exposures have changed.

### Two Perspectives

**Perspective A: Trading Turnover (Current Implementation)**

> "Turnover measures actual trading activity. Holding cohorts don't trade, so turnover = 0."

- Correct for transaction cost estimation
- Consistent with the definition: turnover = trades / portfolio value
- If you don't execute any trades, turnover is 0 regardless of what happens to the portfolio

**Perspective B: Weight Change Turnover**

> "Turnover measures change in portfolio composition. Bond dropouts change composition."

- Better for risk exposure analysis
- A dropout forces implicit reallocation (remaining bonds get more weight)
- May be relevant for style drift, factor exposure changes

### Recommendation

The current implementation (Perspective A) is appropriate for:
- Transaction cost analysis
- Trading activity measurement
- Comparison with industry-standard turnover definitions

For risk/exposure analysis, users may want to compute "weight drift" separately:
```python
weight_drift = sum(|current_weight - previous_weight|) / 2
# Includes passive drift from dropouts, not just active trading
```

### Future Enhancement Possibility

Add an option to track "passive turnover" from dropouts:
```python
TurnoverManager(..., include_dropout_turnover=False)  # Current behavior
TurnoverManager(..., include_dropout_turnover=True)   # Track latent changes
```

---

## Code Reference

### Key Files

| File | Function | Description |
|------|----------|-------------|
| `PyBondLab/PyBondLab.py` | `_form_cohort_portfolios()` | Main loop with `h == 0` fix |
| `PyBondLab/utils_turnover.py` | `accumulate_turnover()` | Core turnover computation |
| `PyBondLab/utils_turnover.py` | `set_zero_for_holding_cohorts()` | Sets 0 for non-rebalancing cohorts |
| `PyBondLab/utils_turnover.py` | `finalize_turnover()` | Averages across cohorts |
| `PyBondLab/numba_core.py` | `compute_turnover_all_portfolios()` | Numba-optimized computation |

### The Fix (PyBondLab.py:2144-2167)

```python
# Handle turnover if requested
# CRITICAL: Only accumulate turnover for h=0 (first horizon)
# For HP>1, multiple horizons share the same formation date (t_idx).
# If we accumulate for all h, later horizons (h=1,2,...) overwrite
# h=0's correct turnover with small values (since h=1 vs h=0 weights
# are nearly identical - same formation month). The fix is to only
# compute turnover once per formation, at h=0.
if self.turnover and h == 0 and not result['weights_df'].empty:
    self.turnover_manager.accumulate(
        self.turnover_state,
        self.cohort,
        tot_nport,
        t_idx,
        result['weights_df'],
        result['weights_scaled_df']
    )
    self.turnover_manager.set_zero_for_holding_cohorts(
        self.turnover_state,
        self.cohort,
        tot_nport,
        t_idx
    )
```

### Verification Script

```bash
python examples/diagnose_turnover.py
```

This creates a balanced panel with maximally-changing signal and verifies:
- HP=1 turnover ≈ 2.0
- HP=2 turnover ≈ 1.0
- HP=3 turnover ≈ 0.67

---

## Changelog

| Date | Change |
|------|--------|
| Dec 2025 | Added shift(1) alignment for turnover and characteristics |
| Dec 2025 | First row of turnover/chars now NaN (warmup period) |
| Dec 2025 | Fixed horizon loop overwrite bug (added `h == 0` condition) |
| Dec 2025 | Created this documentation |

---

## See Also

- [CLAUDE.md](../CLAUDE.md) - Project documentation
- [examples/diagnose_turnover.py](../examples/diagnose_turnover.py) - Diagnostic script for turnover computation
- [examples/validate_shift_alignment.py](../examples/validate_shift_alignment.py) - Validation script for shift(1) alignment
- [examples/diagnose_turnover_unbalanced.py](../examples/diagnose_turnover_unbalanced.py) - Diagnostic for unbalanced panels
