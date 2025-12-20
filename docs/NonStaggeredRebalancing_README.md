# Non-Staggered Rebalancing: Complete Guide

This document provides a comprehensive explanation of how PyBondLab handles non-monthly rebalancing (quarterly, semi-annual, annual), including detailed timelines, weight evolution, and the Phase 17 bug fix.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Key Concepts](#key-concepts)
3. [Timeline Deep Dive](#timeline-deep-dive)
   - [Quarterly Rebalancing Example](#quarterly-rebalancing-example)
   - [What Happens at Each Date](#what-happens-at-each-date)
4. [Weight Evolution](#weight-evolution)
   - [At Rebalancing Dates](#at-rebalancing-dates)
   - [Between Rebalancing Dates](#between-rebalancing-dates)
   - [Weight Renormalization When Bonds Drop Out](#weight-renormalization-when-bonds-drop-out)
5. [Turnover Computation](#turnover-computation)
6. [Phase 17 Bug Fix](#phase-17-bug-fix)
7. [Code Architecture](#code-architecture)
8. [Performance](#performance)
9. [Validation](#validation)

---

## Quick Start

```python
import PyBondLab as pbl

# Quarterly rebalancing with returns collected EVERY month
strategy = pbl.SingleSort(
    holding_period=1,           # Ignored for non-staggered
    sort_var='signal',
    num_portfolios=5,
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rebalance_frequency='quarterly',  # Rebalance every 3 months
    rebalance_month=1,                # First rebalance in January
    turnover=True,
)
result = sf.fit()

# Returns are collected for EVERY month (not just at rebalancing dates!)
ew_ls, vw_ls = result.get_long_short()  # All months have data
```

---

## Key Concepts

### Staggered vs Non-Staggered

| Aspect | Monthly (Staggered) | Non-Monthly (Non-Staggered) |
|--------|---------------------|------------------------------|
| **Cohorts** | Multiple overlapping | Single portfolio |
| **Rankings** | Updated every month | Updated only at rebalancing dates |
| **Returns** | Averaged across cohorts | Direct returns (no averaging) |
| **Weights** | Renormalized monthly | Renormalized EVERY month |

### Critical Insight: Returns Are Collected EVERY Month

**This is the most important concept to understand.**

With quarterly rebalancing:
- **Rankings** are computed 4 times per year (at rebalancing dates)
- **Returns** are collected **12 times per year** (every month)
- **Weights** are renormalized **every month** (to handle bond dropouts)

```
                    RANKINGS           RETURNS
                    computed           collected
                    ────────           ─────────
January (rebal):    YES                YES (Feb return)
February:           NO                 YES (Mar return)
March:              NO                 YES (Apr return)
April (rebal):      YES                YES (May return)
May:                NO                 YES (Jun return)
...and so on
```

---

## Timeline Deep Dive

### Quarterly Rebalancing Example

**Parameters:**
- `rebalance_frequency='quarterly'`
- `rebalance_month=1` (January, April, July, October)
- 12 months of data (Jan-Dec)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     QUARTERLY REBALANCING TIMELINE                             │
│                     rebalance_month=1 (Jan/Apr/Jul/Oct)                        │
└────────────────────────────────────────────────────────────────────────────────┘

MONTH:       Jan     Feb     Mar     Apr     May     Jun     Jul     Aug     Sep     Oct     Nov     Dec
             ─────── ─────── ─────── ─────── ─────── ─────── ─────── ─────── ─────── ─────── ─────── ───────
REBALANCE:   [R1]                    [R2]                    [R3]                    [R4]
             │                       │                       │                       │
RANKINGS:    COMPUTE                 COMPUTE                 COMPUTE                 COMPUTE
             (new)                   (new)                   (new)                   (new)

RETURNS:             ──R1──  ──R2──  ──R3──  ──R4──  ──R5──  ──R6──  ──R7──  ──R8──  ──R9──  ──R10─  ──R11─
                     (Feb)   (Mar)   (Apr)   (May)   (Jun)   (Jul)   (Aug)   (Sep)   (Oct)   (Nov)   (Dec)

WEIGHTS:     W_Jan   W_Feb   W_Mar   W_Apr   W_May   W_Jun   W_Jul   W_Aug   W_Sep   W_Oct   W_Nov   W_Dec
             (new)   (renorm)(renorm)(new)   (renorm)(renorm)(new)   (renorm)(renorm)(new)   (renorm)(renorm)

RANK SOURCE: Jan─────────────────────Apr─────────────────────Jul─────────────────────Oct─────────────────
             ranks                   ranks                   ranks                   ranks
             used for                used for                used for                used for
             Feb,Mar,Apr             May,Jun,Jul             Aug,Sep,Oct             Nov,Dec


Legend:
  [Rn]     = Rebalancing date n (rankings computed from signal)
  ──Rn──   = Return n collected
  W_XXX    = Weights computed/renormalized for month XXX
  (new)    = Fresh weights from new rankings
  (renorm) = Weights renormalized from previous month (handles dropouts)
```

### What Happens at Each Date

#### At a REBALANCING Date (e.g., January)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REBALANCING DATE: JANUARY                                │
│                    (Portfolio Formation)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Compute Rankings from Signal
────────────────────────────────────
  • Read signal values for all bonds at January end-of-month
  • Sort bonds by signal value
  • Assign to quintiles (1=low, 2, 3, 4, 5=high)

  Bond A: signal=0.10 → Portfolio 1 (bottom quintile)
  Bond B: signal=0.25 → Portfolio 2
  Bond C: signal=0.50 → Portfolio 3
  Bond D: signal=0.75 → Portfolio 4
  Bond E: signal=0.90 → Portfolio 5 (top quintile)

STEP 2: Check Intersection with February
────────────────────────────────────────
  • Bonds must exist at BOTH January (formation) AND February (return)
  • Bonds must have valid returns in February
  • Bonds must have valid VW at VW date (Jan for dynamic_weights, else formation)

  ✓ Bond A: exists Jan, exists Feb, valid return → INCLUDE
  ✓ Bond B: exists Jan, exists Feb, valid return → INCLUDE
  ✗ Bond C: exists Jan, missing Feb → EXCLUDE
  ✓ Bond D: exists Jan, exists Feb, valid return → INCLUDE
  ✓ Bond E: exists Jan, exists Feb, valid return → INCLUDE

STEP 3: Compute Weights for February Returns
────────────────────────────────────────────
  Portfolio 1: Bond A
    EW weight = 1/1 = 1.0
    VW weight = VW_A / VW_A = 1.0

  Portfolio 3: (empty - Bond C dropped out)
    EW weight = N/A
    VW weight = N/A

  Portfolio 5: Bond E
    EW weight = 1/1 = 1.0
    VW weight = VW_E / VW_E = 1.0

STEP 4: Collect February Returns
────────────────────────────────
  Portfolio 1: Return = Bond A's Feb return × 1.0 = r_A
  Portfolio 5: Return = Bond E's Feb return × 1.0 = r_E
  Long-Short  = Portfolio 5 - Portfolio 1 = r_E - r_A

STEP 5: Compute Turnover (if enabled)
─────────────────────────────────────
  Compare January weights to previous period's scaled weights
  Turnover = Σ|new_weight - old_scaled_weight| / 2
```

#### At a NON-REBALANCING Date (e.g., February)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 NON-REBALANCING DATE: FEBRUARY                              │
│                 (Pseudo-Formation for March Returns)                        │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Use EXISTING Rankings from January
──────────────────────────────────────────
  • DO NOT recompute rankings!
  • Rankings from January rebalancing are still in effect
  • Bond assignments (1-5) stay fixed

  Bond A: Portfolio 1 (unchanged from January)
  Bond B: Portfolio 2 (unchanged from January)
  Bond D: Portfolio 4 (unchanged from January)
  Bond E: Portfolio 5 (unchanged from January)

STEP 2: Check Intersection with March
─────────────────────────────────────
  • Bonds must exist at BOTH February AND March
  • This is the "pseudo-formation" intersection

  ✓ Bond A: exists Feb, exists Mar, valid return → INCLUDE
  ✗ Bond B: exists Feb, missing Mar → EXCLUDE (dropped out!)
  ✓ Bond D: exists Feb, exists Mar, valid return → INCLUDE
  ✓ Bond E: exists Feb, exists Mar, valid return → INCLUDE

STEP 3: RENORMALIZE Weights for March Returns
─────────────────────────────────────────────
  Bond B dropped out! Weights must be renormalized.

  Portfolio 2: (now empty - Bond B dropped out)
    EW weight = N/A
    VW weight = N/A

  Other portfolios unchanged (assuming no other dropouts)

  IMPORTANT: Weights always sum to 1 within each portfolio
  If a bond drops, remaining bonds get proportionally higher weights.

STEP 4: Collect March Returns
─────────────────────────────
  Portfolio 1: Return = Bond A's Mar return × 1.0 = r_A
  Portfolio 2: Return = NaN (no bonds)
  Portfolio 4: Return = Bond D's Mar return × 1.0 = r_D
  Portfolio 5: Return = Bond E's Mar return × 1.0 = r_E
  Long-Short  = Portfolio 5 - Portfolio 1 = r_E - r_A

STEP 5: Compute Turnover (LATENT TURNOVER)
──────────────────────────────────────────
  Even though we didn't rebalance, turnover can still occur!
  Bond B dropped out → its weight went from X to 0
  This is "latent turnover" from weight renormalization

  Turnover > 0 if any bonds dropped out
  Turnover = 0 only if NO bonds dropped out (perfectly balanced panel)
```

---

## Weight Evolution

### At Rebalancing Dates

At a rebalancing date, weights are computed fresh from the new rankings:

```
REBALANCING DATE (e.g., April when rebalance_month=1)
=====================================================

Step 1: New rankings from April signal
Step 2: Intersection with May (return date)
Step 3: Compute fresh weights:

  Equal-Weighted (EW):
  ────────────────────
  weight_i = 1 / n_bonds_in_portfolio

  Example: Portfolio 3 has 20 bonds
  Each bond weight = 1/20 = 0.05

  Value-Weighted (VW):
  ────────────────────
  weight_i = VW_i / Σ(VW_j for j in portfolio)

  Example: Portfolio 3 has bonds with VW = [100, 200, 300]
  Total VW = 600
  Weights = [100/600, 200/600, 300/600] = [0.167, 0.333, 0.500]
```

### Between Rebalancing Dates

Between rebalancing dates, weights are renormalized based on which bonds are still present:

```
NON-REBALANCING DATE (e.g., May, June)
======================================

Rankings stay fixed from April rebalancing
Only weight RENORMALIZATION occurs

If all bonds still present:
  Weights stay exactly the same

If some bonds dropped out:
  Weights are renormalized so they still sum to 1
```

### Weight Renormalization When Bonds Drop Out

```
WEIGHT RENORMALIZATION EXAMPLE
==============================

Formation (April):
  Portfolio 3: Bond A (VW=100), Bond B (VW=200), Bond C (VW=300)
  Total VW = 600
  Weights: A=0.167, B=0.333, C=0.500

May (Bond B drops out):
  Portfolio 3: Bond A (VW=100), Bond C (VW=300)
  Total VW = 400 (B is gone!)
  NEW Weights: A=100/400=0.250, C=300/400=0.750

  ┌────────────────────────────────────────────────────────────┐
  │  Bond A: 0.167 → 0.250  (+0.083)                          │
  │  Bond B: 0.333 → 0.000  (-0.333) ← DROPPED OUT            │
  │  Bond C: 0.500 → 0.750  (+0.250)                          │
  │                                                            │
  │  Weights still sum to 1: 0.250 + 0.750 = 1.000            │
  └────────────────────────────────────────────────────────────┘

June (all remaining bonds present):
  Portfolio 3: Bond A (VW=110), Bond C (VW=290)
  Total VW = 400
  Weights: A=110/400=0.275, C=290/400=0.725

  (Weights changed slightly due to VW changes, but no dropouts)
```

---

## Turnover Computation

### Turnover is Computed at EVERY Date

**Important:** Turnover is computed at EVERY month, not just rebalancing dates!

```
TURNOVER TIMELINE (Quarterly Rebalancing)
=========================================

January [REBAL]:   Turnover = compare new weights to Dec scaled weights
February:          Turnover = compare renorm weights to Jan scaled weights
March:             Turnover = compare renorm weights to Feb scaled weights
April [REBAL]:     Turnover = compare new weights to Mar scaled weights
May:               Turnover = compare renorm weights to Apr scaled weights
June:              Turnover = compare renorm weights to May scaled weights
... and so on
```

### Sources of Turnover

```
TURNOVER SOURCES
================

1. REBALANCING TURNOVER (at rebalancing dates)
   ────────────────────────────────────────────
   New rankings → different bond assignments
   Example: Bond X moves from Portfolio 3 to Portfolio 5

2. LATENT TURNOVER (between rebalancing dates)
   ────────────────────────────────────────────
   Bond dropouts → weight renormalization
   Example: Bond Y drops out, remaining bonds' weights increase

3. ZERO TURNOVER (only if)
   ────────────────────────
   - Perfectly balanced panel (no dropouts)
   - AND no rebalancing at this date
   - This is rare in real data!
```

### Turnover Calculation Formula

```
TURNOVER CALCULATION
====================

At each date t, for each portfolio p:

1. Current weights (raw):
   raw_weight[i] = VW[i] / Σ VW[j]

2. Previous scaled weights (from t-1):
   prev_scaled[i] = prev_weight[i] × (1 + ret[i]) / (1 + portfolio_ret)

3. Current scaled weights (for next period):
   curr_scaled[i] = raw_weight[i] × (1 + ret[i]) / (1 + portfolio_ret)

4. Turnover:
   turnover = Σ |raw_weight[i] - prev_scaled[i]| / 2
```

---

## Phase 17 Bug Fix

### The Bug (Before Phase 17)

Before Phase 17, non-staggered rebalancing had a critical bug:

```
BUG: Returns only collected at rebalancing dates + 1
====================================================

With quarterly rebalancing:

BEFORE (BUG):
  Jan [R1]    Feb ✓    Mar ✗    Apr [R2]    May ✓    Jun ✗    Jul [R3] ...
              return   MISSING              return   MISSING

  Only 4 return dates per year instead of 11!

AFTER (FIXED):
  Jan [R1]    Feb ✓    Mar ✓    Apr [R2]    May ✓    Jun ✓    Jul [R3] ...
              return   return               return   return

  All 11 return dates collected!
```

### Root Cause

The bug was in how the code iterated over return dates:

```python
# BEFORE (BUG):
for h in range(holding_period):  # Only iterated for hp months!
    ret_d = rebal_d + h + 1

# With hp=1 (common setting), only collected 1 return per rebalancing!

# AFTER (FIXED):
for ret_d in range(rebal_d + 1, next_rebal_d + 1):  # Iterate until next rebal
    # Collect returns for ALL months between rebalancing dates
```

### Files Fixed

| File | Function | Fix |
|------|----------|-----|
| `numba_core.py` | `compute_nonstaggered_full_fast()` | Phase 17a |
| `PyBondLab.py` | `_form_nonstaggered_portfolio()` | Phase 17b |
| `numba_core.py` | `compute_nonstaggered_returns_fast()` | Phase 17c |

---

## Code Architecture

### High-Level Flow

```
StrategyFormation.fit()
        │
        ▼
┌───────────────────────────────────────┐
│  Is rebalance_frequency == 'monthly'? │
└───────────────────────────────────────┘
        │                      │
       YES                    NO
        │                      │
        ▼                      ▼
┌─────────────┐      ┌─────────────────────┐
│ _fit()      │      │ _fit_nonstaggered() │
│ (staggered) │      │ (non-staggered)     │
└─────────────┘      └─────────────────────┘
                               │
                               ▼
                     ┌───────────────────────────┐
                     │ Can use fast path?        │
                     │ - turnover=False          │
                     │ - chars=None              │
                     │ - banding=None            │
                     │ - SingleSort only         │
                     └───────────────────────────┘
                          │              │
                         YES            NO
                          │              │
                          ▼              ▼
              ┌──────────────────┐  ┌────────────────────────┐
              │ _fit_fast_       │  │ _form_nonstaggered_    │
              │ nonstaggered()   │  │ portfolio() loop       │
              │ (numba kernels)  │  │ (slow path)            │
              └──────────────────┘  └────────────────────────┘
```

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `_get_rebalancing_dates()` | `utils_optimized.py` | Get indices of rebalancing dates |
| `_fit_nonstaggered()` | `PyBondLab.py` | Main entry for non-staggered |
| `_form_nonstaggered_portfolio()` | `PyBondLab.py` | Slow path: per-rebalancing logic |
| `compute_nonstaggered_full_fast()` | `numba_core.py` | Fast path with turnover/chars |
| `compute_nonstaggered_returns_fast()` | `numba_core.py` | Fast path returns only |

---

## Portfolio Returns and Weights: Complete Equations

### Weight Computation

At each return date, weights are computed from **formation date VW** but normalized for
bonds present at the return date:

```
For portfolio p at return date t+1:

Equal-Weight (EW):
  w_ew[i] = 1 / N_p(t+1)

  where N_p(t+1) = number of bonds in portfolio p present at t+1

Value-Weight (VW):
  w_vw[i] = VW_t[i] / Σⱼ∈p VW_t[j]

  where:
    - VW_t[i] = value weight of bond i at FORMATION date t
    - Σⱼ∈p = sum over bonds j in portfolio p that are PRESENT at t+1
```

**Key insight**: VW values come from formation date, but normalization uses only bonds
present at the return date. This is the "renormalization" that happens on pseudo-rebalancing dates.

### Weight Renormalization Between Rebalancing Dates

On **TRUE rebalancing dates**, weights are computed fresh:

```
At TRUE rebalancing (e.g., January for quarterly):

  w_raw[i] = VW_Jan[i] / Σⱼ∈p VW_Jan[j]

  where:
    - VW_Jan[i] = value weight of bond i at January
    - Σⱼ∈p = sum over ALL bonds j assigned to portfolio p at January
```

On **PSEUDO-rebalancing dates** (months between true rebalancing), weights are
renormalized using the SAME VW values but a DIFFERENT bond universe:

```
At PSEUDO-rebalancing (e.g., March, after January formation):

  w_raw[i](Mar) = VW_Jan[i] / Σⱼ∈p∩M VW_Jan[j]

  where:
    - VW_Jan[i] = value weight of bond i at January (UNCHANGED)
    - p∩M = bonds in portfolio p that are PRESENT at March
    - Σⱼ∈p∩M = sum only over bonds still available

  If bond k dropped out between Feb and Mar:
    - Bond k is excluded from the sum
    - Remaining bonds' weights increase proportionally
```

**Example of renormalization**:

```
January (TRUE rebalancing):
  Portfolio 1 bonds: A, B, C
  VW: A=100, B=60, C=40 → Total=200
  Weights: w_A=0.50, w_B=0.30, w_C=0.20

February (bonds A, B, C all present):
  Same VW values from Jan, same bonds present
  Weights: w_A=0.50, w_B=0.30, w_C=0.20 (unchanged)

March (bond B drops out):
  VW from Jan: A=100, C=40 → New Total=140
  Renormalized: w_A=100/140=0.714, w_C=40/140=0.286
  (B's weight redistributed proportionally)
```

### Portfolio Return Computation

Returns are computed using **raw weights (w_raw)**, not scaled weights:

```
Portfolio Return (EW):
  R_ew,p(t+1) = (1/N_p) × Σᵢ∈p r_i(t+1)
             = simple average of bond returns

Portfolio Return (VW):
  R_vw,p(t+1) = Σᵢ∈p w_vw[i] × r_i(t+1)
             = Σᵢ∈p [VW_t[i] / Σⱼ∈p VW_t[j]] × r_i(t+1)

  where r_i(t+1) = return of bond i from t to t+1
```

### Scaled Weights (for Turnover)

After computing returns, we compute **scaled weights** for the NEXT period's turnover:

```
Scaled Weight:
  w_scaled[i] = w_raw[i] × (1 + r_i) / (1 + R_p)

  where:
    - w_raw[i] = current raw weight of bond i
    - r_i = bond i's return this period
    - R_p = portfolio p's return this period

Interpretation:
  - If bond i outperforms the portfolio: w_scaled[i] > w_raw[i]
  - If bond i underperforms: w_scaled[i] < w_raw[i]
  - Sum of scaled weights ≈ 1 (exactly 1 if no rounding)
```

---

## Turnover Pipeline

### Code Reuse

Non-staggered rebalancing **reuses the same turnover code** as staggered (monthly) rebalancing.
The key difference is that turnover is computed at **every date** (not just rebalancing dates).

### Slow Path Pipeline

```
_fit_nonstaggered()
        │
        ├── Initialize TurnoverManager (same class as staggered)
        │   self.turnover_manager = TurnoverManager(...)
        │
        └── For each rebalancing date:
              └── _form_nonstaggered_portfolio()
                    └── For each month until next rebalancing:
                          ├── _form_single_period() → computes w_raw, returns
                          ├── compute_scaled_weights_single() → computes w_scaled
                          └── turnover_manager.compute() → computes turnover
```

### Fast Path Pipeline (Numba)

```
_fit_nonstaggered_fast()
        │
        └── compute_nonstaggered_full_fast()  ← Single numba kernel
              │
              └── For each date d:
                    ├── Compute w_raw (current weights)
                    ├── Compute returns using w_raw
                    ├── Compute turnover: compare w_raw vs prev_scaled
                    ├── Compute w_scaled for next period
                    └── Store w_scaled as prev_scaled for d+1
```

### Turnover Computation Timeline

```
TURNOVER CALCULATION FLOW
=========================

At return date t+1, we have:

  ┌─────────────────────────────────────────────────────────────────┐
  │                     PREVIOUS PERIOD (t)                         │
  │                                                                 │
  │  w_raw(t) ──► Returns r_i(t) ──► w_scaled(t)                   │
  │                                      │                          │
  │                                      │ stored as prev_scaled    │
  │                                      ▼                          │
  └──────────────────────────────────────┼──────────────────────────┘
                                         │
  ┌──────────────────────────────────────┼──────────────────────────┐
  │                     CURRENT PERIOD (t+1)                        │
  │                                      │                          │
  │                                      ▼                          │
  │  w_raw(t+1) ◄─── VW from formation, renormalized for t+1       │
  │      │                               │                          │
  │      │                               │                          │
  │      ▼                               ▼                          │
  │  ┌───────────────────────────────────────────┐                  │
  │  │  TURNOVER = |w_raw(t+1) - w_scaled(t)|    │                  │
  │  │           = trading needed to rebalance   │                  │
  │  └───────────────────────────────────────────┘                  │
  │      │                                                          │
  │      ▼                                                          │
  │  Returns R_p(t+1) computed using w_raw(t+1)                     │
  │      │                                                          │
  │      ▼                                                          │
  │  w_scaled(t+1) = w_raw(t+1) × (1+r_i)/(1+R_p) ──► stored       │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### Turnover Formula

```
Turnover at t+1 for portfolio p:

  T_p(t+1) = ½ × Σᵢ |w_raw[i](t+1) - w_scaled[i](t)|

Efficient computation (avoiding per-bond iteration):

  T_p(t+1) = ½ × (prev_sum + curr_sum - 2 × sum_min)

  where:
    - prev_sum = Σᵢ w_scaled[i](t)     (previous scaled weights sum)
    - curr_sum = Σᵢ w_raw[i](t+1)      (current raw weights sum, ≈1)
    - sum_min = Σᵢ min(w_scaled[i](t), w_raw[i](t+1))
```

### Complete Timeline Example

```
QUARTERLY REBALANCING: RETURNS, WEIGHTS, AND TURNOVER
======================================================

                 JANUARY        FEBRUARY         MARCH           APRIL
                 (t=0)          (t=1)            (t=2)           (t=3)
                 TRUE REBAL     pseudo           pseudo          TRUE REBAL
─────────────────────────────────────────────────────────────────────────────

1. WEIGHTS:
   VW Source:    VW_Jan         VW_Jan           VW_Jan          VW_Apr
   Bonds used:   at Jan         at Feb           at Mar          at Apr
   Formula:      VW_Jan[i]/Σ    VW_Jan[i]/Σ'     VW_Jan[i]/Σ''   VW_Apr[i]/Σ
                 (Σ=bonds@Jan)  (Σ'=bonds@Feb)   (Σ''=bonds@Mar)
                     │              │                │               │
                     ▼              ▼                ▼               ▼
2. RAW WEIGHTS: w_raw(0)       w_raw(1)         w_raw(2)        w_raw(3)
                     │              │                │               │
                     │              │                │               │
3. RETURNS:         N/A        R(1)=Σw·r        R(2)=Σw·r       R(3)=Σw·r
   (computed        first      using w_raw(1)   using w_raw(2)  using w_raw(3)
    with w_raw)     period          │                │               │
                                    │                │               │
4. SCALED:     w_scaled(0)     w_scaled(1)      w_scaled(2)     w_scaled(3)
               =w_raw(0)×      =w_raw(1)×       =w_raw(2)×      =w_raw(3)×
               (1+r)/(1+R)     (1+r)/(1+R)      (1+r)/(1+R)     (1+r)/(1+R)
                    │               │                │               │
                    │               │                │               │
                    └───────────────┤                │               │
                                    ▼                │               │
5. TURNOVER:       N/A         T(1)=compare     T(2)=compare    T(3)=compare
                   first       w_raw(1) vs      w_raw(2) vs     w_raw(3) vs
                   period      w_scaled(0)      w_scaled(1)     w_scaled(2)
                                    │                │               │
                                    ▼                ▼               ▼
6. OUTPUT          N/A         Index: Feb       Index: Mar      Index: Apr
   (all indexed                Return: R(1)     Return: R(2)    Return: R(3)
    at return                  Turnover: T(1)   Turnover: T(2)  Turnover: T(3)
    date t+1)                  Chars: from Jan  Chars: from Jan Chars: from Apr
```

### What Turnover Measures

| At Return Date | Turnover Measures | Caused By |
|----------------|-------------------|-----------|
| Feb (after Jan rebal) | First period | N/A (no previous) |
| Mar (pseudo-rebal) | Feb→Mar change | Bonds dropping out, weight renormalization |
| Apr (TRUE rebal) | Mar→Apr change | New rankings + bonds dropping out |

**Latent turnover** on pseudo-rebalancing dates: Even without re-ranking, if bonds drop out,
weights must be renormalized → this creates turnover from the "drift" in portfolio composition.

---

## Filters: `rating` and `subset_filter`

### How Filters Are Applied

Filters are applied at **formation date only** (no look-ahead bias):

| Filter Type | When Applied | Effect |
|-------------|--------------|--------|
| `rating` | Formation date | Exclude bonds outside rating range from ranking |
| `subset_filter` | Formation date | Exclude bonds outside characteristic range from ranking |

### Slow Path (StrategyFormation)

```python
# In _precompute_data() → filter_by_rating() and filter_by_char()
# Filters applied during precomputation
if self.rating is not None:
    sub = sub[(sub['RATING_NUM'] >= min_r) & (sub['RATING_NUM'] <= max_r)]

if self.subset_filter is not None:
    for col, (min_val, max_val) in self.subset_filter.items():
        sub = sub[(sub[col] >= min_val) & (sub[col] <= max_val)]
```

### Fast Path (BatchStrategyFormation)

```python
# In _fit_fast_batch_nonstaggered()
# Build filter mask (True = passes filter)
filter_mask = np.ones(len(data), dtype=np.bool_)

if self.rating is not None:
    if self.rating == 'IG':
        filter_mask &= (rating_vals <= 10)
    elif self.rating == 'NIG':
        filter_mask &= (rating_vals > 10)
    elif isinstance(self.rating, tuple):
        min_r, max_r = self.rating
        filter_mask &= (rating_vals >= min_r) & (rating_vals <= max_r)

if self.subset_filter is not None:
    for col, (min_val, max_val) in self.subset_filter.items():
        filter_mask &= (col_vals >= min_val) & (col_vals <= max_val)

# Set signal to NaN for filtered observations → excluded from ranking
signals_matrix[~filter_mask] = np.nan
```

### No Look-Ahead Bias

Filters are applied at formation, but returns are collected regardless of filter status:

```
Formation (Jan): Bond rated IG → included in portfolio ranking
Return (Feb):    Bond downgraded to NIG → return STILL collected!
                 (We don't use future rating information)
```

### Example Usage

```python
import PyBondLab as pbl

# With rating filter
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(
        holding_period=1,
        sort_var='signal',
        num_portfolios=5,
        rebalance_frequency='quarterly',
    ),
    rating='IG',  # Only investment grade bonds
    # OR rating=(1, 10)  # Explicit rating range
)

# With subset_filter
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(..., rebalance_frequency='quarterly'),
    subset_filter={
        'MATURITY': (1, 5),     # Maturity 1-5 years
        'DURATION': (2, 8),    # Duration 2-8 years
    },
)

# BatchStrategyFormation also supports these
batch = pbl.BatchStrategyFormation(
    data=data,
    signals=['signal1', 'signal2'],
    holding_period=1,
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rating='IG',
    subset_filter={'MATURITY': (1, 5)},
)
```

---

## Date Indexing

### Critical Concept: Return Date vs Formation Date

All outputs are indexed by **RETURN date** (t+1), not formation date (t):

| Output | Index Date | Computed From |
|--------|------------|---------------|
| **Returns** | Return date (t+1) | w_raw × r at t+1 |
| **Turnover** | Return date (t+1) | w_raw(t+1) vs w_scaled(t) |
| **Weights** | Return date (t+1) | VW from formation, renormalized for t+1 |
| **Characteristics** | Return date (t+1) | Char values from formation date |

### Why Return Date Indexing?

This is logically consistent:

```
At index t+1:
  ┌────────────────────────────────────────────────────────────┐
  │  Return[t+1]   = return earned holding the t+1 portfolio   │
  │  Turnover[t+1] = trading done to GET TO the t+1 portfolio  │
  │  Weights[t+1]  = the weights used to compute Return[t+1]   │
  │  Chars[t+1]    = characteristics of what we're holding     │
  └────────────────────────────────────────────────────────────┘

Everything describes the SAME portfolio period.
```

### Timeline with Indexing

```
Quarterly Rebalancing (rebalance_month=1):

Event                    Index in Output
─────────────────────────────────────────
Formation at Jan         (no output)
Return Jan→Feb          Index = Feb
Return Feb→Mar          Index = Mar
Return Mar→Apr          Index = Apr
Formation at Apr         (no output)
Return Apr→May          Index = May
...

Output DataFrames:
  Date (Index)  │ Return  │ Turnover │ Weights Source  │ Char Source
  ──────────────┼─────────┼──────────┼─────────────────┼─────────────
  2024-02-01    │ R(Feb)  │ T(Feb)   │ VW from Jan     │ Chars from Jan
  2024-03-01    │ R(Mar)  │ T(Mar)   │ VW from Jan*    │ Chars from Jan
  2024-04-01    │ R(Apr)  │ T(Apr)   │ VW from Jan*    │ Chars from Jan
  2024-05-01    │ R(May)  │ T(May)   │ VW from Apr     │ Chars from Apr

  * Renormalized for bonds present at that date
```

### Characteristics: Special Case

Characteristics are indexed by return date but **values come from formation date**:

```python
# At return date Feb (index=1), characteristics come from formation date Jan
# This is because characteristics describe the portfolio as formed

# For non-staggered with dynamic_weights=False (always):
char_date = formation_date  # NOT d-1
```

---

## Performance

### Speedup Achieved

| Configuration | Slow Path | Fast Path | Speedup |
|---------------|-----------|-----------|---------|
| Returns only | 3.69s | 0.020s | **182x** |
| With turnover | 4.5s | 0.044s | **103x** |
| With chars | 3.8s | 0.067s | **57x** |
| All features | 5.2s | 0.060s | **86x** |

### When Fast Path is Used

Fast path is automatically enabled when:
- `rebalance_frequency != 'monthly'`
- `SingleSort` only (not DoubleSort)

For DoubleSort, the slow path is used but still benefits from Phase 17 fixes.

---

## Validation

### Test Script

```bash
python examples/validate_nonstaggered_bug.py
```

### Expected Results

With 12 months of data and quarterly rebalancing:
- **Rebalancing dates**: 4 (Jan, Apr, Jul, Oct)
- **Return dates**: 11 (Feb through Dec)
- **All 12 baseline tests**: PASS

### Quick Validation

```python
import PyBondLab as pbl
from PyBondLab.pbl_test import generate_synthetic_data

data = generate_synthetic_data(n_dates=12, n_bonds=100, seed=42)

sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5),
    rebalance_frequency='quarterly',
    rebalance_month=1,
    verbose=False,
)
result = sf.fit()

ew_ls, _ = result.get_long_short()
n_dates = ew_ls.notna().sum()
print(f"Return dates: {n_dates}")  # Should be 11
assert n_dates == 11, f"Expected 11, got {n_dates}"
```

---

## Summary

### Key Takeaways

1. **Returns are collected EVERY month**, not just at rebalancing dates
2. **Rankings are computed only at rebalancing dates**, then reused
3. **Weights are renormalized every month** to handle bond dropouts
4. **Turnover can occur between rebalancing dates** (latent turnover from dropouts)
5. **Phase 17 fixed a critical bug** where returns were only collected at rebalancing dates

### API Reference

```python
pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rebalance_frequency='quarterly',  # 'quarterly', 'semi-annual', 'annual', or int
    rebalance_month=1,                # Which month(s) to rebalance (1-12)
    turnover=True,                    # Compute turnover at EVERY date
)
```

| `rebalance_frequency` | Rebalancing Months (if `rebalance_month=1`) |
|----------------------|---------------------------------------------|
| `'quarterly'` or `3` | Jan, Apr, Jul, Oct |
| `'semi-annual'` or `6` | Jan, Jul |
| `'annual'` or `12` | Jan only |

---

## Important: `dynamic_weights` Does NOT Apply to Non-Staggered

### Key Design Principle

For non-staggered (non-monthly) rebalancing, the `dynamic_weights` parameter **does not apply**.
VW always comes from the **formation date** (true rebalancing date), regardless of the
`dynamic_weights` setting.

This ensures:
1. Consistent behavior across `SingleSort` and `BatchStrategyFormation`
2. Weights are determined at portfolio formation, then renormalized if bonds drop out
3. Results match exactly between different APIs

### How Weights Work for Non-Staggered

**On TRUE rebalancing dates (e.g., Jan for quarterly):**
- Portfolio is formed based on signal
- VW comes from formation date (that month)
- Weights computed as: `VW_i / sum(VW_j for j in portfolio)`

**On PSEUDO-rebalancing dates (e.g., Feb, Mar for quarterly):**
- Portfolio composition stays fixed from true rebalancing
- Check which bonds are still available
- Weights **renormalized** for remaining bonds using formation-date VW values

```
Example: Quarterly rebalancing, formation in January

January (TRUE rebalancing):
  Portfolio 1: Bond A (40%), Bond B (30%), Bond C (30%)
  VW values from January

February (PSEUDO-rebalancing, Bond B drops out):
  Portfolio 1: Bond A (40%/70% = 57.1%), Bond C (30%/70% = 42.9%)
  VW values still from January, but renormalized for remaining bonds

March (PSEUDO-rebalancing, all bonds present again):
  Portfolio 1: Bond A (40%), Bond B (30%), Bond C (30%)
  VW values from January (original weights restored)
```

### Validation

Both `SingleSort` and `BatchStrategyFormation` now produce **identical results** for all
non-staggered rebalancing configurations:

```python
# These produce identical results:
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(
        holding_period=1,
        sort_var='signal',
        num_portfolios=5,
        rebalance_frequency='quarterly',
    ),
    verbose=False,
)

batch = pbl.BatchStrategyFormation(
    data=data,
    signals=['signal'],
    holding_period=1,
    num_portfolios=5,
    rebalance_frequency='quarterly',
)
```

---

## Appendix: Complete Timeline Example

```
12-MONTH QUARTERLY REBALANCING (rebalance_month=1)
==================================================

DATE        TYPE            RANKING     RETURN      WEIGHT        TURNOVER
                            SOURCE      COLLECTED   ACTION        SOURCE
────────────────────────────────────────────────────────────────────────────
January     REBALANCING     COMPUTE     -           Fresh         Compare to Dec
February    Non-rebal       Use Jan     Feb ret     Renormalize   Latent (dropouts)
March       Non-rebal       Use Jan     Mar ret     Renormalize   Latent (dropouts)
April       REBALANCING     COMPUTE     Apr ret     Fresh         Compare to Mar
May         Non-rebal       Use Apr     May ret     Renormalize   Latent (dropouts)
June        Non-rebal       Use Apr     Jun ret     Renormalize   Latent (dropouts)
July        REBALANCING     COMPUTE     Jul ret     Fresh         Compare to Jun
August      Non-rebal       Use Jul     Aug ret     Renormalize   Latent (dropouts)
September   Non-rebal       Use Jul     Sep ret     Renormalize   Latent (dropouts)
October     REBALANCING     COMPUTE     Oct ret     Fresh         Compare to Sep
November    Non-rebal       Use Oct     Nov ret     Renormalize   Latent (dropouts)
December    Non-rebal       Use Oct     Dec ret     Renormalize   Latent (dropouts)
────────────────────────────────────────────────────────────────────────────
TOTALS:     4 rebalancing   4 ranking   11 return   12 weight     12 turnover
            dates           updates     dates       updates       observations
```
