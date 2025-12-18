# Non-Staggered Rebalancing: Technical Deep Dive

This document provides a comprehensive explanation of how PyBondLab handles non-monthly rebalancing (quarterly, semi-annual, annual), including timelines, data flow, and optimization opportunities.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences from Monthly Rebalancing](#key-differences-from-monthly-rebalancing)
3. [Timeline Diagrams](#timeline-diagrams)
   - [Annual Rebalancing](#annual-rebalancing)
   - [Semi-Annual Rebalancing](#semi-annual-rebalancing)
   - [Quarterly Rebalancing](#quarterly-rebalancing)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Current Implementation](#current-implementation)
   - [Rebalancing Date Selection](#rebalancing-date-selection)
   - [Portfolio Formation Loop](#portfolio-formation-loop)
   - [Return Computation](#return-computation)
6. [Turnover Computation](#turnover-computation)
7. [Performance Analysis](#performance-analysis)
8. [Optimization Plan (Phase 15)](#optimization-plan-phase-15)
9. [Validation](#validation)

---

## Overview

PyBondLab supports two rebalancing modes:

| Mode | `rebalance_frequency` | Behavior |
|------|----------------------|----------|
| **Staggered (Monthly)** | `'monthly'` | Multiple overlapping cohorts; returns averaged across cohorts |
| **Non-Staggered** | `'quarterly'`, `'semi-annual'`, `'annual'`, or `int` | Single portfolio held for full period; no cohort averaging |

### Key Insight: Simplicity of Non-Staggered

Non-staggered rebalancing is **much simpler** than staggered:

```
Monthly (Staggered, HP=3):
  - 3 overlapping cohorts, each formed at different dates
  - Complex averaging across cohorts at each return date
  - Turnover requires tracking state for each cohort

Annual (Non-Staggered, HP=12):
  - Single portfolio formed once per year
  - Hold for 12 months, collect 12 returns
  - Turnover is straightforward (compare this year to last year)
```

This simplicity makes non-staggered rebalancing an excellent target for numba optimization.

---

## Key Differences from Monthly Rebalancing

| Aspect | Monthly (Staggered) | Non-Monthly (Non-Staggered) |
|--------|---------------------|------------------------------|
| **Cohorts** | `holding_period` overlapping cohorts | Single cohort |
| **Returns** | Averaged across cohorts | Direct returns |
| **Formation dates** | Every month | Only at specified months |
| **Turnover** | Track state per cohort | Single state |
| **Complexity** | O(n_dates × holding_period) | O(n_rebal_dates × holding_period) |

---

## Timeline Diagrams

### Annual Rebalancing

**Parameters:** `rebalance_frequency='annual'`, `rebalance_month=6`, `holding_period=12`

```
Timeline for Annual Rebalancing (June)
======================================

Year 1:
  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
  ───  ───  ───  ───  ───  [F1] ─R1─ ─R2─ ─R3─ ─R4─ ─R5─ ─R6─
                           │
                           └── Portfolio FORMED using signal at June

Year 2:
  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
  ─R7─ ─R8─ ─R9─ R10─ R11─ R12─ [F2] ─R1─ ─R2─ ─R3─ ─R4─ ─R5─
  └───────────────────────┘     │
  Returns from Year 1 formation  │
                                 └── NEW portfolio formed

Legend:
  [Fn] = Formation date n (portfolio created using signal)
  ─Rn─ = Return collected for month n of holding period
```

**Key Points:**
- Portfolio formed in **June** using signals at June end-of-month
- Returns collected for **July through next June** (12 months)
- First return date is July (month after formation)
- Last return date is next June (completes 12-month hold)

### Semi-Annual Rebalancing

**Parameters:** `rebalance_frequency='semi-annual'`, `rebalance_month=6`, `holding_period=6`

```
Timeline for Semi-Annual Rebalancing (June/December)
====================================================

Year 1:
  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
  ───  ───  ───  ───  ───  [F1] ─R1─ ─R2─ ─R3─ ─R4─ ─R5─ ─R6─
                           │                                 │
                           └── Form using June signal        └── [F2] Form using Dec signal

Year 2:
  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
  ─R1─ ─R2─ ─R3─ ─R4─ ─R5─ ─R6─ ─R1─ ─R2─ ─R3─ ─R4─ ─R5─ ─R6─
  └───────────────────────┘     └───────────────────────────┘
  Returns from Dec formation     Returns from June formation
                           │
                           └── [F3] Form using June signal

Legend:
  [Fn] = Formation date n
  ─Rn─ = Return for month n of holding period
```

**Key Points:**
- Portfolios formed in **June** and **December**
- Each portfolio held for **6 months**
- Returns: July-December (June formation), January-June (December formation)
- Non-overlapping: each month has returns from only one formation

### Quarterly Rebalancing

**Parameters:** `rebalance_frequency='quarterly'`, `rebalance_month=6`, `holding_period=3`

```
Timeline for Quarterly Rebalancing (Mar/Jun/Sep/Dec)
====================================================

Year 1:
  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
  ───  ───  [F1] ─R1─ ─R2─ ─R3─ [F2] ─R1─ ─R2─ ─R3─ [F3] ─R1─
            │    │    │    │   │                        │
            │    └────┴────┘   └── Form with Jun signal │
            │                                           └── Form with Sep signal
            └── Form with Mar signal, returns Apr-Jun

Year 2:
  Jan  Feb  Mar  Apr  May  Jun ...
  ─R2─ ─R3─ [F4] ─R1─ ─R2─ ─R3─
  │    │    │
  └────┘    └── Form with Mar signal
  Returns from Dec formation

Note: rebalance_month=6 generates rebalancing at [3, 6, 9, 12]
      (i.e., months 6, 6+3=9, 6+6=12, 6+9=3)
```

**Key Points:**
- Portfolios formed **every 3 months**
- Each portfolio held for **3 months**
- No overlap between portfolios
- More return observations than annual/semi-annual

---

## Data Flow Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          StrategyFormation.fit()                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │  is_staggered = (rebalance_frequency == 'monthly')
                    └─────────────────────────────────┘
                           │                    │
                          YES                  NO
                           │                    │
                           ▼                    ▼
              ┌──────────────────┐   ┌──────────────────────┐
              │ _fit_staggered() │   │ _fit_nonstaggered()  │
              │ (Monthly rebal)  │   │ (Quarterly/Annual)   │
              └──────────────────┘   └──────────────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │ _get_rebalancing_dates()       │
                              │ Returns indices of rebal dates │
                              └────────────────────────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │ for rebal_idx in rebal_dates:  │
                              │   _form_nonstaggered_portfolio │
                              └────────────────────────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │ for h in range(holding_period):│
                              │   _form_single_period()        │
                              └────────────────────────────────┘
```

### Detailed Flow for Non-Staggered

```
_fit_nonstaggered()
│
├── 1. Get rebalancing date indices
│       rebal_dates_idx = _get_rebalancing_dates(datelist, freq, month)
│       Example: [5, 17, 29, ...] for June dates
│
├── 2. Precompute data
│       precomp = _precompute_data()
│       → ranks_map: {date: {ID: rank}}
│       → vw_map_t0: {date: {ID: VW}}
│       → It0, It1, It1m: DataFrames by date
│
├── 3. Initialize result arrays
│       ew_ret_arr = np.full((TM, nport), np.nan)
│       vw_ret_arr = np.full((TM, nport), np.nan)
│
├── 4. Main loop over rebalancing dates
│       for rebal_idx in rebal_dates_idx:
│           _form_nonstaggered_portfolio(rebal_idx, precomp, ...)
│
│           # Inner loop over holding period
│           for h in range(holding_period):
│               t1_idx = rebal_idx + h + 1  # Return date index
│
│               # Get formation data (signal date)
│               It0 = precomp.It0[date_t]
│
│               # Get return data (t+1 date)
│               It1 = precomp.It1[date_t1]
│
│               # Form portfolio and compute returns
│               result = _form_single_period(It0, It1, ...)
│
│               # Store at return date (NOT formation date)
│               ew_ret_arr[t1_idx, :] = result['returns_ew']
│               vw_ret_arr[t1_idx, :] = result['returns_vw']
│
└── 5. Aggregate results
        _aggregate_results_nonstaggered()
```

---

## Current Implementation

### Rebalancing Date Selection

The `_get_rebalancing_dates()` function determines which dates trigger rebalancing:

```python
def _get_rebalancing_dates(datelist, rebal_freq, rebal_month):
    """
    Returns list of indices in datelist where rebalancing occurs.
    """
    if rebal_freq == 'monthly':
        return list(range(len(datelist)))  # All dates

    elif rebal_freq == 'quarterly':
        # Compute 4 months: [rebal_month, rebal_month+3, rebal_month+6, rebal_month+9]
        rebal_months = [(rebal_month + 3*i - 1) % 12 + 1 for i in range(4)]
        return [i for i, date in enumerate(datelist) if date.month in rebal_months]

    elif rebal_freq == 'semi-annual':
        # 2 months: [rebal_month, rebal_month+6]
        rebal_months = [rebal_month, (rebal_month + 6 - 1) % 12 + 1]
        return [i for i, date in enumerate(datelist) if date.month in rebal_months]

    elif rebal_freq == 'annual':
        # Single month
        return [i for i, date in enumerate(datelist) if date.month == rebal_month]
```

**Example Output (10 years of data):**

| Frequency | `rebal_month` | Number of Rebal Dates |
|-----------|--------------|----------------------|
| quarterly | 6 | 40 (4 per year × 10 years) |
| semi-annual | 6 | 20 (2 per year × 10 years) |
| annual | 6 | 10 (1 per year × 10 years) |

### Portfolio Formation Loop

```python
def _form_nonstaggered_portfolio(self, rebal_idx, precomp, ...):
    """
    Form portfolio at rebal_idx and collect returns for holding_period months.
    """
    date_t = self.datelist[rebal_idx]  # Formation date

    for h in range(self.hor):  # holding_period iterations
        t1_idx = rebal_idx + h + 1  # Return date index

        if t1_idx >= len(self.datelist):
            break  # No more return dates

        date_t1 = self.datelist[t1_idx]  # Return date

        # Get formation data (ranks from date_t)
        It0 = precomp.It0.get(date_t, pd.DataFrame())

        # Get return data (from date_t1)
        It1 = precomp.It1.get(date_t1, pd.DataFrame())

        # Form portfolio
        result = self._form_single_period(It0, It1, ...)

        # Store returns at t1_idx (return date, not formation date)
        ew_ret_arr[t1_idx, :] = result['returns_ew']
        vw_ret_arr[t1_idx, :] = result['returns_vw']
```

### Return Computation

For each (formation_date, return_date) pair:

```python
def _form_single_period(self, It0, It1, ...):
    """
    Compute portfolio returns for bonds ranked at formation date.
    """
    # 1. Intersect IDs (bonds must exist at both dates)
    It0, It1, It1m = intersect_id(It0, It1, It1m, self.dynamic_weights)

    # 2. Map ranks from formation date to return data
    It1['ptf_rank'] = It1['ID'].map(ranks_map[date_t])
    It1 = It1.dropna(subset=['ptf_rank'])

    # 3. Get value weights
    if self.dynamic_weights:
        # VW from t1-1 (day before return)
        It1['VW'] = It1['ID'].map(vw_map_t1m[date_t1_minus1])
    else:
        # VW from formation date
        It1['VW'] = It1['ID'].map(vw_map_t0[date_t])

    # 4. Compute portfolio returns using numba kernels
    ew_ret, vw_ret = compute_portfolio_returns_single(ranks, returns, weights, nport)

    return {'returns_ew': ew_ret, 'returns_vw': vw_ret, ...}
```

---

## Turnover Computation

### Non-Staggered Turnover

For non-staggered rebalancing, turnover is simpler than staggered:

```
Annual Rebalancing Turnover:
============================

Year 1 Portfolio: Formed June 2020
  Bonds: [A, B, C, D, E] with weights [0.1, 0.2, 0.3, 0.2, 0.2]

Year 2 Portfolio: Formed June 2021
  Bonds: [B, C, D, F, G] with weights [0.15, 0.25, 0.2, 0.2, 0.2]

Turnover = Sum of |weight_new - weight_old| / 2

  Bond A: |0 - 0.1| = 0.1  (sold)
  Bond B: |0.15 - 0.2| = 0.05
  Bond C: |0.25 - 0.3| = 0.05
  Bond D: |0.2 - 0.2| = 0.0
  Bond E: |0 - 0.2| = 0.2  (sold)
  Bond F: |0.2 - 0| = 0.2  (bought)
  Bond G: |0.2 - 0| = 0.2  (bought)

Total = (0.1 + 0.05 + 0.05 + 0 + 0.2 + 0.2 + 0.2) / 2 = 0.4
```

**Observed turnover from validation:**

| Frequency | Avg EW Turnover |
|-----------|-----------------|
| annual | 0.39 |
| semi-annual | 0.51 |
| quarterly | 0.74 |

Higher frequency = more turnover (expected).

---

## Performance Analysis

### Current Performance (Slow Path)

From validation script (120 dates, 300 bonds):

| Configuration | Time (s) | Notes |
|---------------|----------|-------|
| Annual, no turnover | 0.69 | Baseline |
| Annual, with turnover | 1.80 | +160% overhead |
| Annual, with chars | 1.40 | +100% overhead |
| Annual, with banding | 0.97 | +40% overhead |
| Semi-annual, no turnover | 0.66 | Similar to annual |
| Quarterly, no turnover | 0.80 | Slightly slower |
| Quarterly, turnover+chars | 1.82 | Full features |

### Performance Bottlenecks

1. **Pandas precomputation** (`_precompute_data`): ~40% of time
   - Creates DataFrames for each date
   - Computes ranks using groupby

2. **Per-date loops**: ~30% of time
   - Python loop over dates
   - DataFrame slicing and merging

3. **ID intersection**: ~15% of time
   - Pandas merge operations

4. **Turnover computation**: ~15% of time (when enabled)
   - Weight scaling and differencing

---

## Optimization Plan (Phase 15)

### Goal

Achieve **5-10x speedup** for non-staggered rebalancing while maintaining exact numerical compatibility.

### Phase 15a: Fast Path (No Turnover/Chars/Banding)

When `turnover=False`, `chars=None`, `banding=None`:

```python
def _fit_nonstaggered_fast(self):
    """
    Ultra-fast non-staggered rebalancing using numba.

    Strategy:
    1. Convert DataFrame to numpy arrays ONCE
    2. Compute ranks for ALL rebalancing dates in parallel
    3. Compute returns for ALL (rebal_date, return_date) pairs in parallel
    """
    # Step 1: Extract numpy arrays
    date_idx = data['date'].map(date_to_idx).values
    id_idx = data['ID'].map(id_to_idx).values
    signal = data['signal'].values
    ret = data['ret'].values
    vw = data['VW'].values

    # Step 2: Identify rebalancing dates
    rebal_dates_idx = _get_rebalancing_dates(datelist, freq, month)

    # Step 3: Compute ranks at each rebalancing date (parallel)
    # Shape: (n_rebal_dates, n_bonds)
    ranks_all = compute_ranks_at_dates(date_idx, signal, rebal_dates_idx, nport)

    # Step 4: Build rank lookup table
    # rank_lookups[rebal_idx, bond_idx] = portfolio rank (or 0 if not ranked)
    rank_lookups = build_rank_lookups(date_idx, id_idx, ranks_all, ...)

    # Step 5: Compute returns for all (rebal, return) pairs in parallel
    ew_ret, vw_ret = compute_nonstaggered_returns(
        date_idx, id_idx, ret, vw,
        rebal_dates_idx, holding_period,
        rank_lookups, nport
    )

    return ew_ret, vw_ret  # Shape: (n_dates, nport)
```

### Phase 15b: Full Path (With Turnover/Chars/Banding)

Optimize the full feature set:

```python
def _fit_nonstaggered_full(self):
    """
    Fast non-staggered rebalancing with turnover, chars, and banding.

    Strategy:
    1. Batch portfolio formation for ALL rebalancing dates
    2. Parallel weight computation
    3. Batch turnover computation
    4. Batch characteristics aggregation
    """
    # Pre-extract all arrays
    ...

    # Batch compute ranks for all rebal dates
    ranks_all = compute_ranks_at_dates_batch(...)

    # For each rebal date, compute weights once
    # Then apply to all holding period months
    for rebal_idx in rebal_dates_idx:
        # Compute weights at formation date
        ew_weights, vw_weights = compute_weights(ranks_all[rebal_idx], vw)

        # Store scaled weights for turnover
        scaled_ew, scaled_vw = compute_scaled_weights(...)

        # Apply to all holding months
        for h in range(holding_period):
            t1_idx = rebal_idx + h + 1
            ew_ret_arr[t1_idx] = compute_returns_from_weights(...)
            vw_ret_arr[t1_idx] = compute_returns_from_weights(...)

        # Compute turnover (compare to previous rebal)
        if prev_weights is not None:
            turnover_arr[rebal_idx] = compute_turnover(prev_weights, current_weights)
```

### Expected Performance Gains

| Configuration | Current | Target | Speedup |
|---------------|---------|--------|---------|
| Annual, no turnover | 0.69s | 0.10s | **7x** |
| Annual, with turnover | 1.80s | 0.30s | **6x** |
| Quarterly, turnover+chars | 1.82s | 0.35s | **5x** |

### Integration with BatchStrategyFormation

Once optimized, the fast non-staggered path will be available in:

```python
batch = BatchStrategyFormation(
    data=data,
    signals=['sig1', 'sig2', ...],
    holding_period=12,
    num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=6,
    turnover=False,  # Enables ultra-fast path
)
results = batch.fit()
```

### Integration with DataUncertaintyAnalysis

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[12],
    rebalance_frequency='annual',  # NEW parameter
    rebalance_month=6,             # NEW parameter
    filters={'trim': [0.2]},
).fit()
```

---

## Validation

### Test Matrix

| `rebalance_frequency` | `rebalance_month` | `holding_period` | Expected Rebal Dates (10yr) |
|----------------------|-------------------|------------------|----------------------------|
| 3 (quarterly) | 1 | 3 | 40 |
| 3 (quarterly) | 6 | 3 | 40 |
| 3 (quarterly) | 12 | 3 | 40 |
| 6 (semi-annual) | 1 | 6 | 20 |
| 6 (semi-annual) | 6 | 6 | 20 |
| 6 (semi-annual) | 12 | 6 | 20 |
| 12 (annual) | 1 | 12 | 10 |
| 12 (annual) | 6 | 12 | 10 |
| 12 (annual) | 12 | 12 | 10 |

### Validation Script

```bash
python examples/validate_nonstaggered_rebalancing.py
```

### Key Validation Checks

1. **Correct rebalancing dates**: Verify formation happens at expected months
2. **Correct return dates**: Returns stored at t+1 through t+hp
3. **No overlap**: Each return date has contribution from exactly one formation
4. **Turnover consistency**: Lower frequency = lower turnover
5. **Numerical match**: Optimized path matches slow path within machine epsilon

---

## Summary

Non-staggered rebalancing is a simpler case than monthly (staggered) rebalancing:

| Aspect | Complexity |
|--------|------------|
| Cohort management | None (single portfolio) |
| Return averaging | None (direct returns) |
| Turnover tracking | Single state (not per-cohort) |
| Parallelization potential | High (all rebal dates independent) |

This simplicity makes it an excellent target for numba optimization, with expected 5-10x speedup.
