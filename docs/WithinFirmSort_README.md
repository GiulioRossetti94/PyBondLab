# WithinFirmSort Strategy

## Overview

`WithinFirmSort` implements a within-firm high-low sorting methodology for constructing bond factors. Unlike standard cross-sectional sorting (SingleSort), this strategy sorts bonds **within each firm**, isolating within-firm bond dispersion from cross-firm differences.

---

## Table of Contents

1. [Economic Intuition](#economic-intuition)
2. [Quick Start](#quick-start)
3. [Key Differences from SingleSort](#key-differences-from-singlesort)
4. [Factor Construction: Step-by-Step](#factor-construction-step-by-step)
5. [Timeline and Indexing](#timeline-and-indexing)
6. [Full API Reference](#full-api-reference)
7. [Custom Column Names](#custom-column-names)
8. [Fast Path vs Slow Path](#fast-path-vs-slow-path)
9. [Feature Support](#feature-support)
10. [Performance](#performance)
11. [Examples](#examples)
12. [Architecture](#architecture)
13. [Validation](#validation)

---

## Economic Intuition

### The Problem with Cross-Sectional Sorting

Standard cross-sectional sorting (SingleSort) ranks ALL bonds in the universe by a characteristic
(e.g., credit spread). This approach conflates two distinct sources of variation:

1. **Cross-firm variation**: Differences between firms (e.g., Apple vs. a distressed retailer)
2. **Within-firm variation**: Differences between bonds issued by the SAME firm

The cross-firm variation often dominates, making it difficult to identify whether the factor
premium is driven by the characteristic of interest or by unobserved issuer-specific factors.

### The Within-Firm Solution

`WithinFirmSort` addresses this by constructing factors that **control for issuer-specific shocks
unrelated to the bond characteristic of interest**. This serves as a pseudo-control for firm-level
fixed effects.

**Key Insight**: By sorting within each firm, we compare bonds that share the same:
- Issuer credit quality (same firm = same default risk)
- Management and operational risk
- Industry exposure
- Macroeconomic sensitivity

The only systematic difference is the bond characteristic we're sorting on (e.g., maturity,
credit spread, liquidity).

### Practical Example: Credit Spread Factor

Consider two approaches to constructing a credit spread factor:

**Cross-Sectional (SingleSort):**
```
Long:  High-spread bonds (distressed retailers, energy companies)
Short: Low-spread bonds (Apple, Microsoft, Johnson & Johnson)

Problem: Are we capturing the "credit spread" premium, or just the fact that
         distressed firms have higher spreads AND higher expected returns?
```

**Within-Firm (WithinFirmSort):**
```
For Apple:     Long Apple's high-spread bonds, Short Apple's low-spread bonds
For Microsoft: Long Microsoft's high-spread bonds, Short Microsoft's low-spread bonds
... (repeat for each firm)

Aggregate across firms and rating groups.

Result: Factor captures ONLY within-firm spread variation.
        Firm-level shocks cancel out in the long-short portfolio.
```

### When to Use WithinFirmSort

| Use Case | Recommended Strategy |
|----------|---------------------|
| General factor construction | SingleSort |
| Testing if a characteristic predicts returns **after controlling for issuer** | **WithinFirmSort** |
| Constructing "pure" maturity, liquidity, or duration factors | **WithinFirmSort** |
| When concerned about omitted firm-level variables | **WithinFirmSort** |
| Maximum sample size / power | SingleSort |

### Academic References

The within-firm sorting methodology is motivated by the corporate bond literature's concern with
firm-level confounds. Key references:

- Bai, Bali, and Wen (2019): "Common risk factors in the cross-section of corporate bond returns"
- Chordia et al. (2017): "Liquidity and credit risk in corporate bonds"
- Kelly, Palhares, and Pruitt (2020): "Factor investing in the cross-section of bonds"

---

## Quick Start

```python
import PyBondLab as pbl

# Initialize WithinFirmSort strategy
strategy = pbl.WithinFirmSort(
    holding_period=1,              # Monthly rebalancing
    sort_var='CS',                 # Sort on credit spread
    firm_id_col='PERMNO',          # Firm identifier column
    min_bonds_per_firm=2,          # Require at least 2 bonds per firm
)

# Run strategy formation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    verbose=True
).fit()

# Get long-short returns
ew_ls, vw_ls = result.get_long_short()
print(f"Mean return: {vw_ls.mean()*100:.3f}% per month")
```

---

## Key Differences from SingleSort

| Aspect | SingleSort | WithinFirmSort |
|--------|-----------|----------------|
| **Grouping** | None (cross-sectional) | Date × Rating Tercile × Firm |
| **Percentiles** | Global (e.g., 20/40/60/80 for quintiles) | Within-firm (33.3/66.7) |
| **Portfolios** | N portfolios (typically 5) | 2 (HIGH/LOW only) |
| **Return Aggregation** | Simple VW average across bonds | Firm-cap-weighted → Rating-averaged |
| **Economic Interpretation** | Cross-sectional factor | Within-issuer factor (firm FE control) |

---

## Factor Construction: Step-by-Step

This section provides a detailed walkthrough of how WithinFirmSort factors are constructed.

### Step 1: Rating Tercile Assignment

Bonds are first grouped into rating terciles based on their credit rating at the **formation date**:

| Tercile | Rating Range | Description |
|---------|-------------|-------------|
| **Tercile 1 (IG+)** | Ratings 1-7 | AAA to A- |
| **Tercile 2 (IG-)** | Ratings 8-10 | BBB+ to BBB- |
| **Tercile 3 (SG)** | Ratings 11+ | BB+ and below |

Custom bins can be specified via `rating_bins` parameter (e.g., `[-inf, 4, 10, inf]`).

```
Example at date t (formation date):
===================================

Bond Universe (10 bonds, 3 firms):

Firm A (IG+, Tercile 1):
  Bond A1: Rating=3 (AA),  Credit Spread=150bp
  Bond A2: Rating=5 (A+),  Credit Spread=180bp
  Bond A3: Rating=6 (A),   Credit Spread=220bp

Firm B (IG-, Tercile 2):
  Bond B1: Rating=8 (BBB+), Credit Spread=280bp
  Bond B2: Rating=9 (BBB),  Credit Spread=320bp
  Bond B3: Rating=9 (BBB),  Credit Spread=350bp

Firm C (SG, Tercile 3):
  Bond C1: Rating=12 (BB),  Credit Spread=520bp
  Bond C2: Rating=13 (BB-), Credit Spread=580bp
  Bond C3: Rating=14 (B+),  Credit Spread=650bp
  Bond C4: Rating=15 (B),   Credit Spread=720bp
```

### Step 2: Within-Firm Portfolio Formation

For each (date, rating tercile, firm) group:

1. **Check minimum bonds**: Require `min_bonds_per_firm` bonds (default: 2)
2. **Compute percentile thresholds** of the signal within the firm:
   - Low threshold: 33.3rd percentile
   - High threshold: 66.7th percentile
3. **Assign bonds to portfolios**:
   - Signal < 33.3rd percentile → **LOW portfolio**
   - Signal > 66.7th percentile → **HIGH portfolio**
   - Middle tercile → **Excluded** (not used)

```
Continuing the example - Portfolio Assignment:
==============================================

Firm A (3 bonds, sort by credit spread):
  Thresholds: p33=165bp, p67=200bp

  Bond A1 (150bp) < 165bp  → LOW  (low spread = safer)
  Bond A2 (180bp) in middle → EXCLUDED
  Bond A3 (220bp) > 200bp  → HIGH (high spread = riskier)

Firm B (3 bonds):
  Thresholds: p33=300bp, p67=335bp

  Bond B1 (280bp) < 300bp  → LOW
  Bond B2 (320bp) in middle → EXCLUDED
  Bond B3 (350bp) > 335bp  → HIGH

Firm C (4 bonds):
  Thresholds: p33=567bp, p67=685bp

  Bond C1 (520bp) < 567bp  → LOW
  Bond C2 (580bp) in middle → EXCLUDED
  Bond C3 (650bp) in middle → EXCLUDED
  Bond C4 (720bp) > 685bp  → HIGH
```

### Step 3: Collect Returns at Return Date

At the return date (t+1), we collect returns for bonds that were assigned to portfolios
at the formation date (t).

```
Return Collection at date t+1:
==============================

Formation date t: Portfolios assigned based on credit spread
Return date t+1:  Collect returns for assigned bonds

Firm A:
  LOW:  Bond A1 return = 0.8%
  HIGH: Bond A3 return = 1.2%

Firm B:
  LOW:  Bond B1 return = 0.9%
  HIGH: Bond B3 return = 1.5%

Firm C:
  LOW:  Bond C1 return = 1.0%
  HIGH: Bond C4 return = 2.1%
```

### Step 4: Hierarchical Return Aggregation

Returns are aggregated in a **hierarchical** manner to construct the final factor:

```
Aggregation Hierarchy:
======================

Level 1: Within-Firm Aggregation
--------------------------------
For each firm, compute VW returns for HIGH and LOW:

  Firm A (VW by market value):
    r_HIGH = VW_return(A3) = 1.20%
    r_LOW  = VW_return(A1) = 0.80%
    H-L_A  = 1.20% - 0.80% = 0.40%
    Cap_A  = $50B (total firm market value)

  Firm B:
    r_HIGH = 1.50%, r_LOW = 0.90%
    H-L_B  = 0.60%
    Cap_B  = $30B

  Firm C:
    r_HIGH = 2.10%, r_LOW = 1.00%
    H-L_C  = 1.10%
    Cap_C  = $20B

Level 2: Across-Firm Aggregation (Cap-Weighted)
-----------------------------------------------
Within each rating tercile, aggregate firm H-L factors using cap-weighting:

  Tercile 1 (IG+): Only Firm A
    Factor_T1 = H-L_A = 0.40%

  Tercile 2 (IG-): Only Firm B
    Factor_T2 = H-L_B = 0.60%

  Tercile 3 (SG): Only Firm C
    Factor_T3 = H-L_C = 1.10%

  (If multiple firms in a tercile:
   Factor_Tk = Σ(Cap_i × H-L_i) / Σ(Cap_i))

Level 3: Across-Rating Aggregation (Simple Average)
----------------------------------------------------
Average across rating terciles:

  Final Factor = (Factor_T1 + Factor_T2 + Factor_T3) / 3
               = (0.40% + 0.60% + 1.10%) / 3
               = 0.70%
```

### Aggregation Equations

For EW portfolios:

$$r_{EW,ptf,firm} = \frac{1}{N_{bonds}} \sum_{i \in ptf} r_i$$

$$Factor_{EW,tercile} = \frac{1}{N_{firms}} \sum_{f} (r_{EW,HIGH,f} - r_{EW,LOW,f})$$

$$Factor_{EW} = \frac{1}{3} \sum_{k=1}^{3} Factor_{EW,tercile_k}$$

For VW portfolios:

$$r_{VW,ptf,firm} = \frac{\sum_{i \in ptf} w_i \cdot r_i}{\sum_{i \in ptf} w_i}$$

$$Factor_{VW,tercile} = \frac{\sum_{f} Cap_f \cdot (r_{VW,HIGH,f} - r_{VW,LOW,f})}{\sum_{f} Cap_f}$$

$$Factor_{VW} = \frac{1}{3} \sum_{k=1}^{3} Factor_{VW,tercile_k}$$

---

## Timeline and Indexing

### Timeline Diagram

All outputs (returns, turnover, characteristics) are indexed by the **return date**.

```
WithinFirmSort Timeline (HP=1):
===============================

Formation Date (t)              Return Date (t+1)
      │                               │
      ▼                               ▼
 ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │ • Read signal values        │ │ • Collect bond returns      │
 │ • Read rating for tercile   │ │ • Aggregate returns         │
 │ • Read VW for weighting     │ │ • Compute turnover          │
 │ • Assign to HIGH/LOW        │ │                             │
 │ • Read chars values (*)     │ │                             │
 └─────────────────────────────┘ └─────────────────────────────┘
                                        │
                                        ▼
                                  OUTPUT INDEXED
                                  AT RETURN DATE

(*) Chars VALUES come from formation date, but OUTPUT is indexed at return date
```

### What Gets Read at Each Date

| Data Item | Read From | Indexed At |
|-----------|-----------|------------|
| **Signal** (sort_var) | Formation date (t) | - |
| **Rating** (for terciles) | Formation date (t) | - |
| **VW** (for weighting) | Formation date (t) | - |
| **Returns** | Return date (t+1) | Return date (t+1) |
| **Turnover** | Return date (t+1) | Return date (t+1) |
| **Chars** | Formation date (t) | Return date (t+1) |

### Why Index at Return Date?

The return date indexing is consistent across all PyBondLab strategies and follows the
standard convention in empirical asset pricing:

1. **Factor returns** should be dated when the return is realized (t+1)
2. **Turnover** reflects trading that occurs to implement the rebalance at t+1
3. **Characteristics** describe the portfolio composition, dated when returns accrue

```
Example:
--------
Formation: December 2024
Return:    January 2025

Output DataFrame:
                 ew_ls    vw_ls    turnover    char1
2025-01-31       0.45%    0.52%    0.18        4.5   ← Indexed at return date

Interpretation:
- The 0.52% VW return was earned in January 2025
- The 0.18 turnover was incurred to rebalance into January 2025
- The 4.5 char1 value was measured in December 2024 (formation)
```

### Result DataFrames

```python
result = sf.fit()

# Returns: indexed by return date
ew_ls, vw_ls = result.get_long_short()
# Index: [2024-02-29, 2024-03-31, ..., 2025-01-31]

# Turnover: indexed by return date (starting from 2nd date)
ew_turn, vw_turn = result.get_turnover()
# Index: [2024-02-29, 2024-03-31, ..., 2025-01-31]

# Characteristics: indexed by return date
ew_chars, vw_chars = result.get_characteristics()
# Each DataFrame has Index: [2024-02-29, 2024-03-31, ..., 2025-01-31]
# VALUES are from formation date, but INDEX is return date
```

---

## Full API Reference

### WithinFirmSort Strategy

```python
pbl.WithinFirmSort(
    holding_period: int,
    sort_var: str,
    firm_id_col: str = 'PERMNO',
    min_bonds_per_firm: int = 2,
    rating_bins: list = [-np.inf, 7, 10, np.inf],
    num_portfolios: int = 2,
    rebalance_frequency: str = 'monthly',
    rebalance_month: int = 6,
    verbose: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holding_period` | int | **required** | Holding period (must be 1 for now) |
| `sort_var` | str | **required** | Signal column name (e.g., 'CS', 'eff_yld') |
| `firm_id_col` | str | `'PERMNO'` | Column name for firm identifier |
| `min_bonds_per_firm` | int | `2` | Minimum bonds required per firm-date-rating group |
| `rating_bins` | list | `[-inf, 7, 10, inf]` | Rating bin edges for terciles |
| `num_portfolios` | int | `2` | Always 2 (HIGH/LOW), other values trigger warning |
| `rebalance_frequency` | str/int | `'monthly'` | Rebalancing frequency |
| `rebalance_month` | int/list | `6` | Month(s) for non-monthly rebalancing |
| `verbose` | bool | `True` | Print initialization details |

---

## Custom Column Names

If your data uses different column names than PyBondLab expects, use the `fit()` method parameters:

```python
import PyBondLab as pbl

strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='credit_spread',        # Your signal column name
    firm_id_col='firm_identifier',   # Your firm ID column name
)

result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
).fit(
    IDvar='cusip_id',         # Bond identifier (default: 'ID')
    RETvar='ret_vw_bgn',      # Return column (default: 'ret')
    VWvar='mcap_e',           # Value weight column (default: 'VW')
    RATINGvar='spc_rat',      # Rating column (default: 'RATING_NUM')
    DATEvar='date',           # Date column (default: 'date')
    PRICEvar='prc_eom',       # Price column (default: 'PRICE', optional)
)

ew_ls, vw_ls = result.get_long_short()
```

### Default Column Names

| Parameter | Default Name | Description |
|-----------|--------------|-------------|
| `IDvar` | `'ID'` | Bond identifier (e.g., CUSIP) |
| `DATEvar` | `'date'` | Date column |
| `RETvar` | `'ret'` | Monthly return |
| `VWvar` | `'VW'` | Value weight (market value) |
| `RATINGvar` | `'RATING_NUM'` | Numeric rating (1=AAA, ..., 21=D) |
| `PRICEvar` | `'PRICE'` | Price (optional, for price filters) |

**Note:** The `firm_id_col` is specified in the `WithinFirmSort` constructor, not in `fit()`.

### Complete Example with Custom Columns

```python
import PyBondLab as pbl
import pandas as pd

# Your data has non-standard column names
data = pd.read_parquet('my_bond_data.parquet')
print(data.columns)
# ['date', 'cusip_id', 'ret_vw_bgn', 'mcap_e', 'spc_rat', 'firm_id', 'credit_spread']

# Initialize strategy with your firm ID and signal column names
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='credit_spread',     # Your signal column
    firm_id_col='firm_id',        # Your firm identifier column
    min_bonds_per_firm=2,
    verbose=True
)

# Run with column mapping in fit()
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
    chars=['credit_spread'],      # Aggregate this characteristic
    verbose=True
)

result = sf.fit(
    IDvar='cusip_id',
    RETvar='ret_vw_bgn',
    VWvar='mcap_e',
    RATINGvar='spc_rat'
)

# Access results
ew_ls, vw_ls = result.get_long_short()
print(f"VW Sharpe: {vw_ls.mean() / vw_ls.std() * 12**0.5:.2f}")
```

---

## Fast Path vs Slow Path

> **Note:** The WithinFirmSort fast path is currently **disabled** due to ranking discrepancies
> between the fast and slow paths. All WithinFirmSort computations use the slow (pandas) path,
> which always produces correct results. The fast path code is retained for future re-enablement
> once the discrepancies are resolved.

### Current Behavior

WithinFirmSort always uses the slow path regardless of configuration. The slow path uses
optimized numba kernels for turnover and characteristics aggregation where applicable.

---

## Feature Support

| Feature | Supported | Path Used | Notes |
|---------|-----------|-----------|-------|
| **Turnover** | ✅ YES | Slow | Uses optimized Phase 4 numba kernels |
| **Characteristics** | ✅ YES | Slow | Hierarchical aggregation (same as returns) |
| **HP=1** | ✅ YES | Slow | Full support |
| **HP>1 (Staggered)** | ❌ DISABLED | N/A | Raises ValueError (known bugs) |
| **Banding** | ❌ NO | N/A | Not applicable (see below) |
| **Rating filter** | ✅ YES | Both | Via subset_filter or rating parameter |

### Why No Banding?

WithinFirmSort only has 2 portfolios (HIGH and LOW). Banding prevents reassignment
when a bond's rank changes by less than `banding/nport`. With `nport=2` and typical
`banding=1`, this would require a rank change of 0.5 (i.e., moving from one portfolio
to the other), which is always the case when rank changes between HIGH and LOW.
Therefore, banding is meaningless for WithinFirmSort and is not implemented.

### Why HP>1 Disabled?

HP>1 (staggered rebalancing) has known bugs in the cohort averaging logic and is
currently disabled. Attempting to use `holding_period > 1` will raise a `ValueError`.

---

## Performance

### Current Performance (Phase 16 Optimizations)

| Configuration | Time | Notes |
|---------------|------|-------|
| HP=1, turnover=False, chars=None | ~1.0s | Slow path (fast path currently disabled) |
| HP=1, turnover=True | ~1.0s | Slow path with optimized turnover |
| HP=1, chars=['char1', 'char2'] | ~1.0s | Slow path with chars aggregation |

*Test data: 11,940 rows, 199 bonds, 50 firms, 60 dates*

### Large Data Performance

| Dataset Size | Time (Slow Path) |
|--------------|------------------|
| 100K rows | ~3s |
| 1M rows | ~15s |
| 2.4M rows | ~35s |

---

## Examples

### Basic Usage

```python
import PyBondLab as pbl

# Initialize strategy
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',                 # Credit spread
    firm_id_col='PERMNO',
    min_bonds_per_firm=2,
)

# Run formation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    verbose=True
).fit()

# Get results
ew_ls, vw_ls = result.get_long_short()
print(f"Mean: {vw_ls.mean()*12:.2%}")
print(f"Sharpe: {vw_ls.mean()/vw_ls.std()*12**0.5:.2f}")
```

### With Turnover

```python
# Slow path with turnover computation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,               # Enables turnover tracking
    verbose=True
).fit()

# Get turnover
ew_turn, vw_turn = result.get_turnover()
print(f"Average turnover: {vw_turn.mean().mean():.1%}")
```

### With Characteristics

```python
# Slow path with characteristics aggregation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    chars=['duration', 'spread', 'rating'],  # Aggregate these
    verbose=True
).fit()

# Get characteristics (tuple of dicts)
ew_chars, vw_chars = result.get_characteristics()

# Access specific characteristic
print(ew_chars['duration'])  # DataFrame with LOW, HIGH columns
print(vw_chars['spread'])
```

### Custom Rating Bins

```python
import numpy as np

# Custom rating terciles
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',
    firm_id_col='PERMNO',
    # Custom bins: [AAA-AA] [A-BBB] [BB and below]
    rating_bins=[-np.inf, 4, 10, np.inf],
    min_bonds_per_firm=3,        # Require 3+ bonds per firm
)
```

### Comparison with SingleSort

```python
import PyBondLab as pbl

# WithinFirmSort - isolates within-firm dispersion
wfs = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',
    firm_id_col='PERMNO',
    verbose=False
)
result_wfs = pbl.StrategyFormation(data, strategy=wfs, turnover=False).fit()
ew_wfs, vw_wfs = result_wfs.get_long_short()

# SingleSort - cross-sectional sorting
ss = pbl.SingleSort(
    holding_period=1,
    sort_var='CS',
    num_portfolios=5,
    verbose=False
)
result_ss = pbl.StrategyFormation(data, strategy=ss, turnover=False).fit()
ew_ss, vw_ss = result_ss.get_long_short()

# Compare results
print(f"WithinFirmSort VW: {vw_wfs.mean()*12:.2%} (annual)")
print(f"SingleSort VW:     {vw_ss.mean()*12:.2%} (annual)")
print(f"Correlation:       {vw_wfs.corr(vw_ss):.3f}")
```

---

## Architecture

### File Structure

```
PyBondLab/
├── StrategyClass.py       # WithinFirmSort class definition
├── utils_within_firm.py   # Core within-firm computation functions
├── numba_core.py          # Fast path numba kernels
├── precompute.py          # Integration with precomputation (line 370+)
└── PyBondLab.py           # Integration with aggregation (line 2185+)
```

### Key Functions

1. **`WithinFirmSort`** (StrategyClass.py)
   - Strategy class with parameters and validation

2. **`compute_within_firm_portfolios()`** (utils_within_firm.py)
   - Creates rating terciles
   - Groups by (date, rating_terc, firm)
   - Calls numba kernel for portfolio assignment

3. **`compute_within_firm_assignments_numba()`** (utils_within_firm.py)
   - Numba-compiled core
   - Computes within-firm percentile thresholds
   - Assigns bonds to HIGH/LOW portfolios

4. **`_fit_withinfirm_fast()`** (PyBondLab.py)
   - Numba-based fast path (currently disabled due to ranking discrepancies)

5. **`_aggregate_within_firm_results()`** (PyBondLab.py)
   - Hierarchical return aggregation
   - Characteristics aggregation

6. **`compute_within_firm_aggregation_fast()`** (numba_core.py)
   - Numba kernel for return aggregation
   - Handles EW and VW separately

7. **`compute_within_firm_chars_aggregation()`** (numba_core.py)
   - Numba kernel for characteristics aggregation
   - Uses same hierarchical structure as returns

---

## Validation

### Run Validation Script

```bash
python examples/validate_withinfirmsort.py
```

This validates:
1. Basic execution
2. Turnover computation
3. Characteristics aggregation
4. Difference from SingleSort (confirming within-firm logic)

### Manual Validation

```python
import PyBondLab as pbl

# Run with and without turnover and compare
strategy = pbl.WithinFirmSort(holding_period=1, sort_var='signal')

# Without turnover
result_no_to = pbl.StrategyFormation(
    data, strategy, turnover=False, verbose=True
).fit()

# With turnover
result_to = pbl.StrategyFormation(
    data, strategy, turnover=True, verbose=True
).fit()

# Compare returns (should match exactly)
ew_no_to, vw_no_to = result_no_to.get_long_short()
ew_to, vw_to = result_to.get_long_short()

diff = (vw_no_to - vw_to).abs().max()
print(f"Max difference: {diff:.2e}")  # Should be ~0
```

---

## Summary

`WithinFirmSort` provides a methodology for constructing bond factors that **control for
issuer-specific shocks unrelated to the bond characteristic of interest**.

### Key Features

1. **Pseudo firm fixed-effect control**: By sorting within each firm, cross-firm variation
   is removed, isolating the return premium associated with the characteristic itself

2. **Hierarchical aggregation**:
   - Level 1: VW returns within each firm
   - Level 2: Cap-weighted across firms within rating tercile
   - Level 3: Simple average across rating terciles

3. **Return date indexing**: All outputs (returns, turnover, characteristics) are indexed
   by return date (t+1), consistent with standard asset pricing conventions

4. **Full feature support** for turnover and characteristics

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Standard factor construction | Use SingleSort |
| Testing if characteristic predicts returns after controlling for issuer | **Use WithinFirmSort** |
| Concerned about firm-level confounds | **Use WithinFirmSort** |
| Constructing "pure" maturity/liquidity/duration factors | **Use WithinFirmSort** |

For batch processing of multiple signals, see [BatchWithinFirmSortFormation_README.md](BatchWithinFirmSortFormation_README.md).
