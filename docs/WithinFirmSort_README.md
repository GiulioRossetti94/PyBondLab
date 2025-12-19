# WithinFirmSort Strategy

## Overview

`WithinFirmSort` implements a within-firm high-low sorting methodology for constructing bond factors. Unlike standard cross-sectional sorting (SingleSort), this strategy sorts bonds **within each firm**, isolating within-firm bond dispersion from cross-firm differences.

## Key Differences from SingleSort

| Aspect | SingleSort | WithinFirmSort |
|--------|-----------|----------------|
| **Grouping** | None (cross-sectional) | Date × Rating Tercile × Firm |
| **Percentiles** | Global (e.g., 20/40/60/80 for quintiles) | Within-firm (33.3/66.7) |
| **Portfolios** | N portfolios (typically 5) | 2 (high/low only) |
| **Return Aggregation** | Simple VW average across bonds | Firm-cap-weighted → Rating-averaged |

## Methodology

### Step 1: Rating Tercile Assignment

Bonds are grouped into rating terciles:
- **IG+ (Tercile 1)**: Ratings 1-7 (AAA to A-)
- **IG- (Tercile 2)**: Ratings 8-10 (BBB+ to BBB-)
- **SG (Tercile 3)**: Ratings 11+ (BB+ and below)

Custom bins can be specified via `rating_bins` parameter.

### Step 2: Within-Firm Portfolio Formation

For each (date, rating tercile, firm) group:
1. Require minimum `min_bonds_per_firm` bonds (default: 2)
2. Compute 33.3rd and 66.7th percentile thresholds of the signal
3. Assign bonds:
   - Signal < 33.3rd percentile → **Low portfolio (Q1)**
   - Signal > 66.7th percentile → **High portfolio (Q2)**
   - Middle tercile bonds → **Unassigned** (excluded)

### Step 3: Hierarchical Return Aggregation

Returns are aggregated in a hierarchical manner:

```
For each date:
    For each rating tercile (1, 2, 3):
        For each firm in this rating tercile:
            - Compute VW return for high portfolio (Q2)
            - Compute VW return for low portfolio (Q1)
            - Compute firm-level H-L factor = Q2 - Q1

        Aggregate across firms (cap-weighted):
            rating_factor = Σ(firm_weight × firm_HL) / Σ(firm_weight)

    Average across rating terciles:
        overall_factor = mean(rating_factors)
```

## Usage

### Basic Example

```python
import PyBondLab as pbl
import numpy as np

# Initialize WithinFirmSort strategy
strategy = pbl.WithinFirmSort(
    holding_period=1,              # Monthly rebalancing
    sort_var='CS',                 # Sort on credit spread
    firm_id_col='PERMNO',          # Firm identifier column
    min_bonds_per_firm=2,          # Require at least 2 bonds per firm
    rating_bins=[-np.inf, 7, 10, np.inf],  # IG+, IG-, SG
    num_portfolios=2,              # High/low portfolios (always 2)
    verbose=True
)

# Run strategy formation
result = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,  # Or True for turnover computation
    verbose=True
).fit()

# Get long-short returns
ew_ls, vw_ls = result.get_long_short()

print(f"Mean return: {vw_ls.mean()*100:.3f}% per month")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holding_period` | int | required | Holding period (1 = monthly) |
| `sort_var` | str | required | Signal column name (e.g., 'CS', 'eff_yld') |
| `firm_id_col` | str | 'PERMNO' | Column name for firm identifier |
| `min_bonds_per_firm` | int | 2 | Minimum bonds required per firm-date-rating group |
| `rating_bins` | list | `[-inf, 7, 10, inf]` | Rating bin edges for terciles |
| `num_portfolios` | int | 2 | Always 2 (high/low), other values trigger warning |
| `rebalance_frequency` | str/int | 'monthly' | Rebalancing frequency |
| `rebalance_month` | int/list | 6 | Month(s) for non-monthly rebalancing |
| `verbose` | bool | True | Print initialization details |

## Feature Support

| Feature | Supported | Notes |
|---------|-----------|-------|
| **Turnover** | ✅ YES | Uses standard PyBondLab machinery |
| **HP>1 (Staggered)** | ✅ YES | Cohort averaging works correctly |
| **Chars** | ⏳ TODO | Will use Option B aggregation |
| **Banding** | ❌ NO | Not applicable (see below) |

### Why No Banding?

WithinFirmSort only has 2 portfolios (HIGH and LOW). Banding prevents reassignment
when a bond's rank changes by less than `banding/nport`. With `nport=2` and typical
`banding=1`, this would require a rank change of 0.5 (i.e., moving from one portfolio
to the other), which is always the case when rank changes between HIGH and LOW.
Therefore, banding is meaningless for WithinFirmSort and is not implemented.

### Characteristics Aggregation (Planned)

When implemented, characteristics will use **Option B aggregation** (same as returns):
1. **Within firm**: Compute VW-average char for HIGH and LOW portfolios
2. **Across firms**: Cap-weight firm-level chars within each rating tercile
3. **Across ratings**: Simple average across rating terciles

Output will be a DataFrame with columns `['LOW', 'HIGH']` for each characteristic.

## Data Requirements

Your data must include:

| Column | Description |
|--------|-------------|
| `ID` | Bond identifier (e.g., CUSIP) |
| `date` | Date column |
| `ret` | Monthly return |
| `VW` | Value weight (market value) |
| `RATING_NUM` | Numeric rating (1=AAA, ..., 21=D) |
| `{firm_id_col}` | Firm identifier (default: 'PERMNO') |
| `{sort_var}` | Signal column for sorting |

## Architecture

### File Structure

```
PyBondLab/
├── StrategyClass.py       # WithinFirmSort class definition
├── utils_within_firm.py   # Core within-firm computation functions
├── precompute.py          # Integration with precomputation (line 370+)
└── PyBondLab.py           # Integration with aggregation (line 2185+)
```

### Key Functions

1. **`compute_within_firm_portfolios()`** (utils_within_firm.py)
   - Creates rating terciles
   - Groups by (date, rating_terc, firm)
   - Calls numba kernel for portfolio assignment

2. **`compute_within_firm_assignments_numba()`** (utils_within_firm.py)
   - Numba-compiled core
   - Computes within-firm percentile thresholds
   - Assigns bonds to high/low portfolios

3. **`compute_within_firm_returns_aggregation()`** (utils_within_firm.py)
   - Post-processes portfolio assignments
   - Implements hierarchical aggregation
   - Computes firm-cap-weighted → rating-averaged returns

### Integration Points

1. **Precompute** (precompute.py:370-413)
   - Detected by `__strategy_name__ == "Within-Firm Sort"`
   - Uses `compute_within_firm_portfolios()` instead of standard thresholds

2. **Aggregation** (PyBondLab.py:2296-2300)
   - Detected by `is_within_firm` check
   - Uses `_aggregate_within_firm_results()` instead of standard aggregation

## Performance

### Current Baseline (Pre-Optimization)

| Configuration | Time |
|---------------|------|
| HP=1, no turnover | ~3.8s |
| HP=1, with turnover | ~3.7s |
| HP=3, no turnover | ~4.2s |
| HP=3, with turnover | ~4.3s |

*Test data: 11,940 rows, 199 bonds, 50 firms, 60 dates*

### Optimization Targets (Phase 16)

| Configuration | Current | Target | Speedup |
|---------------|---------|--------|---------|
| HP=1, no turnover, no chars | ~3.8s | <0.5s | 7x+ |
| HP=3, no turnover, no chars | ~4.2s | <0.6s | 7x+ |
| HP=1, with turnover | ~3.7s | <1.0s | 4x+ |
| HP=1, with chars | TBD | TBD | 5x+ |

See CLAUDE.md for the Phase 16 optimization plan details.

## Comparison with Standard Sorting

```python
# WithinFirmSort - isolates within-firm dispersion
wfs = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',
    firm_id_col='PERMNO',
    verbose=False
)
result_wfs = pbl.StrategyFormation(data, strategy=wfs).fit()

# SingleSort - cross-sectional sorting
ss = pbl.SingleSort(
    holding_period=1,
    sort_var='CS',
    num_portfolios=5,
    verbose=False
)
result_ss = pbl.StrategyFormation(data, strategy=ss).fit()

# Results will differ because:
# - WithinFirmSort sorts within each firm
# - SingleSort sorts across all bonds
```

## Example Script

See `examples/WithinFirmSort_Example.py` for a complete usage example including:
- Data loading/generation
- Basic strategy formation
- Comparison with standard sorting
- Risk-adjusted performance analysis
- Visualization

## Validation

Run the validation script to verify correct behavior:

```bash
python examples/validate_withinfirmsort.py
```

This validates:
1. Basic execution
2. turnover=True vs turnover=False consistency
3. HP=3 staggered rebalancing
4. Difference from SingleSort (confirming within-firm logic is applied)
