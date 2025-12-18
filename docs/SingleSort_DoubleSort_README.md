# SingleSort and DoubleSort Strategy Guide

`SingleSort` and `DoubleSort` are strategy classes that define how bonds are sorted into portfolios based on one or two characteristics. These strategies are then executed using `StrategyFormation` to compute portfolio returns.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [SingleSort Strategy](#singlesort-strategy)
   - [Basic Parameters](#basic-parameters)
   - [Custom Breakpoints](#custom-breakpoints)
   - [Rebalancing Options](#rebalancing-options)
   - [Breakpoint Universe](#breakpoint-universe)
3. [DoubleSort Strategy](#doublesort-strategy)
   - [Unconditional vs Conditional Sorting](#unconditional-vs-conditional-sorting)
   - [DoubleSort Parameters](#doublesort-parameters)
4. [StrategyFormation Execution](#strategyformation-execution)
   - [Rating Filtering](#rating-filtering)
   - [Turnover and Banding](#turnover-and-banding)
   - [Characteristics Tracking](#characteristics-tracking)
5. [Examples](#examples)
   - [Basic SingleSort](#basic-singlesort)
   - [SingleSort with Custom Breakpoints](#singlesort-with-custom-breakpoints)
   - [SingleSort with Different Rebalancing](#singlesort-with-different-rebalancing)
   - [DoubleSort Unconditional](#doublesort-unconditional)
   - [DoubleSort Conditional](#doublesort-conditional)
   - [Rating Filtering Examples](#rating-filtering-examples)
   - [Complete Workflow Example](#complete-workflow-example)
6. [Accessing Results](#accessing-results)
7. [Advanced Options](#advanced-options)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
import PyBondLab as pbl

# Create a single-sort strategy (quintiles based on momentum)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',
    num_portfolios=5,
)

# Execute the strategy
sf = pbl.StrategyFormation(data=data, strategy=strategy)
result = sf.fit()

# Get long-short portfolio returns
ew_ls, vw_ls = result.get_long_short()
print(f"Sharpe: {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
```

---

## SingleSort Strategy

`SingleSort` sorts bonds into portfolios based on a single characteristic.

### Basic Parameters

```python
pbl.SingleSort(
    holding_period: int,              # Required: months to hold portfolios
    sort_var: str,                    # Required: column name to sort on
    num_portfolios: int = 5,          # Number of portfolios (quintiles=5, deciles=10)
    breakpoints: List[float] = None,  # Custom percentile breakpoints
    lookback_period: int = None,      # For signal calculation (optional)
    skip: int = None,                 # Skip period (optional)
    rebalance_frequency: str = 'monthly',  # Rebalancing frequency
    rebalance_month: int = 6,         # Month for annual/semi-annual rebalancing
    breakpoint_universe_func = None,  # Subset for computing breakpoints
    verbose: bool = True,             # Print details
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holding_period` | int | **required** | Number of months to hold each portfolio |
| `sort_var` | str | **required** | Column name containing the sorting variable |
| `num_portfolios` | int | 5 | Number of portfolios (e.g., 5 for quintiles) |
| `breakpoints` | List[float] | None | Custom percentile breakpoints |
| `lookback_period` | int | None | Lookback for derived signals |
| `skip` | int | None | Skip period between signal and holding |
| `rebalance_frequency` | str/int | 'monthly' | How often to rebalance |
| `rebalance_month` | int/List | 6 | Month(s) for non-monthly rebalancing |
| `breakpoint_universe_func` | callable | None | Function to subset breakpoint universe |
| `verbose` | bool | True | Print initialization details |

### Custom Breakpoints

Instead of equal quantiles, use custom percentile breakpoints:

```python
# Three portfolios: bottom 30%, middle 40%, top 30%
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',
    breakpoints=[30, 70],  # Splits at 30th and 70th percentiles
)
# Results in 3 portfolios: [0-30], [30-70], [70-100]

# Asymmetric quintiles focusing on extremes
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='value',
    breakpoints=[10, 30, 70, 90],  # 5 portfolios with different sizes
)
# Results in: [0-10], [10-30], [30-70], [70-90], [90-100]
```

**Note:** `num_portfolios` is automatically set to `len(breakpoints) + 1`.

### Rebalancing Options

```python
# Monthly rebalancing (default)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',
    num_portfolios=5,
    rebalance_frequency='monthly',
)

# Quarterly rebalancing (every 3 months)
strategy = pbl.SingleSort(
    holding_period=3,
    sort_var='value',
    num_portfolios=5,
    rebalance_frequency='quarterly',
)

# Annual rebalancing in June
strategy = pbl.SingleSort(
    holding_period=12,
    sort_var='size',
    num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=6,  # June
)

# Semi-annual rebalancing in June and December
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='momentum',
    num_portfolios=5,
    rebalance_frequency='semi-annual',
    rebalance_month=[6, 12],
)

# Custom frequency (every 4 months)
strategy = pbl.SingleSort(
    holding_period=4,
    sort_var='momentum',
    num_portfolios=5,
    rebalance_frequency=4,  # Integer = months between rebalancing
)
```

**Rebalancing frequency options:**
| Value | Description |
|-------|-------------|
| `'monthly'` | Rebalance every month |
| `'quarterly'` | Rebalance every 3 months |
| `'semi-annual'` | Rebalance every 6 months |
| `'annual'` | Rebalance once per year |
| `int` | Rebalance every N months |

### Breakpoint Universe

Compute breakpoints from a subset of the data (e.g., only Investment Grade bonds):

```python
# Use only IG bonds for computing breakpoints
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',
    num_portfolios=5,
    breakpoint_universe_func=lambda df: df['RATING_NUM'] <= 10,
)

# Use only large bonds (above median market cap)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',
    num_portfolios=5,
    breakpoint_universe_func=lambda df: df['VW'] > df['VW'].median(),
)

# Use a column indicator (e.g., 'nyse' == 1)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',
    num_portfolios=5,
    breakpoint_universe_func='nyse',  # Uses data['nyse'] == 1
)
```

---

## DoubleSort Strategy

`DoubleSort` sorts bonds into portfolios based on two characteristics, creating a grid of portfolios.

### Unconditional vs Conditional Sorting

**Unconditional (Independent) Sort:**
- Both variables are sorted independently
- Creates `num_portfolios × num_portfolios2` portfolios
- Example: 5×5 = 25 portfolios

**Conditional (Dependent) Sort:**
- First sort on primary variable
- Then sort on secondary variable **within** each primary group
- Controls for the primary variable
- Creates `num_portfolios × num_portfolios2` portfolios

```python
# Unconditional: Size and Value sorted independently
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='size',
    sort_var2='value',
    num_portfolios=5,
    num_portfolios2=5,
    how='unconditional',  # Independent sorts
)

# Conditional: Value sorted within Size groups
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='size',      # First sort (control variable)
    sort_var2='value',    # Second sort (within groups)
    num_portfolios=5,
    num_portfolios2=5,
    how='conditional',    # Dependent sort
)
```

### DoubleSort Parameters

```python
pbl.DoubleSort(
    holding_period: int,              # Required: months to hold
    sort_var: str,                    # Required: primary sort variable
    sort_var2: str,                   # Required: secondary sort variable
    num_portfolios: int = 5,          # Primary portfolios
    num_portfolios2: int = 5,         # Secondary portfolios
    breakpoints: List[float] = None,  # Custom primary breakpoints
    breakpoints2: List[float] = None, # Custom secondary breakpoints
    how: str = 'unconditional',       # 'unconditional' or 'conditional'
    lookback_period: int = None,
    skip: int = None,
    rebalance_frequency: str = 'monthly',
    rebalance_month: int = 6,
    breakpoint_universe_func = None,  # For primary sort
    breakpoint_universe_func2 = None, # For secondary sort
    auto_match_signals: bool = False, # Auto-truncate mismatched dates
    verbose: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sort_var` | str | **required** | Primary sorting variable |
| `sort_var2` | str | **required** | Secondary sorting variable |
| `num_portfolios` | int | 5 | Number of primary portfolios |
| `num_portfolios2` | int | 5 | Number of secondary portfolios |
| `breakpoints` | List[float] | None | Custom primary breakpoints |
| `breakpoints2` | List[float] | None | Custom secondary breakpoints |
| `how` | str | 'unconditional' | Sort method: 'unconditional' or 'conditional' |
| `breakpoint_universe_func` | callable | None | Subset for primary breakpoints |
| `breakpoint_universe_func2` | callable | None | Subset for secondary breakpoints |
| `auto_match_signals` | bool | False | Auto-align signals with different date ranges |

---

## StrategyFormation Execution

`StrategyFormation` executes the strategy and computes portfolio returns.

```python
sf = pbl.StrategyFormation(
    data: pd.DataFrame,               # Required: bond panel data
    strategy: Strategy,               # Required: SingleSort or DoubleSort
    config: StrategyFormationConfig = None,  # Optional: detailed config
    # Or use convenience parameters:
    rating: str = None,               # Rating filter
    turnover: bool = False,           # Compute turnover
    chars: List[str] = None,          # Track characteristics
    banding_threshold: float = None,  # Banding for turnover reduction
    verbose: bool = True,
)
```

### Rating Filtering

Filter bonds by credit rating using three methods:

#### 1. String Categories

```python
# Investment Grade only (RATING_NUM 1-10)
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating='IG',
)

# Non-Investment Grade only (RATING_NUM 11-22)
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating='NIG',
)
```

#### 2. Numeric Range (Min, Max)

```python
# BBB bonds only (RATING_NUM 7-10)
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(7, 10),
)

# High-grade IG only (RATING_NUM 1-6, AAA to A)
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(1, 6),
)

# BB and B rated bonds (RATING_NUM 11-16)
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(11, 16),
)
```

#### 3. Using Config Object

```python
from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

# With full config object
config = StrategyFormationConfig(
    data=DataConfig(
        rating=(5, 15),  # Custom rating range
        chars=['duration', 'spread'],
    ),
    formation=FormationConfig(
        compute_turnover=True,
        banding_threshold=0.2,
    )
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    config=config,
)
```

**Rating Numeric Scale:**
| RATING_NUM | S&P Equivalent | Category |
|------------|---------------|----------|
| 1 | AAA | IG |
| 2 | AA+ | IG |
| 3 | AA | IG |
| 4 | AA- | IG |
| 5 | A+ | IG |
| 6 | A | IG |
| 7 | A- | IG |
| 8 | BBB+ | IG |
| 9 | BBB | IG |
| 10 | BBB- | IG |
| 11 | BB+ | NIG |
| 12 | BB | NIG |
| 13 | BB- | NIG |
| 14 | B+ | NIG |
| 15 | B | NIG |
| 16 | B- | NIG |
| 17 | CCC+ | NIG |
| 18 | CCC | NIG |
| 19 | CCC- | NIG |
| 20 | CC | NIG |
| 21 | C | NIG |
| 22 | D | NIG |

### Turnover and Banding

```python
# Compute turnover
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
)
result = sf.fit()
ew_turn, vw_turn = result.get_turnover()

# Use banding to reduce turnover
# Banding threshold = banding / num_portfolios
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
    banding_threshold=0.2,  # Or banding=1 for quintiles
)
```

**How banding works:**
- Bonds don't switch portfolios unless their rank changes by more than the threshold
- `banding=1` with 5 portfolios → threshold = 1/5 = 0.2
- Reduces turnover at the cost of slightly less pure portfolios

### Characteristics Tracking

Track portfolio-level characteristics:

```python
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    chars=['duration', 'spread', 'rating'],
)
result = sf.fit()

# Get portfolio characteristics (EW and VW)
ew_chars, vw_chars = result.get_characteristics()
print(ew_chars['duration'])  # Duration by portfolio and date
```

---

## Examples

### Basic SingleSort

```python
import PyBondLab as pbl
import pandas as pd

# Load your data
data = pd.read_parquet('bond_data.parquet')

# Momentum strategy: quintiles, monthly rebalancing
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum_3m',
    num_portfolios=5,
)

# Execute
sf = pbl.StrategyFormation(data=data, strategy=strategy)
result = sf.fit()

# Results
ew_ls, vw_ls = result.get_long_short()
print(f"EW Mean: {ew_ls.mean() * 12:.2%}")
print(f"VW Mean: {vw_ls.mean() * 12:.2%}")
print(f"Sharpe:  {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
```

### SingleSort with Custom Breakpoints

```python
# Focus on extreme quintiles (top and bottom 20%)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='value',
    breakpoints=[20, 40, 60, 80],  # Standard quintiles
)

# Or asymmetric: small bottom, large top
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='value',
    breakpoints=[10, 50, 90],  # 4 portfolios: [0-10], [10-50], [50-90], [90-100]
)

sf = pbl.StrategyFormation(data=data, strategy=strategy)
result = sf.fit()
```

### SingleSort with Different Rebalancing

```python
# Quarterly momentum with 3-month holding
strategy = pbl.SingleSort(
    holding_period=3,
    sort_var='momentum_3m',
    num_portfolios=5,
    rebalance_frequency='quarterly',
)

# Annual value strategy rebalancing in June
strategy = pbl.SingleSort(
    holding_period=12,
    sort_var='book_to_market',
    num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=6,
)

# Semi-annual in June and December
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='quality',
    num_portfolios=5,
    rebalance_frequency='semi-annual',
    rebalance_month=[6, 12],
)
```

### DoubleSort Unconditional

```python
# Size and Value double sort (5x5 = 25 portfolios)
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='size',
    sort_var2='value',
    num_portfolios=5,
    num_portfolios2=5,
    how='unconditional',
)

sf = pbl.StrategyFormation(data=data, strategy=strategy)
result = sf.fit()

# Results are for the value factor (sort_var2) averaged across size groups
ew_ls, vw_ls = result.get_long_short()
```

### DoubleSort Conditional

```python
# Value sorted within Size groups (controls for size)
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='size',      # Control variable
    sort_var2='value',    # Variable of interest
    num_portfolios=5,
    num_portfolios2=5,
    how='conditional',
)

sf = pbl.StrategyFormation(data=data, strategy=strategy)
result = sf.fit()

# This isolates the value effect independent of size
ew_ls, vw_ls = result.get_long_short()
```

### Rating Filtering Examples

```python
# Investment Grade bonds only
sf_ig = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating='IG',
)
result_ig = sf_ig.fit()
ew_ig, _ = result_ig.get_long_short()

# Non-Investment Grade bonds only
sf_nig = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating='NIG',
)
result_nig = sf_nig.fit()
ew_nig, _ = result_nig.get_long_short()

# Compare
print(f"IG Mean:  {ew_ig.mean() * 12:.2%}")
print(f"NIG Mean: {ew_nig.mean() * 12:.2%}")
```

```python
# BBB-rated bonds only (RATING_NUM 7-10)
sf_bbb = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(7, 10),
)
result_bbb = sf_bbb.fit()

# High-grade IG (AAA to A, RATING_NUM 1-6)
sf_high_ig = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(1, 6),
)
result_high_ig = sf_high_ig.fit()

# Low-grade IG (BBB, RATING_NUM 7-10)
sf_low_ig = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(7, 10),
)
result_low_ig = sf_low_ig.fit()

# BB and B rated (RATING_NUM 11-16)
sf_bb_b = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(11, 16),
)
result_bb_b = sf_bb_b.fit()

# CCC and below (RATING_NUM 17-22)
sf_distressed = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    rating=(17, 22),
)
result_distressed = sf_distressed.fit()
```

### Complete Workflow Example

```python
import PyBondLab as pbl
import pandas as pd
import numpy as np

# Load data
data = pd.read_parquet('bond_data.parquet')

# Pre-compute signals
data['momentum_3m'] = data.groupby('ID')['ret'].transform(
    lambda x: x.shift(1).rolling(3).sum()
)
data['momentum_6m'] = data.groupby('ID')['ret'].transform(
    lambda x: x.shift(1).rolling(6).sum()
)

# Define strategies
strategies = {
    'mom3_quintile': pbl.SingleSort(
        holding_period=1,
        sort_var='momentum_3m',
        num_portfolios=5,
        verbose=False,
    ),
    'mom6_quintile': pbl.SingleSort(
        holding_period=1,
        sort_var='momentum_6m',
        num_portfolios=5,
        verbose=False,
    ),
    'mom3_decile': pbl.SingleSort(
        holding_period=1,
        sort_var='momentum_3m',
        num_portfolios=10,
        verbose=False,
    ),
}

# Test across rating categories
ratings = {
    'ALL': None,
    'IG': 'IG',
    'NIG': 'NIG',
    'BBB': (7, 10),
    'BB_B': (11, 16),
}

# Run all combinations
results = {}
for strat_name, strategy in strategies.items():
    for rating_name, rating in ratings.items():
        key = f"{strat_name}_{rating_name}"

        sf = pbl.StrategyFormation(
            data=data,
            strategy=strategy,
            rating=rating,
            turnover=True,
            verbose=False,
        )
        result = sf.fit()

        ew_ls, vw_ls = result.get_long_short()
        ew_turn, _ = result.get_turnover()

        results[key] = {
            'ew_mean': ew_ls.mean() * 12,
            'ew_std': ew_ls.std() * np.sqrt(12),
            'sharpe': ew_ls.mean() / ew_ls.std() * np.sqrt(12),
            'turnover': ew_turn.mean().mean() if ew_turn is not None else np.nan,
            'n_periods': len(ew_ls),
        }

# Create summary DataFrame
summary_df = pd.DataFrame(results).T
print(summary_df.sort_values('sharpe', ascending=False))
```

---

## Accessing Results

### FormationResults Object

```python
result = sf.fit()

# Long-short returns (top minus bottom portfolio)
ew_ls, vw_ls = result.get_long_short()

# Turnover (if computed)
ew_turnover, vw_turnover = result.get_turnover()

# Characteristics (if tracked)
ew_chars, vw_chars = result.get_characteristics()

# Full portfolio returns (all portfolios)
returns = result.ea.returns
print(returns.ew_df)    # EW returns by portfolio
print(returns.vw_df)    # VW returns by portfolio
print(returns.ewls_df)  # EW long-short
print(returns.vwls_df)  # VW long-short
```

### Return Series Properties

```python
ew_ls, vw_ls = result.get_long_short()

# ew_ls is a pandas Series with DatetimeIndex
print(ew_ls.index)  # Dates
print(ew_ls.mean())  # Average return
print(ew_ls.std())   # Volatility

# Compute statistics
annualized_mean = ew_ls.mean() * 12
annualized_std = ew_ls.std() * np.sqrt(12)
sharpe = annualized_mean / annualized_std
max_drawdown = (ew_ls.cumsum() - ew_ls.cumsum().cummax()).min()
```

### Turnover Analysis

```python
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
)
result = sf.fit()

ew_turnover, vw_turnover = result.get_turnover()

# ew_turnover is a DataFrame: dates × portfolios
print(ew_turnover.mean())  # Average turnover by portfolio
print(ew_turnover.mean().mean())  # Overall average turnover
```

---

## Advanced Options

### Using Config Objects

```python
from PyBondLab.config import (
    StrategyFormationConfig,
    DataConfig,
    FormationConfig,
    FilterConfig,
)

# Full configuration
config = StrategyFormationConfig(
    data=DataConfig(
        rating='IG',
        chars=['duration', 'spread', 'coupon'],
        subset_filter={'VW': (1e6, 1e12)},  # Market cap filter
    ),
    formation=FormationConfig(
        dynamic_weights=True,
        compute_turnover=True,
        banding_threshold=0.2,
        verbose=True,
    ),
    filters=FilterConfig(
        adj='trim',
        level=0.2,
    ),
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    config=config,
)
result = sf.fit()
```

### Staggered Rebalancing

With `holding_period > 1`, portfolios are staggered across multiple cohorts:

```python
# 3-month holding period creates 3 cohorts
strategy = pbl.SingleSort(
    holding_period=3,
    sort_var='momentum',
    num_portfolios=5,
)

# Cohort 0: Formed in month 0, held months 1-3
# Cohort 1: Formed in month 1, held months 2-4
# Cohort 2: Formed in month 2, held months 3-5
# Returns are averaged across cohorts each month
```

### Breakpoint Universe for DoubleSort

```python
# Use IG bonds for size breakpoints, all bonds for value
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='size',
    sort_var2='value',
    num_portfolios=5,
    num_portfolios2=5,
    how='unconditional',
    breakpoint_universe_func=lambda df: df['RATING_NUM'] <= 10,  # IG for size
    breakpoint_universe_func2=None,  # All bonds for value
)
```

---

## Troubleshooting

### Common Issues

**1. "Sort variable not found in data"**
```python
# Check column exists
print(data.columns.tolist())
# Make sure sort_var matches exactly (case-sensitive)
```

**2. "No valid observations for portfolio formation"**
```python
# Check for NaN values in sort variable
print(data['momentum'].isna().sum())
# Check rating filter isn't too restrictive
print(data['RATING_NUM'].value_counts())
```

**3. "Mismatched date ranges" in DoubleSort**
```python
# Use auto_match_signals=True to auto-align
strategy = pbl.DoubleSort(
    sort_var='size',
    sort_var2='momentum',
    auto_match_signals=True,  # Truncates to common dates
    ...
)
```

**4. Empty portfolios**
```python
# Increase data or reduce num_portfolios
# Check if rating filter leaves enough bonds
```

### Validation

```python
# Check portfolio counts
result = sf.fit()
returns = result.ea.returns.ew_df

# Should have n_portfolios columns
print(f"Portfolios: {returns.columns.tolist()}")
print(f"Periods: {len(returns)}")

# Check for NaN values
print(f"NaN values: {returns.isna().sum().sum()}")
```

---

## Summary

| Feature | SingleSort | DoubleSort |
|---------|------------|------------|
| Sort variables | 1 | 2 |
| Portfolios | `num_portfolios` | `num_portfolios × num_portfolios2` |
| Sort method | N/A | 'unconditional' or 'conditional' |
| Rating filter | Via StrategyFormation | Via StrategyFormation |
| Custom breakpoints | Yes | Yes (both sorts) |
| Breakpoint universe | Yes | Yes (both sorts) |

**Rating filter options:**
- `'IG'`: Investment Grade (1-10)
- `'NIG'`: Non-Investment Grade (11-22)
- `(min, max)`: Custom numeric range
- `None`: All bonds

Start with `SingleSort` for simple factor analysis, use `DoubleSort` when you need to control for another variable or study interactions.
