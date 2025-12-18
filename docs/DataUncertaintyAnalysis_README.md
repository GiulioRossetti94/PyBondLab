# DataUncertaintyAnalysis User Guide

`DataUncertaintyAnalysis` is a high-level tool for studying how **data filters** affect factor returns. It systematically tests multiple filter configurations (trimming, price exclusions, bounce-back exclusions, winsorization) across different holding periods, returning both **Ex-Ante (EA)** and **Ex-Post (EP)** factor returns.

This is essential for understanding the robustness of trading strategies to data cleaning choices.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Key Concepts](#key-concepts)
   - [Ex-Ante vs Ex-Post Returns](#ex-ante-vs-ex-post-returns)
   - [Filter Types](#filter-types)
3. [Full API Reference](#full-api-reference)
4. [Examples](#examples)
   - [Basic Usage with Pre-computed Signal](#basic-usage-with-pre-computed-signal)
   - [Custom Column Names](#custom-column-names)
   - [Using Momentum/LTreversal Strategy](#using-momentumltreversal-strategy)
   - [Multiple Signals](#multiple-signals)
   - [Filter Configurations](#filter-configurations)
   - [Rating as a Dimension](#rating-as-a-dimension)
   - [Subset Filter (Characteristic-Based Filtering)](#subset-filter-characteristic-based-filtering)
   - [Filtering and Exporting Results](#filtering-and-exporting-results)
5. [Accessing Results](#accessing-results)
6. [Understanding the Summary Output](#understanding-the-summary-output)
7. [Performance Optimization](#performance-optimization)
8. [Common Use Cases](#common-use-cases)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from PyBondLab import DataUncertaintyAnalysis

# Test how different filters affect momentum returns
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5],           # Trim extreme returns
        'wins': [(99, 'both')],       # Winsorize at 99th percentile
    },
    num_portfolios=5,
).fit()

# View summary with t-statistics
print(results.summary())

# Access factor returns
ew_ea = results.ew_ea  # DataFrame: dates × configurations
```

---

## Key Concepts

### Ex-Ante vs Ex-Post Returns

When filters are applied, the strategy can be evaluated two ways:

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Ex-Ante (EA)** | Returns using **original** (unfiltered) returns | What you would have earned trading the signal |
| **Ex-Post (EP)** | Returns using **filtered** returns | Hypothetical returns if extreme observations never existed |

**Why this matters:**
- EA reflects **realizable** returns (you trade based on filtered signal, but earn actual returns)
- EP reflects **theoretical** returns (useful for understanding signal quality without outliers)
- Comparing EA vs EP reveals how much extreme observations contribute to returns

### Filter Types

| Filter | What it does | EA Returns | EP Returns |
|--------|--------------|------------|------------|
| **Baseline** | No filtering | Original | Original |
| **Trim** | Excludes extreme returns from ranking | Original | Trimmed |
| **Price** | Excludes bonds with extreme prices | Original | Price-filtered |
| **Bounce** | Excludes reversal patterns | Original | Bounce-filtered |
| **Wins** | Winsorizes (clips) extreme returns | Original | **NaN** (not meaningful) |

---

## Full API Reference

```python
DataUncertaintyAnalysis(
    data: pd.DataFrame,

    # Signal specification (one required)
    signals: List[str] = None,           # Pre-computed signal column(s)
    strategy: Strategy = None,           # Momentum or LTreversal object

    # Core parameters
    holding_periods: List[int] = [1, 3, 6],
    num_portfolios: int = 5,
    dynamic_weights: bool = True,

    # Filter configuration
    filters: Dict[str, List] = None,
    include_baseline: bool = True,

    # Rating configuration
    rating: Union[str, Tuple[int, int]] = None,  # 'IG', 'NIG', None, or (min, max) tuple
    ratings: List[...] = None,           # Rating as dimension: ['IG', 'NIG', (1, 10), None]

    # Characteristic filter
    subset_filter: Dict[str, Tuple[float, float]] = None,  # e.g., {'MATURITY': (1, 5)}

    # Column mapping
    columns: Dict[str, str] = None,

    # Execution
    n_jobs: int = 1,
    verbose: bool = True,
    use_fast_path: bool = True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | **required** | Bond panel data |
| `signals` | List[str] | None | Pre-computed signal column name(s) - uses **fast path** |
| `strategy` | Strategy | None | Momentum or LTreversal object - uses **slow path** |
| `holding_periods` | List[int] | [1, 3, 6] | Holding periods to test |
| `num_portfolios` | int | 5 | Number of quantile portfolios |
| `dynamic_weights` | bool | True | Use VW from d-1 (True) or formation date (False) |
| `filters` | Dict | None | Filter configurations (see [Filter Configurations](#filter-configurations)) |
| `include_baseline` | bool | True | Always include no-filter baseline |
| `rating` | str/tuple | None | Single rating: `'IG'`, `'NIG'`, `None`, or tuple `(min, max)` |
| `ratings` | List | None | Test multiple ratings: `['IG', 'NIG', (1, 10), None]` |
| `subset_filter` | Dict | None | **NEW**: Characteristic filter: `{'col': (min, max)}` |
| `columns` | Dict | None | Column name mapping |
| `n_jobs` | int | 1 | Parallel workers (for slow path only) |
| `verbose` | bool | True | Show progress |
| `use_fast_path` | bool | True | Use optimized numba kernels when possible |

### Returns

`DataUncertaintyResults` object with factor returns and summary statistics.

---

## Examples

### Basic Usage with Pre-computed Signal

```python
import pandas as pd
from PyBondLab import DataUncertaintyAnalysis

# Your data should have these columns:
# - date: datetime
# - ID: bond identifier
# - ret: monthly return
# - VW: value weight
# - RATING_NUM: numeric rating (1-22)
# - momentum: your pre-computed signal

results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5],
        'wins': [(99, 'both'), (95, 'both')],
    },
    num_portfolios=5,
    verbose=True,
).fit()

# View results
print(results.summary())
```

**Output:**
```
          signal  hp filter_type  level location  ew_ea_mean  ew_ea_tstat  vw_ea_mean  vw_ea_tstat  ew_ep_mean  ew_ep_tstat  n_obs  sharpe
0       momentum   1    baseline   None     None       0.234        2.45       0.198        2.12       0.234        2.45    120    0.71
1       momentum   1        trim    0.2    right       0.215        2.31       0.182        2.01       0.198        2.18    120    0.67
2       momentum   1        trim    0.5    right       0.189        2.12       0.165        1.89       0.172        1.95    120    0.61
3       momentum   1        wins     99     both       0.228        2.41       0.193        2.08         NaN         NaN    120    0.70
...
```

---

### Custom Column Names

If your data uses different column names:

```python
# Your data has: date, cusip_id, ret_vw, mcap_e, spc_rat, prc_eom, mom_signal

results = DataUncertaintyAnalysis(
    data=data,
    signals=['mom_signal'],
    columns={
        'ID': 'cusip_id',           # Bond identifier
        'ret': 'ret_vw',            # Return column
        'VW': 'mcap_e',             # Value weight
        'RATING_NUM': 'spc_rat',    # Rating column
        'PRICE': 'prc_eom',         # Price (only needed for price filters)
    },
    holding_periods=[1, 3],
    filters={'trim': [0.2]},
).fit()
```

**Default column mapping:**
```python
{
    'date': 'date',
    'ID': 'ID',
    'ret': 'ret',
    'VW': 'VW',
    'RATING_NUM': 'RATING_NUM',
    'PRICE': 'PRICE',
}
```

---

### Using Momentum/LTreversal Strategy

When you don't have a pre-computed signal, use a strategy object:

```python
import PyBondLab as pbl
from PyBondLab import DataUncertaintyAnalysis

# Define momentum strategy (3-month lookback, 1-month skip)
mom = pbl.Momentum(lookback_period=3, skip=1)

results = DataUncertaintyAnalysis(
    data=data,
    strategy=mom,                    # Strategy computes signal from returns
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5],
        'wins': [(99, 'both')],
    },
    num_portfolios=5,
).fit()

print(results.summary())
```

**Key difference with strategy:**
- Signal is computed from returns using `lookback_period` and `skip`
- For **wins** filter, signal is computed from winsorized returns (ex-ante winsorization)
- For **trim/price/bounce**, signal is computed from original returns

**Long-term Reversal example:**
```python
# 36-month lookback, 12-month skip
ltrev = pbl.LTreversal(lookback_period=36, skip=12)

results = DataUncertaintyAnalysis(
    data=data,
    strategy=ltrev,
    holding_periods=[1, 6, 12],
    filters={'trim': [0.3]},
).fit()
```

---

### Multiple Signals

Test multiple signals with the same filter configurations:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum_3m', 'momentum_6m', 'momentum_12m', 'value', 'quality'],
    holding_periods=[1, 3],
    filters={
        'trim': [0.2],
        'wins': [(99, 'both')],
    },
    num_portfolios=5,
    verbose=True,
).fit()

# Total configurations: 5 signals × 2 HPs × 3 filters = 30

# Filter to specific signal
mom3m_results = results.filter(signal='momentum_3m')
print(mom3m_results.summary())

# Compare signals
summary = results.summary()
baseline = summary[summary['filter_type'] == 'baseline']
print(baseline[['signal', 'hp', 'ew_ea_mean', 'ew_ea_tstat']].sort_values('ew_ea_tstat', ascending=False))
```

---

### Filter Configurations

#### Trim Filter (Exclude Extreme Returns)

```python
filters = {
    'trim': [
        0.2,              # Exclude ret > 20% (right tail)
        -0.3,             # Exclude ret < -30% (left tail)
        0.5,              # Exclude ret > 50% (right tail)
        [-0.3, 0.3],      # Exclude ret < -30% OR ret > 30% (both tails)
    ],
}
```

**Location inference:**
| Value | Interpretation |
|-------|---------------|
| `0.2` | Right tail (ret > 20%) |
| `-0.3` | Left tail (ret < -30%) |
| `[-0.3, 0.3]` | Both tails |

#### Price Filter (Exclude Extreme Prices)

```python
filters = {
    # Format: [[left_thresholds], [right_thresholds]]
    'price': [[1, 5], [150, 200]],
}

# This generates:
# - price_1_left: exclude price < $1
# - price_5_left: exclude price < $5
# - price_150_right: exclude price > $150
# - price_200_right: exclude price > $200
# - price_1_150_both: exclude price < $1 OR price > $150
# - price_1_200_both: exclude price < $1 OR price > $200
# - price_5_150_both: exclude price < $5 OR price > $150
# - price_5_200_both: exclude price < $5 OR price > $200
# Total: 2 left + 2 right + 4 both = 8 configurations
```

#### Bounce Filter (Exclude Reversals)

```python
filters = {
    'bounce': [
        0.05,             # Exclude if ret_t > 5% AND ret_{t-1} < -5%
        -0.05,            # Exclude if ret_t < -5% AND ret_{t-1} > 5%
        [-0.05, 0.05],    # Both directions
    ],
}
```

#### Wins Filter (Winsorization)

```python
filters = {
    'wins': [
        (99, 'both'),     # Clip at 1st and 99th percentiles
        (99, 'right'),    # Clip at 99th percentile only
        (95, 'both'),     # Clip at 5th and 95th percentiles
        (95, 'left'),     # Clip at 5th percentile only
    ],
}
```

**Note:** EP returns are NaN for wins filters because winsorization doesn't create a separate filtered return series.

#### Combined Filters

```python
# Test all filter types together
filters = {
    'trim': [0.2, 0.5, [-0.3, 0.3]],
    'price': [[1, 5], [150, 200]],
    'bounce': [0.05, [-0.05, 0.05]],
    'wins': [(99, 'both'), (95, 'both')],
}

results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    filters=filters,
    include_baseline=True,  # Always include no-filter case
).fit()

# With baseline + 3 trim + 8 price + 2 bounce + 2 wins = 16 filter configs
# × 2 HPs = 32 total configurations
```

---

### Rating as a Dimension

Test across Investment Grade (IG) and Non-Investment Grade (NIG) separately:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum', 'value'],
    holding_periods=[1, 3],
    filters={'trim': [0.2]},
    ratings=['IG', 'NIG', None],  # IG, NIG, and All bonds
).fit()

# Total: 2 signals × 3 ratings × 2 filters × 2 HPs = 24 configurations

# Filter by rating
ig_results = results.filter(rating='IG')
nig_results = results.filter(rating='NIG')
all_bonds = results.filter(rating=None)  # No rating restriction

# Compare IG vs NIG
summary = results.summary()
comparison = summary[summary['filter_type'] == 'baseline'][
    ['signal', 'hp', 'rating', 'ew_ea_mean', 'ew_ea_tstat']
]
print(comparison.pivot(index=['signal', 'hp'], columns='rating', values='ew_ea_mean'))
```

**Rating definitions:**
- `'IG'`: RATING_NUM 1-10 (Investment Grade)
- `'NIG'`: RATING_NUM 11-22 (Non-Investment Grade / High Yield)
- `None`: All bonds (no rating restriction)

#### Using Rating Tuples

You can also specify custom rating ranges using tuples `(min, max)`:

```python
# Single rating as tuple (equivalent to 'IG')
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    rating=(1, 10),  # Same as 'IG'
).fit()

# Custom rating range (e.g., only BBB-rated bonds)
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    rating=(7, 10),  # BBB+, BBB, BBB- (RATING_NUM 7-10)
).fit()

# Multiple custom ranges as dimension
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    ratings=[
        'IG',           # Investment Grade (1-10)
        'NIG',          # Non-Investment Grade (11-22)
        (1, 5),         # AA and above
        (6, 10),        # A to BBB-
        (11, 15),       # BB to B
        None,           # All bonds
    ],
).fit()
```

---

### Subset Filter (Characteristic-Based Filtering)

Filter the universe based on bond characteristics at formation date:

```python
# Filter to bonds with specific characteristics
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    subset_filter={
        'MATURITY': (1, 5),      # Maturity 1-5 years
        'DURATION': (2, 8),      # Duration 2-8 years
    },
).fit()
```

**Key behaviors:**
- Filters are applied at **formation date only** (no look-ahead bias)
- Bonds excluded from ranking can still contribute returns if they were ranked in prior periods
- Multiple filters are combined with AND logic

#### Combining Rating and Subset Filter

```python
# IG bonds with maturity 1-5 years
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    rating='IG',
    subset_filter={'MATURITY': (1, 5)},
).fit()

# Custom rating range with characteristic filter
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    rating=(7, 10),              # BBB bonds only
    subset_filter={
        'DURATION': (3, 7),      # Mid-duration
        'char1': (-1.0, 1.0),    # Custom characteristic range
    },
).fit()

# Subset filter with rating as dimension
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3],
    ratings=['IG', 'NIG', None],
    subset_filter={'MATURITY': (1, 5)},  # Applied to all rating categories
).fit()
```

**Note:** The `subset_filter` applies the same filter to all configurations. It is not treated as a dimension (unlike `ratings`).

---

### Filtering and Exporting Results

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum', 'value'],
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5],
        'wins': [(99, 'both')],
    },
).fit()

# Filter by multiple criteria
hp1_trim = results.filter(hp=1, filter_type='trim')
momentum_only = results.filter(signal='momentum')
right_tail = results.filter(location='right')

# Get factor returns for filtered results
ew_ea = hp1_trim.ew_ea  # DataFrame with filtered columns only

# Export to Excel (multiple sheets)
results.to_excel('data_uncertainty_results.xlsx')

# Excel file contains:
# - ew_ea: EW Ex-Ante factor returns
# - vw_ea: VW Ex-Ante factor returns
# - ew_ep: EW Ex-Post factor returns
# - vw_ep: VW Ex-Post factor returns
# - summary: Summary statistics
# - configs: Configuration metadata
```

---

## Accessing Results

### Factor Return DataFrames

```python
# Four factor panels available
ew_ea = results.ew_ea  # Equal-weighted, Ex-Ante
vw_ea = results.vw_ea  # Value-weighted, Ex-Ante
ew_ep = results.ew_ep  # Equal-weighted, Ex-Post
vw_ep = results.vw_ep  # Value-weighted, Ex-Post

# Each is a DataFrame: dates × configurations
print(ew_ea.head())
#             momentum_hp1_baseline  momentum_hp1_trim_0.2  momentum_hp3_baseline  ...
# date
# 2010-01-31                 0.0123                 0.0098                 0.0087
# 2010-02-28                 0.0087                 0.0076                 0.0065
# ...
```

### Column Naming Convention

```
{signal}_hp{hp}_{filter_type}[_{level}][_{location}]
```

Examples:
- `momentum_hp1_baseline`
- `momentum_hp1_trim_0.2` (right tail inferred)
- `momentum_hp3_trim_-0.3_0.3` (both tails explicit)
- `value_hp1_wins_99_both`
- `momentum_hp1_price_5_150_both`

### Configuration Metadata

```python
# Get metadata for all configurations
configs = results.configs

print(configs)
#                    column_name    signal  hp filter_type level location
# 0   momentum_hp1_baseline      momentum   1    baseline  None     None
# 1   momentum_hp1_trim_0.2      momentum   1        trim   0.2    right
# 2   momentum_hp1_trim_0.5      momentum   1        trim   0.5    right
# ...
```

### Summary Statistics

```python
summary = results.summary()

# Columns include:
# - signal, hp, filter_type, level, location, rating
# - ew_ea_mean, ew_ea_tstat (Newey-West)
# - vw_ea_mean, vw_ea_tstat
# - ew_ep_mean, ew_ep_tstat
# - vw_ep_mean, vw_ep_tstat
# - n_obs, sharpe (annualized EW EA Sharpe)

# Sort by t-statistic
print(summary.sort_values('ew_ea_tstat', ascending=False).head(10))
```

---

## Understanding the Summary Output

| Column | Description |
|--------|-------------|
| `signal` | Signal name |
| `hp` | Holding period |
| `filter_type` | 'baseline', 'trim', 'price', 'bounce', 'wins' |
| `level` | Filter threshold value |
| `location` | Tail: 'left', 'right', 'both', or None |
| `rating` | 'IG', 'NIG', or None |
| `ew_ea_mean` | EW Ex-Ante mean return (%) |
| `ew_ea_tstat` | Newey-West t-statistic for EW EA |
| `vw_ea_mean` | VW Ex-Ante mean return (%) |
| `vw_ea_tstat` | Newey-West t-statistic for VW EA |
| `ew_ep_mean` | EW Ex-Post mean return (%) |
| `ew_ep_tstat` | Newey-West t-statistic for EW EP |
| `vw_ep_mean` | VW Ex-Post mean return (%) |
| `vw_ep_tstat` | Newey-West t-statistic for VW EP |
| `n_obs` | Number of time periods |
| `sharpe` | Annualized Sharpe ratio (EW EA) |

**Notes:**
- Means are in percentage terms (×100)
- Newey-West lag = int(T^0.25) where T = n_obs
- Sharpe = (mean / std) × sqrt(12) for monthly data

---

## Performance Optimization

### Fast Path vs Slow Path

| Input | Path Used | Speed |
|-------|-----------|-------|
| `signals=['col1', 'col2']` | **Fast** (numba) | ~75x faster |
| `strategy=Momentum(...)` | **Slow** (pandas) | Baseline |

**Fast path** (pre-computed signals):
- Uses parallel numba kernels
- Processes all (date × filter) combinations at once
- Recommended for most use cases

**Slow path** (strategy objects):
- Required when signal must be computed from returns
- Supports wins filter with ex-ante winsorization
- Uses `n_jobs` for parallelization

### Performance Benchmarks

| Dataset | Configurations | Fast Path | Slow Path | Speedup |
|---------|---------------|-----------|-----------|---------|
| 60 dates × 500 bonds | 20 | 0.43s | 31.9s | **75x** |
| 120 dates × 1000 bonds | 50 | 1.2s | ~80s | **~65x** |

### Recommendations

1. **Use pre-computed signals when possible**: Much faster
   ```python
   # Fast: use signals parameter
   results = DataUncertaintyAnalysis(data=data, signals=['momentum'], ...)

   # Slow: use strategy parameter
   results = DataUncertaintyAnalysis(data=data, strategy=Momentum(...), ...)
   ```

2. **Pre-compute momentum/reversal signals**:
   ```python
   # Compute signal once, then test many filter configurations
   data['momentum'] = data.groupby('ID')['ret'].transform(
       lambda x: x.shift(1).rolling(3).sum()
   )

   results = DataUncertaintyAnalysis(
       data=data,
       signals=['momentum'],  # Fast path
       filters={'trim': [0.1, 0.2, 0.3, 0.4, 0.5]},  # Many filters = fast
   ).fit()
   ```

3. **Parallel processing for slow path**:
   ```python
   results = DataUncertaintyAnalysis(
       data=data,
       strategy=Momentum(lookback_period=3, skip=1),
       n_jobs=4,  # Use 4 workers
   ).fit()
   ```

---

## Common Use Cases

### 1. Factor Robustness Testing

Test if factor returns are driven by extreme observations:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1],
    filters={
        'trim': [0.1, 0.2, 0.3, 0.5, 1.0],  # Progressive trimming
    },
).fit()

# Compare EA vs EP to see outlier impact
summary = results.summary()
print(summary[['filter_type', 'level', 'ew_ea_mean', 'ew_ep_mean']])
```

### 2. Price Filter Sensitivity

Understand impact of penny stocks and high-priced bonds:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['value'],
    holding_periods=[1, 6],
    filters={
        'price': [[1, 5, 10], [100, 150, 200]],
    },
).fit()

# Results show how excluding extreme prices affects returns
```

### 3. Rating Segmentation

Compare factor performance across credit quality:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum', 'value', 'quality'],
    holding_periods=[1, 3],
    filters={'trim': [0.2]},
    ratings=['IG', 'NIG', None],
).fit()

# Pivot to compare IG vs NIG
summary = results.summary()
pivot = summary.pivot_table(
    index=['signal', 'hp', 'filter_type'],
    columns='rating',
    values='ew_ea_mean'
)
print(pivot)
```

### 4. Momentum Lookback Comparison

Compare different momentum specifications:

```python
# Pre-compute multiple momentum signals
data['mom_3m'] = data.groupby('ID')['ret'].transform(lambda x: x.shift(1).rolling(3).sum())
data['mom_6m'] = data.groupby('ID')['ret'].transform(lambda x: x.shift(1).rolling(6).sum())
data['mom_12m'] = data.groupby('ID')['ret'].transform(lambda x: x.shift(1).rolling(12).sum())

results = DataUncertaintyAnalysis(
    data=data,
    signals=['mom_3m', 'mom_6m', 'mom_12m'],
    holding_periods=[1, 3, 6],
    filters={'trim': [0.2], 'wins': [(99, 'both')]},
).fit()

# Compare across lookbacks
summary = results.summary()
baseline = summary[summary['filter_type'] == 'baseline']
print(baseline.pivot(index='hp', columns='signal', values='ew_ea_tstat'))
```

### 5. Full Data Uncertainty Analysis

Comprehensive analysis for a research paper:

```python
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3, 6, 12],
    filters={
        'trim': [0.1, 0.2, 0.3, 0.5],
        'price': [[1, 5], [150, 200]],
        'bounce': [0.05, [-0.05, 0.05]],
        'wins': [(99, 'both'), (95, 'both')],
    },
    ratings=['IG', 'NIG', None],
    num_portfolios=5,
    verbose=True,
).fit()

# Export for paper
results.to_excel('momentum_data_uncertainty.xlsx')

# Create summary table
summary = results.summary()
summary.to_csv('momentum_summary.csv')
```

---

## Troubleshooting

### Common Issues

**1. "Signal columns not found in data"**
```python
# Check signal columns exist
print([s for s in signals if s not in data.columns])
```

**2. "PRICE column required for price filters"**
```python
# Either add PRICE column or use columns mapping
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    columns={'PRICE': 'bond_price'},  # Map your price column
    filters={'price': [[5], [150]]},
).fit()
```

**3. EP returns are NaN for wins filter**

This is expected! Winsorization clips values rather than excluding them, so there's no separate "filtered return" for EP calculation.

**4. Slow performance**
```python
# Use pre-computed signals instead of strategy objects
# Bad (slow):
results = DataUncertaintyAnalysis(strategy=Momentum(...), ...)

# Good (fast):
data['momentum'] = ...  # Pre-compute
results = DataUncertaintyAnalysis(signals=['momentum'], ...)
```

**5. Memory errors with many configurations**
```python
# Reduce scope or process in batches
results_hp1 = DataUncertaintyAnalysis(holding_periods=[1], ...).fit()
results_hp3 = DataUncertaintyAnalysis(holding_periods=[3], ...).fit()
```

### Validation

To verify fast path matches slow path:
```python
# Force slow path
results_slow = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    use_fast_path=False,  # Force slow path
    ...
).fit()

# Compare with fast path
results_fast = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    use_fast_path=True,   # Use fast path (default)
    ...
).fit()

# Should be identical
diff = (results_slow.ew_ea - results_fast.ew_ea).abs().max().max()
print(f"Max difference: {diff:.2e}")  # Should be < 1e-10
```

---

## Complete Example

```python
import pandas as pd
import numpy as np
from PyBondLab import DataUncertaintyAnalysis
from PyBondLab.pbl_test import generate_synthetic_data

# Generate or load data
data = generate_synthetic_data(n_dates=120, n_bonds=500, seed=42)

# Pre-compute momentum signal
data['momentum'] = data.groupby('ID')['ret'].transform(
    lambda x: x.shift(1).rolling(3).sum()
)

# Run comprehensive data uncertainty analysis
print("Running Data Uncertainty Analysis...")
results = DataUncertaintyAnalysis(
    data=data,
    signals=['momentum'],
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5, [-0.3, 0.3]],
        'price': [[5], [150]],
        'wins': [(99, 'both'), (95, 'both')],
    },
    ratings=['IG', 'NIG', None],
    num_portfolios=5,
    verbose=True,
).fit()

# View summary
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
summary = results.summary()
print(summary[['hp', 'rating', 'filter_type', 'level', 'ew_ea_mean', 'ew_ea_tstat', 'ew_ep_mean']].head(20))

# Compare baseline across ratings
print("\n" + "="*80)
print("BASELINE BY RATING")
print("="*80)
baseline = summary[summary['filter_type'] == 'baseline']
print(baseline.pivot(index='hp', columns='rating', values=['ew_ea_mean', 'ew_ea_tstat']))

# Impact of trimming
print("\n" + "="*80)
print("TRIMMING IMPACT (HP=1, All Bonds)")
print("="*80)
hp1_all = summary[(summary['hp'] == 1) & (summary['rating'].isna())]
print(hp1_all[['filter_type', 'level', 'ew_ea_mean', 'ew_ep_mean']])

# Export results
results.to_excel('data_uncertainty_results.xlsx')
print("\nResults exported to data_uncertainty_results.xlsx")

# Access factor returns for custom analysis
ew_ea = results.ew_ea
print(f"\nFactor returns shape: {ew_ea.shape}")
print(f"Columns: {list(ew_ea.columns)[:5]}...")
```

---

## Summary

`DataUncertaintyAnalysis` provides:

1. **Systematic filter testing** across trim, price, bounce, and wins configurations
2. **Ex-Ante and Ex-Post returns** to understand outlier impact
3. **Rating segmentation** for IG/NIG analysis
4. **Fast numba-optimized path** for pre-computed signals (75x speedup)
5. **Comprehensive summary statistics** with Newey-West t-stats
6. **Easy filtering and export** for research workflows

Start with pre-computed signals for fast exploration, then use strategy objects for full Momentum/LTreversal analysis with winsorization support.
