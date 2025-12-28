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
   - [Column Mapping (Custom Column Names)](#column-mapping-custom-column-names)
   - [Rating Filtering](#rating-filtering)
   - [Subset Filter (Characteristic-Based)](#subset-filter-characteristic-based)
   - [Turnover and Banding](#turnover-and-banding)
   - [Characteristics Tracking](#characteristics-tracking)
5. [Execution Paths: Slow, Fast, and Ultra-Fast](#execution-paths-slow-fast-and-ultra-fast)
   - [Path Selection Logic](#path-selection-logic)
   - [Performance Comparison](#performance-comparison)
   - [How to Ensure Fast Path](#how-to-ensure-fast-path)
6. [Examples](#examples)
   - [Basic SingleSort](#basic-singlesort)
   - [SingleSort with Custom Breakpoints](#singlesort-with-custom-breakpoints)
   - [SingleSort with Different Rebalancing](#singlesort-with-different-rebalancing)
   - [DoubleSort Unconditional](#doublesort-unconditional)
   - [DoubleSort Conditional](#doublesort-conditional)
   - [Rating Filtering Examples](#rating-filtering-examples)
   - [Subset Filter Examples](#subset-filter-examples)
   - [Complete Workflow Example](#complete-workflow-example)
7. [Accessing Results](#accessing-results)
8. [Advanced Options](#advanced-options)
9. [Troubleshooting](#troubleshooting)
10. [Detailed Timing and Mechanics (HP=1)](#detailed-timing-and-mechanics-hp1)
    - [Timeline Overview](#timeline-overview-hp1)
    - [Data Flow Diagram](#data-flow-diagram)
    - [Weight Computation](#weight-computation)
    - [Return Computation](#return-computation)
    - [Scaled Weights](#scaled-weights-for-turnover)
    - [Turnover Computation](#turnover-computation)
    - [Characteristics Computation](#characteristics-computation)
    - [Complete Example](#complete-example-hp1)
    - [ID Intersection Logic](#id-intersection-logic)
    - [Cohort Handling for HP=1](#cohort-handling-for-hp1)
    - [Summary Table](#summary-table-hp1)
11. [Detailed Timing and Mechanics (HP=3, Staggered Rebalancing)](#detailed-timing-and-mechanics-hp3-staggered-rebalancing)
    - [What is Staggered Rebalancing?](#what-is-staggered-rebalancing)
    - [Timeline Overview (HP=3)](#timeline-overview-hp3)
    - [Cohort Assignment](#cohort-assignment)
    - [Holding Period Loop](#holding-period-loop)
    - [Results Array Structure](#results-array-structure)
    - [dynamic_weights Parameter](#dynamic_weights-parameter-critical-for-hp1)
    - [Cohort Averaging](#cohort-averaging)
    - [Weight Computation for HP>1](#weight-computation-for-hp1)
    - [Turnover for HP>1](#turnover-for-hp1-staggered)
    - [Characteristics for HP>1](#characteristics-for-hp1)
    - [Complete Example (HP=3)](#complete-example-hp3)
    - [ID Intersection for HP>1](#id-intersection-for-hp1)
    - [Summary Table (HP=3)](#summary-table-hp3)
    - [Key Differences: HP=1 vs HP=3](#key-differences-hp1-vs-hp3)
    - [Turnover Interpretation for HP>1](#turnover-interpretation-for-hp1)
12. [The dynamic_weights Parameter](#the-dynamic_weights-parameter)
    - [What dynamic_weights Controls](#what-dynamic_weights-controls)
    - [Effect on HP=1 (No Effect)](#effect-on-hp1-no-effect)
    - [Effect on HP>1 (Critical Difference)](#effect-on-hp1-critical-difference)
    - [VW Source Date Equations](#vw-source-date-equations)
    - [ID Intersection Behavior](#id-intersection-behavior)
    - [Timeline Diagram: dynamic_weights=True vs False](#timeline-diagram-dynamic_weightstrue-vs-false)
    - [Weight Computation Equations](#weight-computation-equations-dynamic_weights)
    - [Effect on Turnover](#effect-on-turnover)
    - [Effect on Returns](#effect-on-returns)
    - [Defaults Across Classes](#defaults-across-classes)
    - [When to Use Each Setting](#when-to-use-each-setting)
    - [Summary Table (dynamic_weights)](#summary-table-dynamic_weights)

---

## Quick Start

```python
import PyBondLab as pbl

# Create a single-sort strategy (quintiles based on momentum)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum',    # Your signal column name
    num_portfolios=5,
)

# Execute the strategy
sf = pbl.StrategyFormation(data=data, strategy=strategy)

# Map your column names to PyBondLab expected names in .fit()
result = sf.fit(
    IDvar='cusip',          # Your bond ID column (default: 'ID')
    RETvar='ret_vw',        # Your return column (default: 'ret')
    VWvar='mcap_e',         # Your value weight column (default: 'VW')
    RATINGvar='spc_rat',    # Your rating column (default: 'RATING_NUM')
)

# Get long-short portfolio returns
ew_ls, vw_ls = result.get_long_short()
print(f"Sharpe: {ew_ls.mean() / ew_ls.std() * 12**0.5:.2f}")
```

**Column Mapping:** If your data already uses the default names (`ID`, `ret`, `VW`, `RATING_NUM`), you can simply call `result = sf.fit()` without any parameters. See [Column Mapping](#column-mapping-custom-column-names) for details.

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

### Column Mapping (Custom Column Names)

PyBondLab expects specific column names by default. If your data uses different names, map them using the `.fit()` method parameters.

#### Default Expected Columns

| Internal Name | Default | Description |
|---------------|---------|-------------|
| `ID` | `'ID'` | Bond identifier (e.g., CUSIP) |
| `date` | `'date'` | Date column |
| `ret` | `'ret'` | Return column |
| `VW` | `'VW'` | Value weight (market cap) |
| `RATING_NUM` | `'RATING_NUM'` | Numeric rating (1-22) |
| `PRICE` | `'PRICE'` | Bond price (for price filters) |

#### Mapping Your Column Names

Use the `.fit()` method parameters to map your column names:

```python
import PyBondLab as pbl

# Your data has custom column names
print(data.columns)
# ['cusip', 'date', 'ret_vw', 'mcap_e', 'spc_rat', 'my_signal']

# Create strategy (sort_var uses YOUR column name)
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='my_signal',  # Your signal column name
    num_portfolios=5,
)

# Create StrategyFormation
sf = pbl.StrategyFormation(data=data, strategy=strategy, turnover=True)

# Map columns in .fit()
result = sf.fit(
    IDvar='cusip',           # Maps 'cusip' → 'ID'
    RETvar='ret_vw',         # Maps 'ret_vw' → 'ret'
    VWvar='mcap_e',          # Maps 'mcap_e' → 'VW'
    RATINGvar='spc_rat',     # Maps 'spc_rat' → 'RATING_NUM'
)

# Get results
ew_ls, vw_ls = result.get_long_short()
```

#### .fit() Column Parameters

| Parameter | Maps To | Description |
|-----------|---------|-------------|
| `IDvar` | `'ID'` | Your bond identifier column |
| `DATEvar` | `'date'` | Your date column |
| `RETvar` | `'ret'` | Your return column |
| `VWvar` | `'VW'` | Your value weight column |
| `RATINGvar` | `'RATING_NUM'` | Your numeric rating column |
| `PRICEvar` | `'PRICE'` | Your price column (for price filters) |

**Notes:**
- Only specify parameters for columns with different names
- If your column already has the default name (e.g., `'date'`), no mapping needed
- Signal columns (`sort_var`, `sort_var2`) use your original names - no mapping required

#### DoubleSort Example

```python
# DoubleSort with custom column names
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='my_signal',      # Your first signal column
    sort_var2='my_control',    # Your second signal column
    num_portfolios=3,
    num_portfolios2=3,
    how='unconditional',
)

sf = pbl.StrategyFormation(data=data, strategy=strategy)
result = sf.fit(
    IDvar='cusip',
    RETvar='ret_vw',
    VWvar='mcap_e',
    RATINGvar='spc_rat',
)
```

#### Comparison with BatchStrategyFormation

| Aspect | StrategyFormation | BatchStrategyFormation |
|--------|------------------|------------------------|
| Where | `.fit()` method | Constructor (`columns={}`) |
| Format | Individual params | Dictionary |
| Example | `sf.fit(IDvar='cusip')` | `columns={'ID': 'cusip'}` |

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

### Subset Filter (Characteristic-Based)

Filter the bond universe based on any characteristic column. This is useful for:
- Restricting to bonds with specific maturity ranges
- Filtering by duration, size, or other continuous variables
- Creating custom subsets without modifying your data

#### Basic Usage

```python
from PyBondLab.config import StrategyFormationConfig, DataConfig

# Filter to bonds with maturity 1-5 years
config = StrategyFormationConfig(
    data=DataConfig(
        subset_filter={'MATURITY': (1, 5)},
    ),
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    config=config,
)
result = sf.fit()
```

#### Multiple Filters (AND Logic)

```python
# Filter: maturity 1-5 years AND duration 2-8 years
config = StrategyFormationConfig(
    data=DataConfig(
        subset_filter={
            'MATURITY': (1, 5),
            'DURATION': (2, 8),
        },
    ),
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    config=config,
)
result = sf.fit()
```

#### Combining with Rating Filter

```python
# IG bonds with maturity 1-5 years
config = StrategyFormationConfig(
    data=DataConfig(
        rating='IG',
        subset_filter={'MATURITY': (1, 5)},
    ),
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    config=config,
)
result = sf.fit()

# BBB bonds (rating 7-10) with short duration
config = StrategyFormationConfig(
    data=DataConfig(
        rating=(7, 10),
        subset_filter={'DURATION': (0, 5)},
    ),
)

sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    config=config,
)
result = sf.fit()
```

#### Examples by Use Case

```python
# 1. Short-maturity bonds only
config = StrategyFormationConfig(
    data=DataConfig(subset_filter={'MATURITY': (0, 3)}),
)

# 2. Large bonds only (by market cap / VW)
config = StrategyFormationConfig(
    data=DataConfig(subset_filter={'VW': (1e8, 1e12)}),  # $100M+
)

# 3. Intermediate duration bonds
config = StrategyFormationConfig(
    data=DataConfig(subset_filter={'DURATION': (3, 7)}),
)

# 4. Multiple constraints: IG, short maturity, intermediate duration
config = StrategyFormationConfig(
    data=DataConfig(
        rating='IG',
        subset_filter={
            'MATURITY': (1, 5),
            'DURATION': (2, 6),
        },
    ),
)

# 5. Custom characteristic filter
config = StrategyFormationConfig(
    data=DataConfig(
        subset_filter={
            'spread': (50, 500),      # Spread 50-500 bps
            'coupon': (2.0, 8.0),     # Coupon 2-8%
        },
    ),
)
```

**Key Behaviors:**
- Filters are applied at **formation date only** (no look-ahead bias)
- Bonds excluded from ranking can still contribute returns if they were ranked in prior periods (for staggered holding periods)
- Multiple filters are combined with AND logic
- Filter bounds are inclusive: `(min, max)` means `min <= value <= max`

**Required Columns:**
- The column names in `subset_filter` must exist in your data
- Values must be numeric for comparison

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

## Execution Paths: Slow, Fast, and Ultra-Fast

PyBondLab automatically selects the optimal execution path based on your configuration. Understanding these paths helps you maximize performance.

### Overview of Execution Paths

| Path | Speed | When Used | Limitations |
|------|-------|-----------|-------------|
| **Slow Path** | Baseline | Default, supports all features | Full functionality |
| **Fast Path** | ~2x faster | Returns-only mode | No turnover/chars/banding |
| **Ultra-Fast Path** | ~5x faster | Large panels, returns-only | No turnover/chars/banding, SingleSort only |

### Path Selection Logic

#### StrategyFormation (Single Signal)

```
                    ┌─────────────────────────────────┐
                    │     StrategyFormation.fit()     │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  Can use Ultra-Fast Path?     │
                    │  ALL conditions must be TRUE: │
                    │  • turnover = False           │
                    │  • chars = None               │
                    │  • banding_threshold = None   │
                    │  • Strategy is SingleSort     │
                    │  • rebalance = 'monthly'      │
                    └───────────────┬───────────────┘
                           │                │
                          YES              NO
                           │                │
                           ▼                ▼
                    ┌─────────────┐  ┌─────────────┐
                    │ ULTRA-FAST  │  │    SLOW     │
                    │   PATH      │  │    PATH     │
                    │  (numba)    │  │  (pandas)   │
                    └─────────────┘  └─────────────┘
```

**Ultra-Fast Path Requirements (ALL must be true):**

| Condition | Required Value | Why |
|-----------|---------------|-----|
| `turnover` | `False` | Turnover requires tracking state across periods |
| `chars` | `None` | Characteristics need per-portfolio aggregation |
| `banding_threshold` | `None` | Banding requires lag rank tracking |
| Strategy type | `SingleSort` | DoubleSort has complex interactions |
| `rebalance_frequency` | `'monthly'` | Non-monthly requires special handling |
| `filters` | `None` | Filters require additional data processing |

**What happens when Ultra-Fast Path is used:**
- Bypasses pandas DataFrame operations entirely
- Converts data to numpy arrays once
- Computes ranks for ALL dates in parallel using numba
- Computes returns for ALL dates in parallel using numba
- ~5x speedup for large panels (1M+ rows)

#### BatchStrategyFormation (Multiple Signals)

```
                    ┌─────────────────────────────────┐
                    │  BatchStrategyFormation.fit()   │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  Can use Fast Batch Path?     │
                    │  ALL conditions must be TRUE: │
                    │  • turnover = False           │
                    │  • chars = None               │
                    │  • banding = None             │
                    │  • rating = None              │
                    └───────────────┬───────────────┘
                           │                │
                          YES              NO
                           │                │
                           ▼                ▼
                    ┌─────────────┐  ┌─────────────┐
                    │ FAST BATCH  │  │    SLOW     │
                    │   PATH      │  │    PATH     │
                    │  (numba)    │  │(multiproc)  │
                    └─────────────┘  └─────────────┘
```

**Fast Batch Path Requirements:**

| Condition | Required Value | Why |
|-----------|---------------|-----|
| `turnover` | `False` | No turnover tracking |
| `chars` | `None` | No characteristics |
| `banding` | `None` | No banding |
| `rating` | `None` | No rating filter |

**What happens when Fast Batch Path is used:**
- Processes ALL signals simultaneously using numba kernels
- Computes ranks for all (date × signal) combinations in parallel
- ~2.7-3x speedup compared to multiprocessing slow path
- Avoids Python multiprocessing overhead

### Performance Comparison

#### StrategyFormation (Single Signal)

| Dataset | Slow Path | Ultra-Fast Path | Speedup |
|---------|-----------|-----------------|---------|
| 25K rows (test) | 0.41s | 0.24s | **1.7x** |
| 500K rows | 2.1s | 0.8s | **2.6x** |
| 3M rows | 5.7s | 1.2s | **4.9x** |

#### BatchStrategyFormation (10 Signals)

| Dataset | Slow Path (n_jobs=4) | Fast Batch Path | Speedup |
|---------|---------------------|-----------------|---------|
| 25K rows, HP=1 | 10.4s | 3.8s | **2.7x** |
| 25K rows, HP=3 | 4.3s | 1.5s | **2.9x** |

### How to Ensure Fast Path

#### For StrategyFormation (Single Signal)

```python
# ✅ ULTRA-FAST PATH - All conditions met
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(  # ✅ SingleSort
        holding_period=1,
        sort_var='momentum',
        num_portfolios=5,
        rebalance_frequency='monthly',  # ✅ Monthly (default)
    ),
    turnover=False,           # ✅ No turnover
    # chars not specified     # ✅ No characteristics
    # banding not specified   # ✅ No banding
)
result = sf.fit()  # Uses Ultra-Fast Path
```

```python
# ❌ SLOW PATH - turnover=True disables fast path
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(
        holding_period=1,
        sort_var='momentum',
        num_portfolios=5,
    ),
    turnover=True,  # ❌ Forces slow path
)
```

```python
# ❌ SLOW PATH - DoubleSort disables fast path
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.DoubleSort(  # ❌ Not SingleSort
        holding_period=1,
        sort_var='size',
        sort_var2='value',
        num_portfolios=5,
        num_portfolios2=5,
    ),
    turnover=False,
)
```

```python
# ❌ SLOW PATH - Annual rebalancing disables fast path
sf = pbl.StrategyFormation(
    data=data,
    strategy=pbl.SingleSort(
        holding_period=12,
        sort_var='value',
        num_portfolios=5,
        rebalance_frequency='annual',  # ❌ Not monthly
    ),
    turnover=False,
)
```

#### For BatchStrategyFormation (Multiple Signals)

```python
# ✅ FAST BATCH PATH - All conditions met
batch = pbl.BatchStrategyFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3', ...],
    holding_period=1,
    num_portfolios=5,
    turnover=False,   # ✅ Required
    chars=None,       # ✅ Required (default)
    banding=None,     # ✅ Required (default)
    rating=None,      # ✅ Required (default)
)
results = batch.fit()
# Prints: "FAST BATCH PATH: Processing N signals with numba..."
```

```python
# ❌ SLOW PATH - turnover=True
batch = pbl.BatchStrategyFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3'],
    holding_period=1,
    num_portfolios=5,
    turnover=True,    # ❌ Forces slow path
    n_jobs=4,         # Uses multiprocessing
)
```

```python
# ❌ SLOW PATH - rating filter applied
batch = pbl.BatchStrategyFormation(
    data=data,
    signals=['sig1', 'sig2', 'sig3'],
    holding_period=1,
    num_portfolios=5,
    turnover=False,
    rating='IG',      # ❌ Forces slow path
    n_jobs=4,
)
```

### Checking Which Path is Used

```python
# Enable verbose output to see path selection
sf = pbl.StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=False,
    verbose=True,     # Shows path info
)
result = sf.fit()
# If ultra-fast: prints "Using ULTRA-FAST returns-only path..."
# If slow: prints standard progress info

# For BatchStrategyFormation
batch = pbl.BatchStrategyFormation(
    data=data,
    signals=signals,
    turnover=False,
    verbose=True,
)
results = batch.fit()
# If fast batch: prints "FAST BATCH PATH: Processing N signals with numba..."
# If slow: prints "Processing N signals with M worker(s)..."
```

### Decision Guide

| Your Use Case | Recommended Configuration | Expected Path |
|---------------|--------------------------|---------------|
| Quick factor screening | `turnover=False`, `SingleSort` | Ultra-Fast |
| Batch signal testing | `BatchStrategyFormation`, `turnover=False` | Fast Batch |
| Full analysis with turnover | `turnover=True` | Slow |
| Characteristics tracking | `chars=[...]` | Slow |
| DoubleSort analysis | `DoubleSort` | Slow |
| Rating-filtered analysis | `rating='IG'` or `(min,max)` | Slow |
| Banding for lower turnover | `banding=1` | Slow |

### Best Practices

1. **Two-stage workflow**: Use fast path for screening, slow path for detailed analysis
   ```python
   # Stage 1: Fast screening of 100 signals
   batch = pbl.BatchStrategyFormation(
       data=data, signals=all_signals, turnover=False, ...
   )
   results = batch.fit()  # Fast batch path
   top_signals = get_top_performers(results)

   # Stage 2: Detailed analysis of top 5
   for signal in top_signals:
       sf = pbl.StrategyFormation(
           data=data,
           strategy=pbl.SingleSort(sort_var=signal, ...),
           turnover=True,
           chars=['duration', 'spread'],
       )
       detailed_result = sf.fit()  # Slow path, full analysis
   ```

2. **Pre-compute signals** before running DataUncertaintyAnalysis for 75x speedup

3. **Use `verbose=True`** to confirm which path is being used

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

### Subset Filter Examples

Filter by bond characteristics using `subset_filter`:

```python
from PyBondLab.config import StrategyFormationConfig, DataConfig

# Define a momentum strategy
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='momentum_3m',
    num_portfolios=5,
)

# Example 1: Short-maturity bonds (1-5 years)
config = StrategyFormationConfig(
    data=DataConfig(subset_filter={'MATURITY': (1, 5)}),
)
sf = pbl.StrategyFormation(data=data, strategy=strategy, config=config)
result_short = sf.fit()
ew_short, _ = result_short.get_long_short()

# Example 2: Long-maturity bonds (10+ years)
config = StrategyFormationConfig(
    data=DataConfig(subset_filter={'MATURITY': (10, 30)}),
)
sf = pbl.StrategyFormation(data=data, strategy=strategy, config=config)
result_long = sf.fit()
ew_long, _ = result_long.get_long_short()

# Compare short vs long maturity
print(f"Short maturity mean: {ew_short.mean() * 12:.2%}")
print(f"Long maturity mean:  {ew_long.mean() * 12:.2%}")
```

```python
# Example 3: Intermediate duration, IG bonds
config = StrategyFormationConfig(
    data=DataConfig(
        rating='IG',
        subset_filter={'DURATION': (3, 7)},
    ),
)
sf = pbl.StrategyFormation(data=data, strategy=strategy, config=config)
result = sf.fit()
```

```python
# Example 4: Multiple characteristic filters
# Large bonds ($100M+) with intermediate duration
config = StrategyFormationConfig(
    data=DataConfig(
        subset_filter={
            'VW': (1e8, 1e12),        # Market cap $100M+
            'DURATION': (2, 8),       # Duration 2-8 years
        },
    ),
)
sf = pbl.StrategyFormation(data=data, strategy=strategy, config=config)
result = sf.fit()
```

```python
# Example 5: BBB bonds with short maturity
# Combines rating tuple with subset_filter
config = StrategyFormationConfig(
    data=DataConfig(
        rating=(7, 10),              # BBB only
        subset_filter={
            'MATURITY': (1, 5),      # Short maturity
        },
    ),
)
sf = pbl.StrategyFormation(data=data, strategy=strategy, config=config)
result = sf.fit()
```

```python
# Example 6: Compare maturity buckets across rating categories
import pandas as pd

results = {}
for rating_name, rating in [('IG', 'IG'), ('NIG', 'NIG')]:
    for mat_name, mat_range in [('Short', (0, 5)), ('Int', (5, 10)), ('Long', (10, 30))]:
        key = f"{rating_name}_{mat_name}"

        config = StrategyFormationConfig(
            data=DataConfig(
                rating=rating,
                subset_filter={'MATURITY': mat_range},
            ),
        )
        sf = pbl.StrategyFormation(
            data=data, strategy=strategy, config=config, verbose=False
        )
        result = sf.fit()
        ew_ls, _ = result.get_long_short()

        results[key] = {
            'mean': ew_ls.mean() * 12,
            'sharpe': ew_ls.mean() / ew_ls.std() * 12**0.5,
        }

summary = pd.DataFrame(results).T
print(summary)
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

**5. "Data missing required columns: ['ret']"**

Your data uses custom column names. Map them in `.fit()`:

```python
# Check what columns your data has
print(data.columns.tolist())
# ['cusip', 'date', 'ret_vw', 'mcap_e', 'spc_rat', ...]

# Map your column names to PyBondLab expected names
result = sf.fit(
    IDvar='cusip',           # Your ID column
    RETvar='ret_vw',         # Your return column
    VWvar='mcap_e',          # Your value weight column
    RATINGvar='spc_rat',     # Your rating column
)
```

**6. "Column 'RATING_NUM' not found"**

Map your rating column:

```python
result = sf.fit(RATINGvar='your_rating_column')
```

**7. "Column 'VW' not found"**

Map your value weight (market cap) column:

```python
result = sf.fit(VWvar='your_vw_column')
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
| Subset filter | Via config | Via config |
| Custom breakpoints | Yes | Yes (both sorts) |
| Breakpoint universe | Yes | Yes (both sorts) |

**Rating filter options:**
- `'IG'`: Investment Grade (1-10)
- `'NIG'`: Non-Investment Grade (11-22)
- `(min, max)`: Custom numeric range
- `None`: All bonds

**Subset filter options:**
- `{'MATURITY': (1, 5)}`: Single characteristic
- `{'MATURITY': (1, 5), 'DURATION': (2, 8)}`: Multiple characteristics (AND logic)
- Combined with rating: `rating='IG', subset_filter={'MATURITY': (1, 5)}`

Start with `SingleSort` for simple factor analysis, use `DoubleSort` when you need to control for another variable or study interactions.

---

## Detailed Timing and Mechanics (HP=1)

This section provides a precise specification of how portfolio returns, weights, turnover, and characteristics are computed for `holding_period=1` (monthly rebalancing). Understanding these mechanics is essential for interpreting results correctly.

### Timeline Overview (HP=1)

```
Monthly Rebalancing Timeline (HP=1):
====================================

Month:   0         1         2         3         4         5
         |         |         |         |         |         |
Time: ───┼─────────┼─────────┼─────────┼─────────┼─────────┼───
         │         │         │         │         │         │
         [F₀]──R₀──[F₁]──R₁──[F₂]──R₂──[F₃]──R₃──[F₄]──R₄──[F₅]
               ↑         ↑         ↑         ↑         ↑
            return    return    return    return    return
            at t=1    at t=2    at t=3    at t=4    at t=5

Legend:
  [Fₜ] = Formation date t (portfolio formed, ranks assigned)
  Rₜ   = Return over period t → t+1 (collected at t+1)
```

**Key Insight for HP=1:**
- Each formation date t creates a portfolio held for exactly 1 month
- Return over period [t, t+1] is collected at date t+1
- Only one cohort exists (cohort 0), so no cohort averaging is needed
- Results are indexed by **return date (t+1)**, not formation date (t)

---

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FORMATION DATE t                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT DATA:                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │     It0      │    │    It1m      │    │  ranks_map   │              │
│  │  (date = t)  │    │  (date = t)  │    │  (date = t)  │              │
│  │              │    │              │    │              │              │
│  │ - Bond IDs   │    │ - Bond IDs   │    │ - Bond → Ptf │              │
│  │ - Signal     │    │ - VW         │    │   rank map   │              │
│  │ - VW         │    │ - Chars      │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                        RETURN DATE t+1                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT DATA:                                                            │
│  ┌──────────────┐                                                       │
│  │     It1      │                                                       │
│  │ (date = t+1) │                                                       │
│  │              │                                                       │
│  │ - Bond IDs   │                                                       │
│  │ - Returns    │                                                       │
│  │ - VW         │                                                       │
│  └──────────────┘                                                       │
│                                                                         │
│  PROCESSING (intersect_id):                                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Intersection = Bonds in BOTH It0 AND It1                          │  │
│  │                                                                   │  │
│  │ If dynamic_weights=True:                                          │  │
│  │   Also require bond in It1m (for VW at t)                         │  │
│  │                                                                   │  │
│  │ Result: Only bonds present at BOTH formation (t) and return (t+1) │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  COMPUTATION:                                                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │    Weights    │  │   Returns     │  │    Chars      │               │
│  │               │  │               │  │               │               │
│  │ w_raw[i]      │  │ R_p = Σ w×r   │  │ C_p = Σ w×c   │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
│          │                  │                  │                        │
│          │                  │                  │                        │
│          ▼                  ▼                  ▼                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   OUTPUTS (indexed at t+1)                        │  │
│  │                                                                   │  │
│  │  - ew_ret_arr[t+1]     = EW portfolio returns                     │  │
│  │  - vw_ret_arr[t+1]     = VW portfolio returns                     │  │
│  │  - chars_arr[t+1]      = Portfolio characteristics                │  │
│  │  - port_idx[t+1]       = Bond weights (for turnover)              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  TURNOVER (computed at formation time τ = t):                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  turnover[τ] = compare w_raw(t+1) vs w_scaled(t)                  │  │
│  │                                                                   │  │
│  │  Note: Turnover at τ measures weight changes from the PREVIOUS   │  │
│  │        period's scaled weights to CURRENT period's raw weights   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Weight Computation

#### Raw Weights (w_raw)

Raw weights are computed at each formation date for bonds in the intersection:

**Equal Weights (EW):**
```
w_ew[i] = 1 / n_p

where:
  n_p = number of bonds in portfolio p

Note: All bonds in portfolio p have equal weight summing to 1.
```

**Value Weights (VW):**
```
              VW_t[i]
w_vw[i] = ─────────────
           Σⱼ∈p VW_t[j]

where:
  VW_t[i] = Value weight of bond i at formation date t
  p = portfolio containing bond i
  Σⱼ∈p = sum over all bonds j in portfolio p

Note: Weights sum to 1 within each portfolio.
```

**VW Source Date for HP=1:**

For `holding_period=1`, the VW source date is the same regardless of `dynamic_weights`:

| Setting | VW Source | Reasoning |
|---------|-----------|-----------|
| `dynamic_weights=True` | t+1-1 = t (formation date) | Use VW from "day before return" |
| `dynamic_weights=False` | t (formation date) | Use VW from formation date |

**Important:** For HP=1, both settings produce identical VW because the "day before return" (t+1-1) equals the formation date (t). The distinction only matters for HP>1.

---

### Return Computation

Portfolio returns are computed for each portfolio p at return date t+1:

**Equal-Weighted Return:**
```
                1
R_ew,p(t+1) = ───── × Σᵢ∈p r_i(t+1)
               n_p

            = mean(r_i) for bonds i in portfolio p
```

**Value-Weighted Return:**
```
R_vw,p(t+1) = Σᵢ∈p w_vw[i] × r_i(t+1)

where:
  w_vw[i] = VW weight (computed at formation date t)
  r_i(t+1) = Return of bond i over period [t, t+1]
```

**Return Indexing:**
- Returns are stored at index `t+1` (the return date)
- This matches the convention that `R(t+1)` is the return realized at date t+1

---

### Scaled Weights (for Turnover)

After computing returns, scaled weights are calculated for turnover tracking:

```
                      w_raw[i] × (1 + r_i(t+1))
w_scaled[i](t+1) = ──────────────────────────────
                        (1 + R_p(t+1))

where:
  w_raw[i] = Raw weight at formation
  r_i(t+1) = Bond i's return over [t, t+1]
  R_p(t+1) = Portfolio p's return over [t, t+1]
```

**Purpose:** Scaled weights represent what the portfolio weights would be at the END of the holding period if no rebalancing occurred. They are compared with the NEXT period's raw weights to compute turnover.

**Important:** Returns are computed with `w_raw`, NOT `w_scaled`. Scaled weights are only used for turnover computation.

---

### Turnover Computation

Turnover measures how much the portfolio composition changes between periods.

**Turnover Formula:**
```
T_p(t+1) = ½ × (prev_sum + curr_sum - 2 × sum_min)

where:
  prev_sum = Σ w_scaled[i](t)   [sum of previous period's scaled weights]
  curr_sum = Σ w_raw[i](t+1)    [sum of current period's raw weights]
  sum_min  = Σ min(w_scaled[i](t), w_raw[i](t+1))  [for matching bonds]
```

**Turnover Timeline for HP=1:**

```
Turnover Timeline:
==================

Formation:  τ=0       τ=1       τ=2       τ=3       τ=4
            │         │         │         │         │
Time:    ───┼─────────┼─────────┼─────────┼─────────┼───
            │         │         │         │         │
        [Form₀]   [Form₁]   [Form₂]   [Form₃]   [Form₄]
            │         │         │         │         │
Weights:  w_raw₀    w_raw₁    w_raw₂    w_raw₃    w_raw₄
            │         │         │         │         │
After     w_scaled₀ w_scaled₁ w_scaled₂ w_scaled₃ w_scaled₄
returns:    │         │         │         │         │
            │         │         │         │         │
Turnover:  NaN*     T(τ=1)    T(τ=2)    T(τ=3)    T(τ=4)
                    compare:  compare:  compare:  compare:
                    w_raw₁    w_raw₂    w_raw₃    w_raw₄
                    vs        vs        vs        vs
                    w_scaled₀ w_scaled₁ w_scaled₂ w_scaled₃

* First formation has no previous period → NaN turnover
```

**First Period Handling:**
- At τ=0, there is no previous period's weights
- `prev_seen=False`, so turnover is NaN
- After τ=0, `prev_seen=True` and turnover is computed

**Last Period Liquidation:**
- At the final date τ_last, assume complete liquidation
- Turnover = prev_sum (all positions sold, no new positions)

**Turnover Indexing:**
- Turnover is computed at formation time τ
- Stored at index τ in the turnover array
- The value T(τ) reflects weight changes when rebalancing at time τ

---

### Characteristics Computation

Portfolio-level characteristics are aggregated from bond-level data:

**Data Source:**
```
Characteristics come from It1m (data at formation date t, NOT return date t+1)
```

**Equal-Weighted Characteristic:**
```
                1
C_ew,p = ───── × Σᵢ∈p char[i]
          n_p

       = mean(char[i]) for bonds i in portfolio p
```

**Value-Weighted Characteristic:**
```
C_vw,p = Σᵢ∈p w_vw[i] × char[i]

where:
  w_vw[i] = VW weight (from return computation)
  char[i] = Characteristic value at formation date
```

**Characteristic Indexing:**
- Like returns, characteristics are indexed by return date t+1
- However, the VALUES come from formation date t
- This maintains alignment: chars at formation → return at realization

---

### Complete Example (HP=1)

```
Example: 5 Portfolios, Formation at t=2, Return at t=3
======================================================

STEP 1: Formation Date (t=2)
----------------------------
It0 (signal data at t=2):
  Bond A: signal=0.95 (high)
  Bond B: signal=0.85 (high)
  Bond C: signal=0.45 (mid)
  Bond D: signal=0.15 (low)
  Bond E: signal=0.05 (low)

Ranking: Assign to quintiles based on signal percentiles
  P1 (low):  Bond D, E
  P2:        (empty in this example)
  P3 (mid):  Bond C
  P4:        (empty in this example)
  P5 (high): Bond A, B

STEP 2: Intersection Check (t=2 → t=3)
--------------------------------------
It1 (return data at t=3):
  Bond A: ret=2.0%, VW=100M   ✓ in intersection
  Bond B: ret=1.5%, VW=50M    ✓ in intersection
  Bond C: ret=0.5%, VW=80M    ✓ in intersection
  Bond D: ret=-1.0%, VW=30M   ✓ in intersection
  Bond E: MISSING             ✗ dropped from intersection

After intersection:
  P1: Bond D only (E dropped)
  P5: Bond A, B

STEP 3: Weight Computation
--------------------------
P5 (high quintile):
  n_p = 2 (bonds A, B)

  EW weights:
    w_ew[A] = 1/2 = 0.50
    w_ew[B] = 1/2 = 0.50

  VW weights (VW from t=2):
    VW_A = 100M, VW_B = 50M, total = 150M
    w_vw[A] = 100/150 = 0.667
    w_vw[B] = 50/150  = 0.333

STEP 4: Return Computation
--------------------------
P5 returns at t=3:
  R_ew,P5 = 0.50 × 2.0% + 0.50 × 1.5% = 1.75%
  R_vw,P5 = 0.667 × 2.0% + 0.333 × 1.5% = 1.833%

STEP 5: Scaled Weight Computation (for turnover)
------------------------------------------------
P5 scaled weights (end of t=3):
  w_scaled[A] = 0.667 × (1 + 0.02) / (1 + 0.01833) = 0.668
  w_scaled[B] = 0.333 × (1 + 0.015) / (1 + 0.01833) = 0.332

STEP 6: Output Indexing
-----------------------
All stored at index t+1 = 3:
  ew_ret_arr[3, P5] = 1.75%
  vw_ret_arr[3, P5] = 1.833%
  port_idx[3] = {A: {rank=5, ew=0.50, vw=0.667}, B: {rank=5, ew=0.50, vw=0.333}}

Turnover stored at τ=2 (formation time):
  Compare w_raw(t=3) vs w_scaled(t=2)
```

---

### ID Intersection Logic

A critical step is the intersection of bond IDs across dates:

```python
# From utils.py: intersect_id()
def intersect_id(It0, It1, It1m, dynamic_weights):
    """
    Find bonds present at both formation and return dates.

    Parameters
    ----------
    It0 : DataFrame
        Signal data at formation date t
    It1 : DataFrame
        Return data at return date t+1
    It1m : DataFrame
        VW data at VW source date (t for HP=1)
    dynamic_weights : bool
        If True, also require bond in It1m

    Returns
    -------
    Filtered It0, It1, It1m with only common bonds
    """
    common = set(It0['ID']) & set(It1['ID'])
    if dynamic_weights:
        common &= set(It1m['ID'])

    It0 = It0[It0['ID'].isin(common)]
    It1 = It1[It1['ID'].isin(common)]
    It1m = It1m[It1m['ID'].isin(common)]

    return It0, It1, It1m
```

**Why Intersection Matters:**
- A bond ranked at formation may not have return data at realization
- Including such bonds would bias returns (missing returns ≠ zero returns)
- The intersection ensures we only use bonds with valid data at BOTH dates

---

### Cohort Handling for HP=1

For `holding_period=1`, the cohort dimension is trivial:

```
Cohort Index = t % holding_period = t % 1 = 0 (always)

Therefore:
  - Only cohort 0 exists
  - No cohort averaging is performed
  - Results shape: (n_dates,) not (n_dates, n_cohorts)
```

**Aggregation for HP=1:**
```python
# No averaging needed - direct assignment
ew_returns_final = ew_ret_arr[:, 0, :]  # Just take cohort 0
vw_returns_final = vw_ret_arr[:, 0, :]  # Just take cohort 0
```

---

### Summary Table (HP=1)

| Output | Indexed At | Computed From | VW Source |
|--------|------------|---------------|-----------|
| Portfolio Returns | t+1 (return date) | r(t+1), w_raw(t) | t (formation) |
| Portfolio Weights (port_idx) | t+1 (return date) | VW(t), ranks(t) | t (formation) |
| Scaled Weights | t+1 (return date) | w_raw(t+1), r(t+1), R_p(t+1) | N/A |
| Turnover | τ=t (formation time) | w_raw(t+1) vs w_scaled(t) | N/A |
| Characteristics | t+1 (return date) | char(t), w(t) | t (formation) |

**Key Points:**
1. Returns and chars use data from formation (t) but are indexed at return date (t+1)
2. Turnover compares consecutive formations and is indexed at formation time (τ)
3. For HP=1, `dynamic_weights=True` and `=False` produce identical results
4. Only bonds in the intersection of formation and return dates are included

---

## Detailed Timing and Mechanics (HP=3, Staggered Rebalancing)

This section provides a precise specification of how portfolio returns, weights, turnover, and characteristics are computed for `holding_period=3` (staggered rebalancing). The key distinction from HP=1 is the presence of **overlapping cohorts**.

### What is Staggered Rebalancing?

With `holding_period > 1`, portfolios are rebalanced in a staggered fashion:
- **3 overlapping cohorts** exist simultaneously (for HP=3)
- Each month, **1 cohort rebalances** while **2 cohorts hold** their existing positions
- Final returns are **averaged across all active cohorts** at each date

**Why stagger?** Staggering reduces the impact of formation timing on factor returns and provides smoother portfolio transitions.

---

### Timeline Overview (HP=3)

```
Staggered Rebalancing Timeline (HP=3):
======================================

Month:    0         1         2         3         4         5         6         7         8
          |         |         |         |         |         |         |         |         |
Time:  ───┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼───
          │         │         │         │         │         │         │         │         │
Cohort 0: [F₀]─────R₀₁───────R₀₂───────R₀₃       [F₃]─────R₃₁───────R₃₂───────R₃₃
              held 1     held 2     held 3           held 1     held 2     held 3
          │         │         │         │         │         │         │         │         │
Cohort 1:           [F₁]─────R₁₁───────R₁₂───────R₁₃       [F₄]─────R₄₁───────R₄₂───────R₄₃
                        held 1     held 2     held 3           held 1     held 2     held 3
          │         │         │         │         │         │         │         │         │
Cohort 2:                     [F₂]─────R₂₁───────R₂₂───────R₂₃       [F₅]─────R₅₁───────R₅₂
                                  held 1     held 2     held 3           held 1     held 2

Final:    NaN       NaN       NaN      R̄₃        R̄₄        R̄₅        R̄₆        R̄₇        R̄₈
Return                              =avg(R₀₃, =avg(R₀₃, =avg(R₁₃, =avg(R₃₁, =avg(R₃₂, =avg(R₃₃,
                                     R₁₂,      R₁₃,      R₂₃,      R₄₁,      R₄₂,      R₄₃,
                                     R₂₁)      R₂₂)      R₃₁)      R₅₁)      R₅₂)      R₅₃)

Legend:
  [Fₜ] = Formation date t (cohort formed, ranks assigned)
  Rₜₕ  = Return for cohort formed at t, holding period h (h = 1, 2, or 3)
  R̄ₜ   = Final return at date t = average across active cohorts
```

**Key Insight:**
- At any given return date, up to 3 cohorts contribute returns (one at each holding horizon)
- Cohort index = `formation_date % 3`
- The first non-NaN final return is at t=3 (first time all 3 cohorts have contributed)

---

### Cohort Assignment

Each formation date is assigned to a cohort based on modulo arithmetic:

```
Cohort = formation_date_index % holding_period

Example for HP=3:
  Formation date 0 → Cohort 0 (0 % 3 = 0)
  Formation date 1 → Cohort 1 (1 % 3 = 1)
  Formation date 2 → Cohort 2 (2 % 3 = 2)
  Formation date 3 → Cohort 0 (3 % 3 = 0)  ← Cohort 0 rebalances
  Formation date 4 → Cohort 1 (4 % 3 = 1)  ← Cohort 1 rebalances
  Formation date 5 → Cohort 2 (5 % 3 = 2)  ← Cohort 2 rebalances
  ...
```

---

### Holding Period Loop

For each formation date t (assigned to cohort c), we compute returns for h = 1, 2, ..., HP:

```
Formation date: t
Cohort:        c = t % HP

For h in range(HP):  # h = 0, 1, 2 for HP=3
    return_date = t + h + 1

    # Compute returns for this (formation, return_date) pair
    R[return_date, cohort, portfolio] = compute_return(...)

Example for t=1, HP=3, c=1:
  h=0: return_date = 1+0+1 = 2  → Store at ret_arr[2, 1, :]
  h=1: return_date = 1+1+1 = 3  → Store at ret_arr[3, 1, :]
  h=2: return_date = 1+2+1 = 4  → Store at ret_arr[4, 1, :]
```

---

### Results Array Structure

```
Array Shape: (n_dates, n_cohorts, n_portfolios) = (TM, HP, nport)

For HP=3 and 5 portfolios:
  ew_ret_arr.shape = (TM, 3, 5)

  ret_arr[t, c, p] = Return at date t for cohort c, portfolio p

  Example contents at return date t=5:
    ret_arr[5, 0, :] = Returns from cohort 0 (formed at t=3, held for 2 months)
    ret_arr[5, 1, :] = Returns from cohort 1 (formed at t=4, held for 1 month)
    ret_arr[5, 2, :] = Returns from cohort 2 (formed at t=2, held for 3 months)
```

---

### dynamic_weights Parameter (Critical for HP>1)

For `holding_period > 1`, the `dynamic_weights` parameter controls which date's VW is used:

| Setting | VW Source Date | Description |
|---------|---------------|-------------|
| `dynamic_weights=True` | t+h = return_date - 1 | VW from "day before return" |
| `dynamic_weights=False` | t = formation_date | VW from when portfolio was formed |

**Why This Matters:**

For HP=3, cohort formed at t=0 has returns at t=1, t=2, and t=3:

```
Cohort formed at t=0:
==============================

                                        dynamic_weights=True   dynamic_weights=False
                                        ─────────────────────  ──────────────────────
Return at t=1 (h=0): Uses VW from date  t+0 = 0                0
Return at t=2 (h=1): Uses VW from date  t+1 = 1                0
Return at t=3 (h=2): Uses VW from date  t+2 = 2                0

With dynamic_weights=True:
  - VW is updated each month based on most recent available data
  - Bonds that grew in value get higher VW weight

With dynamic_weights=False:
  - VW is fixed at formation date
  - Portfolio composition is "frozen" at formation
```

**Code Reference:**
```python
# In _form_cohort_portfolios():
date_t1_minus1 = None
if self.dynamic_weights and t1_idx > 0:
    date_t1_minus1 = self.datelist[t1_idx - 1]  # VW from return_date - 1

# If dynamic_weights=False, date_t1_minus1 stays None
# → VW comes from formation date (date_t)
```

---

### Cohort Averaging

Final returns are computed by averaging across all active cohorts:

```python
# Aggregation
ew_ret_final = np.nanmean(ew_ret_arr, axis=1)  # Average over cohort dimension

# Result shape: (TM, nport) - cohort dimension is collapsed
```

**Example at return date t=5:**
```
ret_arr[5, 0, p] = R from cohort 0 (formation=3, h=2)
ret_arr[5, 1, p] = R from cohort 1 (formation=4, h=1)
ret_arr[5, 2, p] = R from cohort 2 (formation=2, h=3)

final_ret[5, p] = (ret_arr[5, 0, p] + ret_arr[5, 1, p] + ret_arr[5, 2, p]) / 3
```

**NaN Handling:**
- During initialization (first HP-1 dates), not all cohorts have returns
- `np.nanmean` ignores NaN values, averaging only available cohorts
- First complete average is at t = HP (when all cohorts have contributed)

---

### Weight Computation for HP>1

Same formulas as HP=1, but VW source date differs based on `dynamic_weights`:

**Equal Weights (EW):**
```
w_ew[i] = 1 / n_p  (same as HP=1)
```

**Value Weights (VW):**
```
              VW_source[i]
w_vw[i] = ─────────────────────
           Σⱼ∈p VW_source[j]

where VW_source depends on dynamic_weights:
  - dynamic_weights=True:  VW from date (return_date - 1) = t + h
  - dynamic_weights=False: VW from formation date t
```

---

### Turnover for HP>1 (Staggered)

Turnover is tracked **per cohort** and then averaged:

**Per-Cohort Tracking:**
```
turnover_arr.shape = (TM, HP, nport)

At each formation date τ:
  - Cohort c = τ % HP rebalances → compute turnover from scaled weights
  - Other cohorts (c ≠ τ % HP) are holding → turnover = 0.0 (NOT NaN)
```

**Cohort States:**
| State | Turnover Value | Meaning |
|-------|---------------|---------|
| Not yet formed | NaN | Cohort doesn't exist |
| Holding (not rebalancing) | 0.0 | Cohort exists but isn't trading |
| Rebalancing | Computed value | Active weight changes |

**Timeline Example (HP=3):**
```
Turnover per Cohort:
====================

Formation τ:  0      1      2      3      4      5      6
              │      │      │      │      │      │      │
Cohort 0:    NaN*  hold=0 hold=0  T₃    hold=0 hold=0  T₆
Cohort 1:    NaN*   NaN*  hold=0 hold=0  T₄    hold=0 hold=0
Cohort 2:    NaN*   NaN*   NaN*  hold=0 hold=0  T₅    hold=0

Final avg:   NaN    NaN    NaN   T̄₃     T̄₄     T̄₅     T̄₆
                             =avg  =avg  =avg  =avg
                             (T₃,  (T₄,  (T₅,  (T₆,
                              0,0)  0,0)  0,0)  0,0)

* First time seeing cohort → NaN
```

**Turnover Formula (same as HP=1):**
```
T_c,p(τ) = ½ × (prev_sum_c + curr_sum_c - 2 × sum_min_c)

where subscript c indicates cohort-specific state
```

**Final Turnover:**
```python
# Average turnover across cohorts (ignoring NaN for non-existent cohorts)
turnover_final = np.nanmean(turnover_arr, axis=1)

# Typical result for HP=3:
# turnover_final[τ] ≈ (1/3) × T_c(τ) + (2/3) × 0.0 = T_c(τ) / 3
# Because only 1 cohort rebalances, 2 cohorts hold (turnover=0)
```

---

### Characteristics for HP>1

Characteristics are aggregated exactly like returns:

```
chars_arr.shape = (TM, HP, nport)

# Per cohort: compute chars at each (formation, return_date) pair
# Final: average across cohorts
chars_final = np.nanmean(chars_arr, axis=1)
```

**Data Source:**
- Characteristics come from **formation date** (It1m at date_t)
- NOT from the return date
- This is consistent with HP=1 behavior

---

### Complete Example (HP=3)

```
Example: HP=3, 5 Portfolios, dynamic_weights=True
=================================================

Formation at t=1 (Cohort 1):
----------------------------
  Signal data at t=1 → Assign bonds to portfolios P1-P5
  Ranks fixed for this cohort for 3 months

  Holding Period 1 (h=0):
    Return date = t + 1 = 2
    VW source = return_date - 1 = 1 (formation date, same as h=0)
    Compute: ret_arr[2, 1, :] = portfolio returns

  Holding Period 2 (h=1):
    Return date = t + 2 = 3
    VW source = return_date - 1 = 2
    Note: VW updated to date 2 (more recent than formation!)
    Compute: ret_arr[3, 1, :] = portfolio returns

  Holding Period 3 (h=2):
    Return date = t + 3 = 4
    VW source = return_date - 1 = 3
    Note: VW updated to date 3
    Compute: ret_arr[4, 1, :] = portfolio returns

What happens at return date t=4:
--------------------------------
  ret_arr[4, 0, :] = Cohort 0, formed at t=3, holding for 1 month (h=0)
  ret_arr[4, 1, :] = Cohort 1, formed at t=1, holding for 3 months (h=2)
  ret_arr[4, 2, :] = Cohort 2, formed at t=2, holding for 2 months (h=1)

  final_ret[4, :] = average of the three cohort returns
```

---

### ID Intersection for HP>1

Same logic as HP=1, applied at each (formation, return_date) pair:

```python
# For each holding period h:
return_date = formation_date + h + 1

# Intersection:
common = set(It0['ID']) & set(It1['ID'])  # formation ∩ return

if dynamic_weights:
    # Also need VW at (return_date - 1)
    common &= set(It1m['ID'])
```

**Important:** The intersection is computed **independently for each (formation, holding_h) pair**. A bond may:
- Be included at h=0 (exists at formation and return_date_1)
- Be excluded at h=1 (dropped by return_date_2)
- Be included at h=2 (returns by return_date_3)

---

### Summary Table (HP=3)

| Output | Indexed At | Computed From | VW Source | Cohort Handling |
|--------|------------|---------------|-----------|-----------------|
| Per-Cohort Returns | [return_date, cohort, ptf] | r(return_date), w(formation or VW_date) | dynamic_weights dependent | One value per cohort |
| Final Returns | [return_date, ptf] | nanmean across cohorts | N/A | Average of 3 cohorts |
| Per-Cohort Turnover | [formation_τ, cohort, ptf] | w_raw(τ+1) vs w_scaled(τ) | N/A | Rebalancing cohort only |
| Final Turnover | [formation_τ, ptf] | nanmean across cohorts | N/A | ~1/3 of rebalancing turnover |
| Per-Cohort Chars | [return_date, cohort, ptf] | char(formation), w(formation or VW_date) | dynamic_weights dependent | One value per cohort |
| Final Chars | [return_date, ptf] | nanmean across cohorts | N/A | Average of 3 cohorts |

---

### Key Differences: HP=1 vs HP=3

| Aspect | HP=1 | HP=3 |
|--------|------|------|
| Cohorts | 1 (trivial) | 3 (overlapping) |
| Cohort averaging | None needed | nanmean across axis=1 |
| `dynamic_weights` effect | None (both same) | Significant (VW date differs by h) |
| Turnover per period | All cohorts rebalance | 1 rebalances, 2 hold (turnover=0) |
| First valid return | t=1 | t=3 (first complete average) |
| Rebalancing frequency | Every month | Each cohort every 3 months |

---

### Turnover Interpretation for HP>1

For HP=3, the final turnover is approximately **1/3 of a single cohort's turnover**:

```
At any formation τ:
  - 1 cohort rebalances: turnover = T_c
  - 2 cohorts hold: turnover = 0.0 each

  final_turnover[τ] = (T_c + 0 + 0) / 3 = T_c / 3
```

**Interpretation:**
- Final turnover reflects the **fraction of the portfolio being traded**
- With HP=3, only 1/3 of the portfolio is rebalanced each month
- This is the **expected reduction in turnover** from staggered rebalancing

**To get "full portfolio" turnover:**
```
full_portfolio_turnover = final_turnover × holding_period
                        = final_turnover × 3
```

---

## The dynamic_weights Parameter

The `dynamic_weights` parameter controls **which date's value weights (VW) are used** for computing VW portfolio returns and characteristics. This section provides a complete specification of its behavior.

---

### What dynamic_weights Controls

`dynamic_weights` has **two effects**:

1. **VW Source Date**: Where the value weights come from
2. **ID Intersection**: Which bonds are included in the return calculation

| Setting | VW Source Date | ID Intersection |
|---------|---------------|-----------------|
| `True` | return_date - 1 (d-1) | 3-way: formation ∩ return ∩ VW_date |
| `False` | formation_date (t) | 2-way: formation ∩ return |

---

### Effect on HP=1 (No Effect)

**For `holding_period=1`, both settings produce IDENTICAL results.**

```
HP=1 Timeline:
==============

Formation:  t=0       t=1       t=2       t=3
            │         │         │         │
Time:    ───┼─────────┼─────────┼─────────┼───
            │         │         │         │
        [Form₀]    [Form₁]    [Form₂]    [Form₃]
            │         │         │         │
Return:           R₀→₁      R₁→₂      R₂→₃
                  at t=1    at t=2    at t=3

For return at t=1 (formed at t=0):
  dynamic_weights=True:  VW from return_date - 1 = 1 - 1 = 0 = formation_date
  dynamic_weights=False: VW from formation_date = 0

→ SAME DATE! Both use VW from t=0.
```

**Mathematical Proof:**

```
For HP=1:
  return_date = formation_date + 1

dynamic_weights=True:
  VW_date = return_date - 1 = (formation_date + 1) - 1 = formation_date

dynamic_weights=False:
  VW_date = formation_date

→ Both settings use VW from formation_date when HP=1.
```

**Conclusion:** For HP=1, `dynamic_weights` has **no effect** on results.

---

### Effect on HP>1 (Critical Difference)

**For `holding_period > 1`, the settings produce DIFFERENT results.**

The difference arises because cohorts held for multiple months use different VW dates:

```
HP=3, Formation at t=0 (Cohort 0):
==================================

                                          dynamic_weights
                                          ───────────────────────────
Return date:   t=1       t=2       t=3    True          False
               │         │         │      ────          ─────
Holding h:    h=0       h=1       h=2
               │         │         │
VW Source:   d-1=0     d-1=1     d-1=2    VW varies     VW fixed at t=0
               │         │         │      per return    (formation)
               ▼         ▼         ▼      date
            [VW at 0] [VW at 1] [VW at 2]
               ↓         ↓         ↓
            Same as    DIFFERS    DIFFERS   ← Key difference!
            False      from       from
                       False      False
```

---

### VW Source Date Equations

**For a cohort formed at date `t` with return at date `d`:**

```
dynamic_weights=True:
  VW_date = d - 1  (day before return date)

  This is FIXED for all cohorts at a given return date.
  At return date d=5: VW_date = 4 for ALL cohorts.

dynamic_weights=False:
  VW_date = t  (formation date)

  This VARIES by cohort.
  At return date d=5:
    Cohort formed at t=2: VW_date = 2
    Cohort formed at t=3: VW_date = 3
    Cohort formed at t=4: VW_date = 4
```

**General Formula:**

```
           ┌ d - 1         if dynamic_weights = True
VW_date =  │
           └ t             if dynamic_weights = False

where:
  d = return date
  t = formation date for the cohort
```

---

### ID Intersection Behavior

The `dynamic_weights` parameter also affects **which bonds are included**:

```
intersect_id(It0, It1, It1m, dynamic_weights):
    """
    It0  = Bonds at formation date (t)
    It1  = Bonds at return date (d)
    It1m = Bonds at VW source date
    """

    common_ids = It0.ID ∩ It1.ID

    if dynamic_weights:
        common_ids = common_ids ∩ It1m.ID  # 3-way intersection

    return filtered(It0, It1, It1m, common_ids)
```

**Practical Impact:**

| Setting | Bond Requirement | Effect |
|---------|-----------------|--------|
| `True` | Must exist at formation AND return AND VW_date | More restrictive; excludes bonds missing at d-1 |
| `False` | Must exist at formation AND return | Less restrictive; includes more bonds |

**Example:**
```
Bond X timeline:
  - Exists at formation (t=0): YES
  - Exists at return (t=3): YES
  - Exists at VW_date (t=2): NO (dropped temporarily)

With dynamic_weights=True:  Bond X EXCLUDED (missing at t=2)
With dynamic_weights=False: Bond X INCLUDED (VW from t=0)
```

---

### Timeline Diagram: dynamic_weights=True vs False

```
HP=3 with dynamic_weights Comparison:
=====================================

Return date:        t=3         t=4         t=5         t=6
                     │           │           │           │
                     ▼           ▼           ▼           ▼

Cohort 0 (form t=0):
  Holding:          h=2         -           -           -
  VW (True):       d-1=2        -           -           -
  VW (False):      t=0          -           -           -

Cohort 1 (form t=1):
  Holding:          h=1        h=2          -           -
  VW (True):       d-1=2      d-1=3         -           -
  VW (False):      t=1        t=1           -           -

Cohort 2 (form t=2):
  Holding:          h=0        h=1         h=2          -
  VW (True):       d-1=2      d-1=3       d-1=4         -
  VW (False):      t=2        t=2         t=2           -

Cohort 0 (form t=3):
  Holding:           -         h=0         h=1         h=2
  VW (True):         -        d-1=3       d-1=4       d-1=5
  VW (False):        -        t=3         t=3         t=3

                    ↑
                    │
At return date t=3, dynamic_weights=True uses VW from t=2 for ALL cohorts
                    dynamic_weights=False uses VW from t=0, t=1, t=2 (varies by cohort)
```

**Key Insight:**
- `dynamic_weights=True`: All cohorts at a return date use the SAME VW date (d-1)
- `dynamic_weights=False`: Each cohort uses VW from its OWN formation date

---

### Weight Computation Equations (dynamic_weights)

**VW Weight Formula:**

```
                    VW[i, VW_date]
w_vw[i] = ────────────────────────────────
           Σⱼ∈portfolio VW[j, VW_date]

where:
         ┌ d - 1    if dynamic_weights = True
VW_date = │
         └ t        if dynamic_weights = False
```

**Detailed Example (HP=3, return at d=5):**

```
Cohort formed at t=2, return at d=5 (h=2):
==========================================

Bonds in portfolio: A, B, C

dynamic_weights=True (VW from d-1=4):
  VW_A(4) = 100M,  VW_B(4) = 80M,  VW_C(4) = 120M
  Total = 300M

  w_vw[A] = 100/300 = 0.333
  w_vw[B] = 80/300  = 0.267
  w_vw[C] = 120/300 = 0.400

dynamic_weights=False (VW from t=2):
  VW_A(2) = 90M,   VW_B(2) = 85M,  VW_C(2) = 100M
  Total = 275M

  w_vw[A] = 90/275  = 0.327
  w_vw[B] = 85/275  = 0.309
  w_vw[C] = 100/275 = 0.364

→ Different weights → Different VW returns!
```

---

### Effect on Turnover

**Turnover is NOT directly affected by `dynamic_weights`.**

Turnover compares raw weights at formation τ+1 vs scaled weights at formation τ:

```
T(τ+1) = ½ × |w_raw(τ+1) - w_scaled(τ)|
```

The `w_raw` weights at each formation use VW from that formation date.

**However, `dynamic_weights` indirectly affects turnover:**
- Different VW dates → Different bond sets (via ID intersection)
- Different bond sets → Different weight compositions
- Different compositions → Different turnover values

---

### Effect on Returns

**VW Returns ARE affected by `dynamic_weights` for HP>1.**

```
VW Return (portfolio p):
  R_vw,p = Σᵢ∈p w_vw[i] × r[i]

Since w_vw[i] depends on VW_date:
  dynamic_weights=True:  Weights reflect RECENT VW (bonds that grew are weighted more)
  dynamic_weights=False: Weights reflect ORIGINAL VW (weights frozen at formation)
```

**Interpretation:**

| Setting | Meaning | Use Case |
|---------|---------|----------|
| `True` | "Dynamic VW" - weights adjust monthly based on recent VW | Realistic portfolio simulation |
| `False` | "Static VW" - weights frozen at formation | Pure factor exposure analysis |

---

### Defaults Across Classes

**IMPORTANT: Different classes have different defaults!**

| Class | Default | Source | Notes |
|-------|---------|--------|-------|
| `FormationConfig` | `False` | config.py:277 | Base default |
| `StrategyFormation` | Uses config | (FormationConfig) | Defaults to `False` |
| `BatchStrategyFormation` | **`True` (hardcoded)** | batch.py:128,192 | Forces `True` |
| `DataUncertaintyAnalysis` | **`True` (default param)** | data_uncertainty.py:635 | Default `True`, can change |
| `pbl_test.py` (baseline tests) | **`True` (hardcoded)** | pbl_test.py:484 | Baseline uses `True` |
| `SingleSort` / `DoubleSort` | N/A | - | No parameter (uses config) |

**Critical Warning:**

```python
# These produce DIFFERENT results for HP>1:

# Method 1: Direct StrategyFormation (default = False)
sf1 = StrategyFormation(data, strategy, turnover=True)  # dynamic_weights=False

# Method 2: BatchStrategyFormation (hardcoded True)
batch = BatchStrategyFormation(data, signals=[...])     # dynamic_weights=True

# Method 3: DataUncertaintyAnalysis (default True, can change)
dua = DataUncertaintyAnalysis(data, signals=[...])      # dynamic_weights=True
```

**To ensure consistency, explicitly set `dynamic_weights`:**

```python
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig

config = StrategyFormationConfig(
    data=DataConfig(),
    formation=FormationConfig(
        dynamic_weights=True  # Explicit setting
    )
)

sf = StrategyFormation(data, strategy, config=config)
```

---

### When to Use Each Setting

**Use `dynamic_weights=True` when:**
- Simulating realistic portfolio behavior (VW adjusts as bonds grow/shrink)
- Comparing with BatchStrategyFormation or DataUncertaintyAnalysis results
- Replicating baseline test results
- Portfolio weights should reflect current market values

**Use `dynamic_weights=False` when:**
- Analyzing pure factor exposure (weights fixed at formation)
- Isolating signal effect from VW changes
- Comparing with older PyBondLab implementations
- Simpler interpretation (VW source is always formation)

---

### Summary Table (dynamic_weights)

| Aspect | `dynamic_weights=True` | `dynamic_weights=False` |
|--------|------------------------|-------------------------|
| **VW Source** | return_date - 1 (d-1) | formation_date (t) |
| **VW Consistency** | Same VW date for all cohorts at return date | Different VW date per cohort |
| **ID Intersection** | 3-way (formation ∩ return ∩ VW_date) | 2-way (formation ∩ return) |
| **Bond Filtering** | More restrictive (must exist at d-1) | Less restrictive |
| **HP=1 Effect** | **None** (both identical) | **None** (both identical) |
| **HP>1 Effect** | Weights vary by return date | Weights fixed at formation |
| **Interpretation** | "Dynamic" - tracks VW changes | "Static" - captures original VW |
| **Default in FormationConfig** | No | **Yes (default=False)** |
| **BatchStrategyFormation** | **Yes (hardcoded)** | No |
| **DataUncertaintyAnalysis** | **Yes (default)** | Optional |
| **pbl_test baseline** | **Yes (hardcoded)** | No |

**Key Takeaways:**

1. **For HP=1:** `dynamic_weights` has **NO effect** - both settings are mathematically identical
2. **For HP>1:** `dynamic_weights` has **SIGNIFICANT effect** - different VW sources and bond sets
3. **Watch for inconsistency:** Different classes default to different settings
4. **Explicitly set for reproducibility:** Always specify `dynamic_weights` when comparing across methods
