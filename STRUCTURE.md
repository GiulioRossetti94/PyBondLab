# PyBondLab Package Structure

This document describes the PyBondLab package architecture, module organization, and data flow.

## Table of Contents

1. [Package Overview](#package-overview)
2. [Directory Structure](#directory-structure)
3. [Core Modules](#core-modules)
4. [Strategy Classes](#strategy-classes)
5. [Configuration System](#configuration-system)
6. [Data Flow](#data-flow)
7. [Subpackages](#subpackages)
8. [Dependencies](#dependencies)

---

## Package Overview

PyBondLab is organized into three main layers:

1. **Core Layer**: Portfolio formation engine (`PyBondLab.py`, `StrategyClass.py`)
2. **Analysis Layer**: Results, turnover tracking, anomaly analysis
3. **Utilities Layer**: Data loading, I/O, visualization, filtering

The package follows a modular design where strategy definitions are decoupled from the portfolio formation engine, allowing for flexible extension.

---

## Directory Structure

```
PyBondLab_v2/
├── PyBondLab/                      # Main package directory
│   ├── __init__.py                 # Package exports and version
│   │
│   ├── PyBondLab.py               # Core: StrategyFormation class
│   ├── StrategyClass.py           # Core: Strategy definitions
│   ├── FilterClass.py             # Core: Data filtering
│   ├── config.py                  # Configuration dataclasses
│   ├── constants.py               # Package constants and enums
│   │
│   ├── results.py                 # Results container and methods
│   ├── turnover_tracking.py       # Turnover diagnostic system
│   ├── AnomalyAssayer.py          # Anomaly correlation analysis
│   ├── anomaly_correlation.py     # Correlation computations
│   ├── rolling_beta.py            # Rolling beta estimation (NEW)
│   │
│   ├── precompute.py              # Precomputation utilities
│   ├── utils.py                   # General utilities + panel validation
│   ├── utils_all.py               # Utility functions
│   ├── utils_optimized.py         # Performance-optimized functions (Numba)
│   ├── utils_portfolio.py         # Portfolio computation utilities
│   ├── utils_turnover.py          # Turnover calculation functions
│   ├── utils_within_firm.py       # Within-firm sorting utilities
│   │
│   ├── data/                       # Data loading subpackage
│   │   ├── __init__.py
│   │   ├── data_loading.py        # Data loading utilities
│   │   └── WRDS/
│   │       ├── __init__.py
│   │       └── breakpoints_wrds.csv  # Reference breakpoints
│   │
│   ├── describe/                   # Pre-analysis statistics subpackage (NEW)
│   │   ├── __init__.py
│   │   ├── base.py                # Base classes for describe modules
│   │   ├── pre_analysis.py        # PreAnalysisStats class
│   │   ├── results.py             # PreAnalysisResult class
│   │   ├── correlations.py        # Correlation computations
│   │   └── utils.py               # Statistical utility functions
│   │
│   ├── iotools/                    # I/O utilities subpackage
│   │   ├── __init__.py
│   │   ├── PyBondLabResults.py    # Results I/O handler
│   │   └── table.py               # LaTeX table generation
│   │
│   ├── visualization/              # Visualization subpackage
│   │   ├── __init__.py
│   │   ├── plotting.py            # Plotting functions
│   │   └── _latex.py              # LaTeX formatting helpers
│   │
│   └── optm/                       # Optimization subpackage (reserved)
│       └── __init__.py
│
├── examples/                       # Example scripts
│   ├── strategies/                # Basic strategy examples
│   ├── rebalancing/               # Rebalancing frequency examples
│   ├── formation_options/         # Formation options examples
│   ├── turnover/                  # Turnover tracking examples
│   └── real_data_examples/        # Real data examples
│
├── tests/                          # Test suite
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
├── README.md                       # User documentation
├── STRUCTURE.md                    # This file
├── PACKAGE_SUMMARY.md             # Internal reference
└── LICENSE                         # MIT License
```

---

## Core Modules

### `PyBondLab.py` - Portfolio Formation Engine

**Primary Class**: `StrategyFormation`

**Responsibilities**:
- Execute portfolio sorting based on strategy definitions
- Manage staggered and non-staggered rebalancing
- Compute equal-weighted and value-weighted returns
- Calculate turnover and portfolio characteristics
- Coordinate data flow through filtering and precomputation

**Key Methods**:
- `fit()`: Execute portfolio formation
- `_handle_double_sorting()`: Double sort implementation
- `_handle_single_sorting()`: Single sort implementation
- `_compute_returns()`: Return aggregation
- `_compute_turnover()`: Turnover calculation

**Related Functions**:
- `load_breakpoints_WRDS()`: Load reference breakpoints from WRDS data

**Inputs**:
- Panel data (date, ID, ret, sorting variables)
- Strategy object (defines sorting logic)
- Configuration (filtering, characteristics, rebalancing)

**Outputs**:
- Results object containing portfolio returns, turnover, characteristics

---

### `StrategyClass.py` - Strategy Definitions

**Abstract Base Class**: `Strategy`

**Concrete Strategy Classes**:

1. **`SingleSort`**: Univariate portfolio sorting
   - Parameters: holding_period, sort_var, num_portfolios, breakpoints
   - Optional: lookback_period, skip, rebalance_frequency
   - Methods: `compute_signal()`, `get_sort_var()`

2. **`DoubleSort`**: Bivariate portfolio sorting
   - Types: 'unconditional' (independent sorts), 'conditional' (sequential sorts)
   - Parameters: Two sets of sorting variables and portfolio counts
   - Methods: `get_sort_var()`, `get_sort_var2()`

3. **`Momentum`**: Past return-based strategies
   - Parameters: holding_period, lookback_period, skip
   - Signal: Cumulative return over formation period
   - Method: `compute_signal()` computes rolling returns

4. **`LTreversal`**: Long-term reversal strategies
   - Parameters: holding_period, lookback_period, skip
   - Signal: Long-term returns minus recent returns
   - Method: `compute_signal()` computes contrarian signal

5. **`WithinFirmSort`**: Within-firm high-low sorting
   - Groups bonds by firm, rating, and date
   - Creates high-low portfolios within each firm
   - Aggregates across firms within rating groups

**Common Properties**:
- `holding_period`: Portfolio holding duration
- `num_portfolios`: Number of portfolios to form
- `rebalance_frequency`: 'monthly', 'quarterly', 'semi-annual', 'annual', or int
- `rebalance_month`: Month(s) for rebalancing

**Inheritance Structure**:
```
Strategy (ABC)
├── SingleSort
├── DoubleSort
├── Momentum
├── LTreversal
└── WithinFirmSort
```

---

### `FilterClass.py` - Data Filtering

**Primary Class**: `Filter`

**Adjustment Methods**:

1. **Trimming** (`adj='trim'`):
   - Remove observations beyond threshold
   - Parameters: `w` (threshold or [lower, upper])
   - Effect: Sets filtered returns to NaN

2. **Winsorizing** (`adj='wins'`):
   - Cap extreme values at percentiles
   - Parameters: `w` (percentile), `loc` ('both', 'left', 'right')
   - Modes: Ex-ante (using past data) or ex-post
   - Method: `_winsorizing()` computes time-varying thresholds

3. **Price Filtering** (`adj='price'`):
   - Filter based on bond price levels
   - Parameters: `w` (price threshold), `price_threshold` (reference)
   - Requires: `PRICE` column in data

4. **Bounce Filtering** (`adj='bounce'`):
   - Remove price reversal patterns
   - Signal: Product of consecutive returns
   - Identifies: Large swings followed by reversals

**Method**: `apply_filters()` - Apply selected filter to data

**Output**: Filtered data with new column `ret_{adj}` containing adjusted returns

---

### `config.py` - Configuration System

**Configuration Classes** (dataclasses with validation):

1. **`DataConfig`**:
   - `rating`: Filter by rating ('IG', 'NIG', or tuple)
   - `subset_filter`: Characteristic-based filters {column: (min, max)}
   - `chars`: Characteristics to compute

2. **`StrategyFormationConfig`**:
   - Aggregates all configuration options
   - Data configuration
   - Filtering specifications
   - Output preferences

**Validation**:
- Parameter type checking
- Value range validation
- Consistency checks between related parameters

---

### `constants.py` - Package Constants

**Key Constants**:

- **`Defaults`**: Default parameter values
  - Portfolio counts, rebalancing frequencies
  - Filtering thresholds
  - Column name defaults

- **`RatingBounds`**: Rating category definitions
  - Investment grade (IG): 1-10
  - Non-investment grade (NIG): 11-22
  - Rating terciles for within-firm sorting

- **`ColumnNames`**: Standard column name mappings
  - Required: date, ID, ret
  - Optional: VW, PRICE, PERMNO, RATING_NUM

- **`FilterType`**: Enumeration of filter types
  - TRIM, WINS, PRICE, BOUNCE

- **`ValidationMessages`**: Error message templates

---

### `results.py` - Results Container

**Primary Class**: `Results`

**Attributes**:
- `ret_ew`: Equal-weighted portfolio returns (DataFrame)
- `ret_vw`: Value-weighted portfolio returns (DataFrame)
- `hml_ew`: High-minus-low spread (equal-weighted)
- `hml_vw`: High-minus-low spread (value-weighted)
- `turnover`: Portfolio turnover statistics
- `chars`: Portfolio characteristics over time
- `n_bonds`: Bond counts per portfolio

**Methods**:
- `summary()`: Print results summary
- `save()`: Save results to file
- `plot_returns()`: Visualize portfolio returns
- `to_dataframe()`: Export as DataFrame

---

### `turnover_tracking.py` - Turnover Diagnostics

**Primary Classes**:

1. **`TurnoverState`**: Container for turnover state
   - Tracks portfolio composition over time
   - Stores bond IDs, weights, and trading activity

2. **`TurnoverManager`**: Coordinates turnover logging
   - Classifies turnover events (13 case types)
   - Generates diagnostic reports
   - Exports detailed logs

**Key Functions**:
- `accumulate_turnover()`: Update turnover state
- `compute_nonstaggered_turnover()`: Non-staggered calculation
- `finalize_turnover()`: Generate final turnover metrics

**Turnover Cases**:
- `ZERO_NO_TRADING`: No portfolio changes (rare)
- `NORMAL_REBALANCING`: Standard rebalancing
- `HEAVY_REBALANCING`: Turnover > 100%
- `FULL_LIQUIDATION`: Portfolio sold
- `NEW_PORTFOLIO`: Portfolio created
- `BOTH_EMPTY`: Empty portfolio (should exclude)
- Additional cases for partial states

---

### `AnomalyAssayer.py` - Anomaly Analysis

**Primary Class**: `AssayAnomalyRunner`

**Functionality**:
- Correlation analysis across anomalies
- Time-series regression for multiple factors
- Heatmap visualization of correlations
- Summary statistics for factor construction

**Methods**:
- `run()`: Execute full anomaly analysis
- `compute_correlations()`: Pairwise correlations
- `plot_heatmap()`: Visualization
- `save_results()`: Export analysis

**Use Case**: Comparing multiple sorting variables or strategies to assess redundancy and novel information content.

---

### `rolling_beta.py` - Rolling Beta Estimation (NEW)

**Primary Class**: `RollingBeta`

**Purpose**: Compute rolling-window factor betas on bond panels as a precomputation step before portfolio sorting.

**Key Features**:
- Two computation engines: 'numba' (fast) and 'numpy' (feature-rich)
- Efficient O(1) window extraction via cumulative sums
- Proper NaN handling via pre-filtering valid observations
- Support for multiple return columns in one pass (numpy engine)
- Control variables support (numpy engine)

**Parameters**:
- `factors`: DataFrame with factor returns (must include 'date' column)
- `window`: Rolling window size (default: 60 months)
- `min_obs`: Minimum observations required (default: 36)
- `ret_col`: Return column name (default: 'ret')
- `date_col`: Date column name (default: 'date')
- `id_col`: Entity identifier column (default: 'ID')
- `engine`: Computation engine ('numba' or 'numpy')
- `controls`: Control variables (numpy engine only)

**Output Columns**:
- `beta_{factor}`: Factor beta estimates
- `alpha`: Regression intercept
- `idio_vol`: Idiosyncratic volatility
- `total_vol`: Total volatility
- `adj_r2`: Adjusted R-squared

**Usage**:
```python
from PyBondLab import RollingBeta

# Single-factor model
rb = RollingBeta(factors=market_factor, window=60, min_obs=36)
data_with_betas = rb.fit_transform(data)

# Multi-factor model (numpy engine)
rb = RollingBeta(factors=ff_factors, window=60, engine='numpy')
data_with_betas = rb.fit_transform(data)

# Then use with SingleSort
strategy = SingleSort(holding_period=6, sort_var='beta_MKT', num_portfolios=10)
```

---

### `utils.py` - Panel Validation Utilities

**Key Functions**:

1. **`validate_panel()`**: Validate panel data for duplicate ID-date pairs
   - Detects duplicates that cause errors in portfolio formation
   - Options: 'error' (raise), 'warn' (print warning), 'drop' (remove duplicates)
   - Supports aliases (IDvar, DATEvar) for consistency with StrategyFormation

2. **`check_duplicates()`**: Quick check for duplicate ID-date pairs
   - Returns boolean indicating presence of duplicates
   - Faster than validate_panel for simple checks

**Usage**:
```python
from PyBondLab import validate_panel, check_duplicates

# Check and warn about duplicates
df = validate_panel(bond_data, id_col='cusip', date_col='date')

# Check and automatically drop duplicates
df_clean = validate_panel(bond_data, handle_duplicates='drop')

# Quick check
has_dupes = check_duplicates(bond_data, id_col='cusip', date_col='date')
```

---

## Strategy Classes

### Inheritance Hierarchy

```
Strategy (Abstract Base Class)
│
│   Properties: holding_period, num_portfolios, lookback_period, skip
│   Methods: compute_signal(), get_sort_var()
│
├── SingleSort
│   │   Additional: sort_var, breakpoints, breakpoint_universe_func
│   │   Returns: Data unchanged
│   │
├── DoubleSort
│   │   Additional: sort_var2, num_portfolios2, how ('unconditional'/'conditional')
│   │   Returns: Data unchanged
│   │
├── Momentum
│   │   Signal: Cumulative past returns
│   │   Returns: Data with 'signal' column
│   │
├── LTreversal
│   │   Signal: Long-term returns - recent returns
│   │   Returns: Data with 'signal' column
│   │
└── WithinFirmSort
    │   Additional: firm_id_col, min_bonds_per_firm, rating_bins
    │   Returns: Data unchanged (sorting logic in formation)
```

### Strategy Selection Logic

When `StrategyFormation.fit()` is called:

1. Check if strategy requires signal computation (Momentum, LTreversal)
   - If yes: Call `strategy.compute_signal(data)`
   - Result: Data with computed signal column

2. Determine sorting type:
   - Single sort: `DoubleSort` attribute missing or 0
   - Double sort: `DoubleSort` attribute present and > 0

3. Apply filtering (if specified):
   - Create `Filter` object
   - Apply filter method
   - Use `ret_{adj}` column for returns

4. Execute sorting:
   - Compute breakpoints (quantiles or custom)
   - Assign bonds to portfolios
   - Aggregate returns within portfolios

5. Handle rebalancing:
   - Staggered: Monthly overlapping cohorts
   - Non-staggered: Discrete rebalancing dates
   - Custom frequency: User-defined schedule

---

## Configuration System

### Configuration Flow

```
User Inputs
    │
    ├─→ Strategy Object (e.g., SingleSort)
    │       └─→ holding_period, sort_var, num_portfolios
    │
    ├─→ Data Configuration (optional)
    │       └─→ rating filter, subset_filter, chars
    │
    ├─→ Filter Specification (optional)
    │       └─→ adj, w, loc
    │
    └─→ StrategyFormation
            └─→ Coordinates all components
```

### Configuration Validation

All configuration classes use dataclass `__post_init__` for validation:

1. **Type checking**: Ensure correct parameter types
2. **Value validation**: Check ranges and allowed values
3. **Consistency checks**: Verify related parameters are compatible
4. **Default assignment**: Fill missing optional parameters

Example validation chain:
```
SingleSort(holding_period=6, sort_var='RATING_NUM', num_portfolios=5)
    │
    ├─→ Validate holding_period > 0
    ├─→ Validate sort_var is string
    ├─→ Validate num_portfolios > 0
    ├─→ Validate rebalance_frequency in allowed values
    └─→ Build strategy name string
```

---

## Data Flow

### Primary Data Flow

```
Input Data (Panel)
    │
    ├─→ Filter (optional)
    │       └─→ Trimming, winsorizing, price filter, bounce filter
    │
    ├─→ Strategy Signal Computation (if needed)
    │       └─→ Momentum: Cumulative returns
    │       └─→ LTreversal: Long-term - recent returns
    │
    ├─→ StrategyFormation.fit()
    │       │
    │       ├─→ Rating/Subset Filtering
    │       ├─→ Sort Data by ID and Date
    │       ├─→ Compute Breakpoints
    │       │       └─→ Quantiles or custom breakpoints
    │       │       └─→ Optional: Universe filtering (e.g., NYSE only)
    │       │
    │       ├─→ Assign to Portfolios
    │       │       └─→ Single or double sorting
    │       │       └─→ Staggered or non-staggered
    │       │
    │       ├─→ Compute Returns
    │       │       └─→ Equal-weighted and value-weighted
    │       │       └─→ High-minus-low spreads
    │       │
    │       ├─→ Compute Turnover (if requested)
    │       │       └─→ Track bond composition
    │       │       └─→ Calculate trading activity
    │       │       └─→ Diagnostic logging
    │       │
    │       └─→ Compute Characteristics (if requested)
    │               └─→ Average characteristics per portfolio
    │
    └─→ Results Object
            ├─→ ret_ew, ret_vw (portfolio returns)
            ├─→ hml_ew, hml_vw (spreads)
            ├─→ turnover (statistics)
            ├─→ chars (characteristics)
            └─→ n_bonds (counts)
```

### Turnover Tracking Flow

```
Portfolio Formation
    │
    ├─→ For each rebalancing date:
    │       │
    │       ├─→ Get portfolio composition at t-1
    │       ├─→ Get portfolio composition at t
    │       │
    │       ├─→ Identify bonds:
    │       │       ├─→ Added (in t, not in t-1)
    │       │       ├─→ Removed (in t-1, not in t)
    │       │       └─→ Held (in both)
    │       │
    │       ├─→ Compute weights:
    │       │       ├─→ Weight changes for held bonds
    │       │       ├─→ Sum of added weights
    │       │       └─→ Sum of removed weights
    │       │
    │       ├─→ Calculate turnover:
    │       │       └─→ 0.5 * (|added| + |removed|)
    │       │
    │       └─→ Classify case:
    │               └─→ ZERO_NO_TRADING, NORMAL_REBALANCING, etc.
    │
    └─→ TurnoverManager
            ├─→ Detailed log (all calculations)
            ├─→ Alerts log (suspicious cases)
            └─→ Summary log (aggregated statistics)
```

---

## Subpackages

### `data/` - Data Loading

**Purpose**: Load and preprocess bond data

**Modules**:
- `data_loading.py`: Generic data loading utilities
- `WRDS/breakpoints_wrds.csv`: Reference breakpoints from WRDS dataset

**Key Function**: `load_breakpoints_WRDS()` in `PyBondLab.py`

**Usage**: Reference breakpoints for reproducibility across studies

---

### `describe/` - Pre-Analysis Statistics (NEW)

**Purpose**: Compute summary statistics for panel data before portfolio formation

**Modules**:
- `base.py`: Abstract base class for describe modules
- `pre_analysis.py`: PreAnalysisStats class for cross-sectional statistics
- `results.py`: PreAnalysisResult container class
- `correlations.py`: Pairwise correlation computations
- `utils.py`: Statistical utility functions (skewness, kurtosis, etc.)

**Primary Classes**:

1. **`PreAnalysisStats`**: Compute cross-sectional distribution statistics
   - For each time period, computes mean, std, skewness, kurtosis, percentiles
   - Aggregates cross-sectional statistics over time
   - Supports optional filtering (trim, wins, price, bounce)
   - Supports rating and subset filtering

2. **`PreAnalysisResult`**: Container for pre-analysis results
   - Cross-sectional statistics by variable
   - Time-series of statistics
   - Summary tables and plots

**Usage**:
```python
from PyBondLab import PreAnalysisStats

# Compute summary statistics
stats = PreAnalysisStats(
    data=bond_data,
    variables=['duration', 'maturity', 'ret'],
    date_col='date',
    id_col='cusip',
    issuer_col='PERMNO'
)
result = stats.compute()

# Get summary for a variable
print(result.summary('duration'))

# Get cross-sectional stats over time
cs_stats = result.get_cs_stats('ret')
```

**Key Features**:
- Cross-sectional percentiles (5th, 25th, 50th, 75th, 95th)
- Higher moments (skewness, kurtosis)
- Entity and issuer counts
- Optional return filtering effects

---

### `iotools/` - Input/Output

**Purpose**: Save and load results, generate tables

**Modules**:
- `PyBondLabResults.py`: Results serialization
- `table.py`: LaTeX table generation

**Functionality**:
- Save Results objects to disk
- Load Results for later analysis
- Generate publication-ready LaTeX tables
- Format output for different use cases

---

### `visualization/` - Plotting and Visualization

**Purpose**: Create plots and visualizations

**Modules**:
- `plotting.py`: Portfolio return plots, turnover visualizations
- `_latex.py`: LaTeX formatting for tables and figures

**Key Functions**:
- `plot_cumulative_returns()`: Cumulative return plots
- `plot_turnover_distribution()`: Turnover histograms
- `plot_correlation_heatmap()`: Anomaly correlation matrices

---

### `optm/` - Optimization (Reserved)

**Purpose**: Reserved for future optimization features

**Status**: Placeholder for portfolio optimization functionality

**Potential Features**:
- Mean-variance optimization
- Factor timing strategies
- Portfolio weighting schemes

---

## Dependencies

### External Dependencies

**Core Scientific Computing**:
- `numpy`: Array operations, numerical computations
- `pandas`: Data manipulation, panel data structures
- `statsmodels`: Time-series regression, Newey-West standard errors
- `scipy`: Statistical functions (skewness, kurtosis, kernel density)

**Visualization**:
- `matplotlib`: Plotting and visualization

**Data Access**:
- `wrds`: WRDS database connection (optional, for data download)
- `pyarrow`: Efficient data serialization (parquet support)

**Performance Optimization** (optional but recommended):
- `numba`: JIT compilation for performance-critical functions
  - Used in: `rolling_beta.py`, `utils_optimized.py`, `utils_within_firm.py`
  - Falls back gracefully if not installed

**Requirements**:
- Python >= 3.11 (for improved type hints and performance)
- numpy < 2 (compatibility with current codebase)

### Internal Dependencies

**Module Dependency Graph**:

```
PyBondLab.py (Core)
    ├─→ StrategyClass.py (strategy definitions)
    ├─→ FilterClass.py (data filtering)
    ├─→ config.py (configuration)
    ├─→ constants.py (constants)
    ├─→ results.py (results container)
    ├─→ turnover_tracking.py (turnover)
    ├─→ utils_portfolio.py (portfolio utilities)
    ├─→ utils_turnover.py (turnover utilities)
    └─→ precompute.py (precomputation)

StrategyClass.py
    └─→ constants.py

FilterClass.py
    └─→ No internal dependencies

AnomalyAssayer.py
    ├─→ anomaly_correlation.py
    └─→ visualization/plotting.py
```

**Circular Dependency Prevention**:
- Core modules (`PyBondLab.py`, `StrategyClass.py`) do not import from analysis modules
- Analysis modules (`results.py`, `AnomalyAssayer.py`) can import from core
- Utilities are self-contained or import only from core

---

## API Summary

### Primary User Interface

**Main Entry Point**:
```python
from PyBondLab import StrategyFormation, SingleSort

strategy = SingleSort(holding_period=6, sort_var='RATING_NUM', num_portfolios=5)
results = StrategyFormation(data, strategy=strategy).fit()
```

**Exported Classes** (via `__init__.py`):
- `StrategyFormation`: Portfolio formation engine
- `SingleSort`, `DoubleSort`, `Momentum`, `LTreversal`, `WithinFirmSort`: Strategies
- `AssayAnomaly`: Anomaly analysis
- `RollingBeta`: Rolling beta estimation (NEW)
- `PreAnalysisStats`, `PreAnalysisResult`: Pre-analysis statistics (NEW)

**Exported Functions**:
- `load_breakpoints_WRDS()`: Load reference breakpoints
- `validate_panel()`: Validate panel data for duplicates (NEW)
- `check_duplicates()`: Quick duplicate check (NEW)
- `build_precomputed_data()`: Build precomputed data for multiple strategies

### Advanced Usage

**Precomputation** (for repeated analyses):
```python
from PyBondLab import build_precomputed_data

precomp = build_precomputed_data(data, strategies=[strategy1, strategy2])
results1 = StrategyFormation(data, strategy=strategy1, precomputed=precomp).fit()
results2 = StrategyFormation(data, strategy=strategy2, precomputed=precomp).fit()
```

**Turnover Tracking**:
```python
from PyBondLab.turnover_tracking import TurnoverLogger

logger = TurnoverLogger(output_dir="logs")
results = StrategyFormation(data, strategy=strategy, turnover=True,
                           turnover_logger=logger).fit()
logger.save("strategy_name")
```

---

## Extension Points

### Adding New Strategies

To add a custom strategy:

1. Subclass `Strategy` from `StrategyClass.py`
2. Implement required methods:
   - `compute_signal(data)`: Return data with signal column (or data unchanged)
   - `get_sort_var(adj=None)`: Return sorting variable name
3. Set required attributes in `__init__`
4. Register in `__init__.py` if part of package

Example:
```python
from PyBondLab.StrategyClass import Strategy

class CustomStrategy(Strategy):
    def __init__(self, holding_period, num_portfolios, custom_param):
        super().__init__(holding_period, num_portfolios)
        self.custom_param = custom_param
        self.__strategy_name__ = "Custom Strategy"

    def compute_signal(self, data):
        # Implement custom signal logic
        data['signal'] = ...  # Your signal computation
        return data

    def get_sort_var(self, adj=None):
        return 'signal' if not adj else f'signal_{adj}'
```

### Adding New Filters

To add a custom filter:

1. Add method to `Filter` class in `FilterClass.py`
2. Update `apply_filters()` to call new method
3. Add adjustment type to constants if needed

---

## Performance Considerations

**Optimized Modules**:
- `utils_optimized.py`: Numba-accelerated functions for bottleneck operations
- `utils_portfolio.py`: Vectorized portfolio computations
- Precomputation system: Cache repeated calculations

**Memory Management**:
- Chunked processing for large datasets
- In-place operations where possible
- Efficient data structures (numpy arrays, pandas DataFrames)

**Computational Bottlenecks**:
- Portfolio assignment (O(N × T) where N = bonds, T = periods)
- Turnover tracking (O(N × T) with bond-level tracking)
- Within-firm sorting (O(N × T × F) where F = firms)

**Optimization Strategies**:
- Use precomputed data for multiple strategies
- Disable turnover tracking if not needed
- Filter data before portfolio formation
- Use value-weighted returns when appropriate (faster than equal-weighted)

---

## Version History

**v0.2.0** (Current):
- Rolling beta estimation module (`RollingBeta`)
- Pre-analysis statistics module (`PreAnalysisStats`, `PreAnalysisResult`)
- Panel validation utilities (`validate_panel`, `check_duplicates`)
- Enhanced turnover tracking with diagnostic logging
- Non-staggered rebalancing frequencies
- Within-firm sorting functionality
- Configuration system with validation
- Performance optimizations with Numba
- Scipy integration for statistical functions

**v0.1.0**:
- Initial release
- Basic single and double sorting
- Momentum and long-term reversal strategies
- Data filtering utilities

---

## Future Development

**Planned Features**:
- Portfolio optimization (optm/ subpackage)
- Additional rebalancing methods (calendar-based, signal-triggered)
- Extended characteristic computation
- Performance attribution analysis
- Risk model integration

**Community Contributions**:
See README.md for contribution guidelines and GitHub issue tracker.

---

**Document Version**: 1.1
**Last Updated**: 2025-12-15
**Maintainers**: Giulio Rossetti, Alex Dickerson
