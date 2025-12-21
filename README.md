# PyBondLab

PyBondLab is a Python package for portfolio sorting and empirical asset pricing research, with a focus on corporate bonds. It is part of the [Open Source Bond Asset Pricing project](https://openbondassetpricing.com/).

## Overview

PyBondLab provides tools for computing and evaluating investment strategies. It features look-ahead bias free data cleaning procedures to ensure the integrity and reliability of empirical results.

**Key Features:**
- Single and double portfolio sorting
- Momentum and long-term reversal strategies
- Within-firm sorting
- Flexible rebalancing frequencies (staggered and non-staggered)
- Custom breakpoint universes (e.g., NYSE breakpoints for Fama-French factors)
- Portfolio turnover computation and tracking
- Rolling beta estimation
- Pre-analysis summary statistics
- Look-ahead bias free filtering procedures
- Configurable factor naming with sign correction

## Installation

Install the latest release using pip:
```bash
pip install PyBondLab
```

Or install from source:
```bash
git clone https://github.com/GiulioRossetti94/PyBondLab.git
cd PyBondLab
pip install -e .
```

For optional performance optimizations (recommended):
```bash
pip install PyBondLab[performance]  # Includes numba for faster computation
```

## Quick Start

```python
import PyBondLab as pbl
import pandas as pd

# Load your bond data (must have: date, bond ID, returns, and sorting variable)
data = pd.read_csv("your_bond_data.csv")

# Define a single-sort strategy
strategy = pbl.SingleSort(
    holding_period=1,       # 1-month holding period
    sort_var='RATING_NUM',  # Sort by credit rating
    num_portfolios=5        # Quintile portfolios
)

# Run the strategy
sf = pbl.StrategyFormation(data, strategy=strategy)
results = sf.fit(IDvar='cusip', RETvar='ret')

# Get long-short portfolio returns
ew, vw = results.get_long_short()
```

## Usage Examples

For complete examples, see the [examples](examples/) folder.

### Portfolio Sorting

Sort corporate bonds into quintile portfolios based on credit ratings:

```python
import PyBondLab as pbl
import pandas as pd

# Load bond data with columns: date, ISSUE_ID, RATING_NUM, ret, VW
data = pd.read_csv("bond_data.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['ISSUE_ID', 'date'])

# Define strategy
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='RATING_NUM',
    num_portfolios=5
)

# Fit the strategy
results = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='ISSUE_ID',
    RETvar='ret'
)

# Get equal-weighted and value-weighted long-short returns
ew, vw = results.get_long_short()
```

### Double Sorting

Sort bonds first by rating, then by maturity within each rating group:

```python
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='RATING_NUM',
    num_portfolios=3,
    sort_var2='maturity',
    num_portfolios2=3,
    how='conditional'  # or 'unconditional' for independent sorts
)

results = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='cusip',
    RETvar='ret'
)
```

### Momentum Strategy

```python
strategy = pbl.Momentum(
    holding_period=6,      # 6-month holding period
    lookback_period=6,     # Past 6-month returns
    skip=1,                # Skip most recent month
    num_portfolios=5
)

results = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='cusip',
    RETvar='ret'
)
```

### Portfolio Turnover

```python
# Enable turnover computation
results = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='cusip',
    RETvar='ret',
    turnover=True
)

# Get turnover statistics
ew_turnover, vw_turnover = results.get_ptf_turnover()
```

### Factor Naming (NamingConfig)

Use `NamingConfig` for consistent, readable factor names:

```python
from PyBondLab import NamingConfig

# Default naming: lowercase signal names
cfg = NamingConfig()
ew, vw = results.get_long_short(naming=cfg)
print(ew.name)  # Output: rating_num (instead of EWEA_ALL_1)

# With sign correction: flip negative factors, add '*' suffix
cfg = NamingConfig(sign_correct=True)
ew, vw = results.get_long_short(naming=cfg)
print(ew.name)  # Output: rating_num* (if factor was flipped)

# With weighting prefix
cfg = NamingConfig(weighting_prefix=True)
ew, vw = results.get_long_short(naming=cfg)
print(ew.name, vw.name)  # Output: ew_rating_num, vw_rating_num

# Factor-level turnover (average of long and short legs)
ew_turn, vw_turn = results.get_turnover(level='factor', naming=cfg)
print(ew_turn.name)  # Output: ew_rating_num_turnover
```

For complete documentation, see [docs/NamingConfig_README.md](docs/NamingConfig_README.md).

### Rolling Beta Estimation

Compute rolling betas for use in portfolio sorting:

```python
# Load factor data
factors = pd.read_csv("factors.csv")  # Must have 'date' column

# Compute rolling betas
rb = pbl.RollingBeta(factors=factors, window=60, min_obs=36)
data_with_betas = rb.fit_transform(data)

# Sort on market beta
strategy = pbl.SingleSort(
    holding_period=1,
    sort_var='beta_MKT',
    num_portfolios=5
)
```

### Pre-Analysis Statistics

Compute summary statistics before portfolio formation:

```python
stats = pbl.PreAnalysisStats(
    data=data,
    variables=['ret', 'duration', 'maturity'],
    date_col='date',
    id_col='cusip'
)
result = stats.compute()

# Get summary for a variable
print(result.summary('ret'))
```

## Data Cleaning / Filtering

PyBondLab provides four data cleaning procedures commonly used in corporate bond research. Filters are passed as a dictionary:

```python
filters = {'adj': 'trim', 'w': 0.2}
results = pbl.StrategyFormation(data, strategy=strategy, filters=filters).fit(...)
```

### Return Exclusion (Trimming)

Filter out bonds with returns above/below a threshold:

- `{'adj': 'trim', 'w': 0.2}` excludes returns > 20%
- `{'adj': 'trim', 'w': -0.2}` excludes returns < -20%
- `{'adj': 'trim', 'w': [-0.2, 0.2]}` excludes returns outside [-20%, 20%]

### Price Exclusion

Filter out bonds with prices above/below a threshold. Requires a `PRICE` column or specify `PRICEvar` in `.fit()`:

- `{'adj': 'price', 'w': 150}` excludes prices > 150
- `{'adj': 'price', 'w': 20}` excludes prices < 20
- `{'adj': 'price', 'w': [20, 150]}` excludes prices outside [20, 150]

The threshold for below/above exclusion defaults to 25. Use `'price_threshold'` to change:
- `{'adj': 'price', 'w': 26, 'price_threshold': 30}` excludes prices < 26

### Return Bounce-Back Exclusion

Filter out bonds where the product of consecutive returns meets a threshold:

- `{'adj': 'bounce', 'w': 0.01}` excludes if $R_t \times R_{t-1} > 0.01$
- `{'adj': 'bounce', 'w': -0.01}` excludes if $R_t \times R_{t-1} < -0.01$
- `{'adj': 'bounce', 'w': [-0.01, 0.01]}` excludes if $|R_t \times R_{t-1}| > 0.01$

### Return Winsorization

**Ex Ante Winsorization**: Affects portfolio formation for strategies using signals derived from past returns (momentum, reversal). Returns are winsorized using the distribution up until the portfolio formation month $t$, then the signal is computed. This is look-ahead bias free.

**Ex Post Winsorization**: Affects portfolio performance computation by modifying returns used to calculate portfolio returns. This introduces look-ahead bias and makes performance unattainable.

- `{'adj': 'wins', 'w': 98, 'loc': 'right'}` winsorize right tail at 98th percentile
- `{'adj': 'wins', 'w': 98, 'loc': 'left'}` winsorize left tail at 2nd percentile
- `{'adj': 'wins', 'w': 98, 'loc': 'both'}` winsorize both tails

For ex ante winsorization, returns at time $t$ are winsorized based on the pooled distribution from $t_0$ to $t$. To avoid recomputing percentiles at each iteration, pass precomputed breakpoints:

```python
BREAKPOINTS = pbl.load_breakpoints_WRDS()
BREAKPOINTS.index = pd.to_datetime(BREAKPOINTS.index)

filters = {
    'adj': 'wins',
    'w': 98,
    'loc': 'both',
    'df_breakpoints': BREAKPOINTS
}
```

## Rebalancing Frequencies

Control when portfolios are rebalanced:

```python
# Monthly rebalancing (default)
strategy = pbl.SingleSort(holding_period=1, sort_var='var', num_portfolios=5)

# Quarterly rebalancing
strategy = pbl.SingleSort(
    holding_period=3,
    sort_var='var',
    num_portfolios=5,
    rebalance_frequency='quarterly'
)

# Annual rebalancing in June (Fama-French style)
strategy = pbl.SingleSort(
    holding_period=12,
    sort_var='var',
    num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=7  # See note below about timing
)
```

### Important: Understanding `rebalance_month` Timing

The `rebalance_month` parameter specifies **the calendar month when portfolio formation occurs**. Returns are then measured **starting the following month**.

For Fama-French style "June rebalancing", use `rebalance_month=7`:

| `rebalance_month` | Formation Month | Returns Start |
|-------------------|-----------------|---------------|
| `6` | June | July |
| `7` | July | August |


## Custom Breakpoint Universes

By default, breakpoints are computed using all observations in the dataset. You can specify a custom filter function to compute breakpoints on a subset of observations while applying them to the full dataset.

This is useful for replicating methodologies like Fama-French, which use NYSE stocks to compute breakpoints but apply them to all stocks (NYSE, AMEX, NASDAQ).

### Example: NYSE Breakpoints for Fama-French Factors

```python
def nyse_filter(df):
    """Filter to NYSE stocks for breakpoint computation."""
    return (
        (df['EXCHCD'] == 1) &              # NYSE only
        (df['BtM'] > 0) &                  # Positive book-to-market
        (df['ME'] > 0) &                   # Positive market equity
        (df['SHRCD'].isin([10, 11]))       # Ordinary common shares
    )

# 2x3 Fama-French style double sort
strategy = pbl.DoubleSort(
    holding_period=12,
    sort_var='ME',                          # Size
    sort_var2='BtM',                        # Book-to-Market
    num_portfolios=2,                       # 2 size groups
    num_portfolios2=3,                      # 3 BtM groups
    breakpoints=[50],                       # Median for size
    breakpoints2=[30, 70],                  # 30/70 percentiles for BtM
    how='unconditional',                    # Independent sorts
    rebalance_frequency='annual',
    rebalance_month=7,                      # June rebalancing
    breakpoint_universe_func=nyse_filter,   # NYSE breakpoints for size
    breakpoint_universe_func2=nyse_filter   # NYSE breakpoints for BtM
)

results = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='PERMNO',
    RETvar='ret'
)
```

The filter function receives a DataFrame and returns a boolean Series indicating which rows to include in breakpoint computation. Breakpoints are computed on the filtered subset, then applied to all observations when forming portfolios.

See [examples/real_data_examples/fama_french_factors/](examples/real_data_examples/fama_french_factors/) for a complete Fama-French factor replication.

## Data Uncertainty

The scripts in [examples/](examples/) provide replications of results from Dickerson, Robotti, and Rossetti ([2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)).

These scripts compare the effects of ex-ante and ex-post data cleaning procedures on long-short portfolio returns, highlighting the look-ahead bias introduced by ex-post cleaning.

## Requirements

- Python >= 3.11
- numpy < 2
- pandas >= 1.5
- statsmodels >= 0.14
- matplotlib >= 3.5
- scipy >= 1.10
- pyarrow

Optional:
- numba >= 0.57 (for performance optimization)
- wrds (for WRDS data access)

## References

Dickerson, Robotti, and Rossetti ([2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)). Data Uncertainty in Corporate Bond Markets.

Novy-Marx and Velikov ([2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4338007)). Assaying Anomalies.

[Open Source Bond Asset Pricing](https://openbondassetpricing.com/)

## Contact

- Giulio Rossetti - giulio.rossetti.1@wbs.ac.uk
- Alex Dickerson - alexander.dickerson1@unsw.edu.au

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
