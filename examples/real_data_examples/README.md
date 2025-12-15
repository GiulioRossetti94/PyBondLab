# Real Data Examples

Concise examples demonstrating PyBondLab strategies using real bond market data.

## Overview

These examples use actual corporate bond data to demonstrate portfolio formation strategies. Unlike the synthetic data examples, these show real-world performance characteristics and data considerations.

## Data Sources

**Primary**: OSBAP (Open Source Bond Asset Pricing)
- Website: https://openbondassetpricing.com/
- Free access to academic-quality bond data
- Covers major corporate bonds with validated returns

**Alternative**: WRDS Enhanced TRACE
- Requires institutional access through WRDS
- Comprehensive bond transaction data

## Setup

### 1. Data Path Configuration

Update the default path in `data_loader.py`:

```python
def load_bond_data(data_path=None, ...):
    if data_path is None:
        data_path = "/path/to/your/bond_data.csv"  # Update this
```

Or specify the path when running examples:

```python
from data_loader import load_bond_data
data = load_bond_data(data_path="/custom/path/to/data.csv")
```

### 2. Required Data Columns

Minimum required columns:
- `date`: Date in YYYY-MM-DD format
- `cusip` or `ID`: Bond identifier
- `ret` or `bond_ret`: Monthly returns (decimal format)
- `RATING_NUM` or `rating`: Numeric credit rating (1-22)
- `BOND_VALUE` or `VW`: Market value for value-weighting

Optional but recommended:
- `PRICE` or `BONDPRC`: Bond price
- `PERMNO`: Firm identifier (for within-firm sorts)
- `OAS`: Option-adjusted spread
- `DURATION`: Duration

### 3. Running Examples

```bash
cd examples/real_data_examples

# Basic strategies
python rating_sort.py
python momentum.py
python double_sort.py

# Filtering and subsets
python with_winsorization.py
python ig_vs_nig.py

# Rebalancing
python quarterly_rebalancing.py
```

## Examples

### Basic Strategies

**`rating_sort.py`** - Credit Rating Sort
- 5 portfolios sorted by credit rating
- 6-month holding period
- Shows VW and EW long-short returns
- Includes turnover statistics

**`momentum.py`** - Momentum Strategy
- 6-1-6 specification (6-month formation, 1-month skip, 6-month hold)
- 10 portfolios (deciles)
- Tests bond momentum effect

**`double_sort.py`** - Bivariate Sort
- 3×3 independent sort on rating and size
- Demonstrates unconditional double sorting
- Shows interaction effects

### Filtering Examples

**`with_winsorization.py`** - Return Winsorization
- Applies 99% winsorization (1st/99th percentiles)
- Compares winsorized vs raw returns
- Shows impact on volatility and Sharpe ratios

**`ig_vs_nig.py`** - Investment Grade vs High Yield
- Separate analysis for IG (ratings 1-10) and NIG (ratings 11-22)
- Compares rating spread performance
- Tests if effects differ by credit quality

### Rebalancing Examples

**`quarterly_rebalancing.py`** - Rebalancing Frequency
- Compares monthly vs quarterly rebalancing
- Shows turnover reduction
- Performance vs cost trade-off

## Output Format

All examples print concise results:

```
================================================================================
Strategy Name - Real Data
================================================================================

Loading data from: /path/to/data.csv
  Loaded 1,234,567 observations
  Filtered to 2002-08-31 onwards: 987,654 observations
  Unique bonds: 12,345
  Date range: 2002-08-31 to 2023-12-31
  Data shape: (987654, 25)

Forming portfolios...

Results Summary:
--------------------------------------------------------------------------------
  Sample period: 2002-09 to 2023-12
  N observations: 256

Equal-Weighted Long-Short:
  Mean:     0.123% per month
  Std:      2.456%
  Sharpe:   0.17 (annualized)
  t-stat:   0.80

Value-Weighted Long-Short:
  Mean:     0.234% per month
  Std:      3.456%
  Sharpe:   0.23 (annualized)
  t-stat:   1.09

Average Turnover: 0.154 (15.4%)

================================================================================
Complete!
================================================================================
```

## Data Loader Utilities

The `data_loader.py` module provides helper functions:

### Load Data

```python
from data_loader import load_bond_data

# Basic load
data = load_bond_data()

# Custom path
data = load_bond_data(data_path="/custom/path.csv")

# Custom start date
data = load_bond_data(start_date="2010-01-01")
```

### Sample Period

```python
from data_loader import get_sample_period

# Filter to specific years
data_subset = get_sample_period(data, start_year=2010, end_year=2020)
```

### Apply Filters

```python
from data_loader import apply_data_filters

# Rating and price filters
data_filtered = apply_data_filters(
    data,
    rating_range=(5, 15),     # Ratings 5-15 only
    min_price=70,              # Price >= $70
    max_price=130              # Price <= $130
)
```

## Key Differences from Synthetic Examples

**Real Data Characteristics:**
1. **Missing data**: Some bonds may have gaps in coverage
2. **Survivorship**: Delisted bonds may be excluded
3. **Time-varying**: Sample composition changes over time
4. **Realistic turnover**: Actual portfolio rebalancing costs
5. **Economic cycles**: Results reflect market conditions

**Synthetic Data Characteristics:**
1. **Complete data**: No missing values
2. **Controlled properties**: Known data-generating process
3. **Stable sample**: Same bonds throughout
4. **Educational**: Isolates specific effects
5. **Reproducible**: Same random seed gives same results

## Typical Performance Ranges

Based on academic literature and OSBAP data:

**Credit Rating Sorts:**
- Mean: 0.10-0.30% per month
- Sharpe: 0.15-0.35 (annualized)
- Higher in NIG than IG

**Momentum (6-1-6):**
- Mean: 0.20-0.50% per month
- Sharpe: 0.20-0.50 (annualized)
- Stronger in liquid bonds

**Turnover:**
- Monthly rebalancing: 10-25% per month
- Quarterly rebalancing: 15-30% per quarter
- Higher for momentum than rating sorts

## Best Practices

### Data Quality

1. **Check for errors**: Inspect extreme returns
2. **Winsorize returns**: Standard practice at 1%/99%
3. **Filter by price**: Exclude distressed bonds (price < $70)
4. **Require liquidity**: Filter small/illiquid issues

### Sample Selection

1. **Start date**: OSBAP data begins 2002-08-31
2. **Crisis periods**: Consider subperiod analysis (pre/post 2008)
3. **Rating coverage**: Check sufficient bonds per rating
4. **Firm coverage**: For within-firm sorts, require 2+ bonds per firm

### Performance Analysis

1. **Statistical significance**: Report t-statistics
2. **Robustness**: Test across subsamples
3. **Transaction costs**: Adjust for turnover
4. **Risk adjustment**: CAPM or multi-factor models

## Common Issues

**Issue**: "Data file not found"
- **Solution**: Update `data_path` in `data_loader.py` or pass custom path

**Issue**: "Missing required columns"
- **Solution**: Check data has `ID`, `date`, `ret`, `RATING_NUM`, `VW`

**Issue**: Very high/low Sharpe ratios
- **Solution**: Apply winsorization, check for data errors

**Issue**: Empty portfolios
- **Solution**: Reduce `num_portfolios`, widen rating filters

**Issue**: High turnover
- **Solution**: Use quarterly rebalancing or longer holding periods

## Extending Examples

### Add Custom Filters

```python
from data_loader import load_bond_data

data = load_bond_data()

# Custom filtering
data = data[data['DURATION'].between(3, 8)]  # Duration 3-8 years
data = data[data['BOND_VALUE'] > 1e6]         # Size > $1M
```

### Different Strategies

```python
import PyBondLab as pbl

# Long-term reversal
strategy = pbl.LTreversal(
    holding_period=6,
    lookback_period=36,
    skip=13,
    num_portfolios=10
)

# Within-firm sort
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='CS',
    firm_id_col='PERMNO',
    min_bonds_per_firm=2,
    rating_bins=[-np.inf, 7, 10, np.inf],
    num_portfolios=2
)
```

### Save Results

```python
# Export long-short returns
ew_ls, vw_ls = results.get_long_short()
pd.DataFrame({'EW': ew_ls, 'VW': vw_ls}).to_csv('results.csv')

# Export portfolio returns
ptf_ew, ptf_vw = results.get_ptf()
ptf_vw.to_csv('portfolio_returns.csv')
```

## References

### Data Sources
- Dickerson, Mueller, Robotti (2023): "OSBAP: Open Source Bond Asset Pricing"
- WRDS Enhanced TRACE: https://wrds-www.wharton.upenn.edu/

### Key Papers
- Bai, Bali, Wen (2019): "Common risk factors in the cross-section of corporate bond returns"
- Jostova et al. (2013): "Momentum in corporate bond returns"
- Gebhardt, Hvidkjaer, Swaminathan (2005): "Stock and bond market interaction"

## Support

For questions:
1. Check `data_loader.py` for data formatting
2. Review main `examples/README.md` for strategy details
3. See `STRUCTURE.md` for package architecture
4. Visit https://openbondassetpricing.com/ for data documentation

---

**Package**: PyBondLab v0.2.0
**Authors**: Giulio Rossetti, Alex Dickerson
**Last Updated**: 2025-01-21
