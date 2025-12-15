# Rebalancing Frequency Examples

This folder contains examples demonstrating different portfolio rebalancing frequencies in PyBondLab.

## Examples

### `quarterly_rebalancing.py` - Quarterly Rebalancing
Rebalance portfolios every 3 months (4 times per year).

**Covered:**
- Standard quarters (Mar, Jun, Sep, Dec)
- Offset quarters (Feb, May, Aug, Nov)
- Comparison with monthly rebalancing
- Holding period vs rebalancing frequency
- Different holding periods with quarterly rebalancing

**Run time**: ~15 seconds

### `annual_rebalancing.py` - Annual Rebalancing
Rebalance portfolios once per year (minimal turnover).

**Covered:**
- Different rebalancing months (Jan, Jun, Dec)
- Frequency comparison (monthly vs quarterly vs annual)
- Different holding periods with annual rebalancing
- Practical month selection
- Transaction cost implications

**Run time**: ~20 seconds

### `custom_frequency.py` - Custom Intervals
Rebalance at custom intervals (e.g., every 4, 5, or 8 months).

**Covered:**
- 4-month rebalancing
- Different starting months
- Various custom frequencies (2, 4, 5, 8, 10 months)
- Matching holding period to frequency
- Use cases and practical considerations
- Complete frequency spectrum

**Run time**: ~15 seconds

## Quick Start

All examples use synthetic data:

```bash
cd examples/rebalancing
python quarterly_rebalancing.py
python annual_rebalancing.py
python custom_frequency.py
```

## Rebalancing Frequency Guide

### Standard Frequencies

| Frequency | Code | Rebalances/Year | Typical Use |
|-----------|------|-----------------|-------------|
| Monthly | `'monthly'` | 12 | Default, research standard |
| Quarterly | `'quarterly'` | 4 | Institutional, fiscal quarters |
| Semi-annual | `'semi-annual'` | 2 | Low-turnover strategies |
| Annual | `'annual'` | 1 | Minimal turnover, buy-and-hold |

### Custom Frequencies

| Interval | Code | Rebalances/Year | Use Case |
|----------|------|-----------------|----------|
| 2 months | `2` | 6 | Between monthly and quarterly |
| 4 months | `4` | 3 | Between quarterly and semi-annual |
| 5 months | `5` | 2.4 | Non-standard, research |
| 8 months | `8` | 1.5 | Low-turnover alternative |

## Basic Usage

### Standard Frequency

```python
import PyBondLab as pbl

# Quarterly rebalancing
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',    # Standard frequency
    rebalance_month=[3, 6, 9, 12]      # Mar, Jun, Sep, Dec
)

results = pbl.StrategyFormation(data, strategy=strategy).fit()
```

### Custom Frequency

```python
# Rebalance every 4 months starting in January
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency=4,              # Custom: every 4 months
    rebalance_month=1                   # Start in January
)

results = pbl.StrategyFormation(data, strategy=strategy).fit()
```

## Key Concepts

### Holding Period vs Rebalancing Frequency

**Holding Period**: How long to hold each portfolio
**Rebalancing Frequency**: How often to form new portfolios

#### Example 1: Overlapping Portfolios
```python
holding_period=6, rebalance_frequency='quarterly'
# Form portfolio every 3 months
# Hold each portfolio for 6 months
# Result: 2 portfolios active at any time (overlapping)
```

#### Example 2: Non-Overlapping Portfolios
```python
holding_period=3, rebalance_frequency='quarterly'
# Form portfolio every 3 months
# Hold each portfolio for 3 months
# Result: No overlap between portfolios
```

#### Example 3: Gaps Between Portfolios
```python
holding_period=6, rebalance_frequency='annual'
# Form portfolio once per year
# Hold each portfolio for 6 months
# Result: 6 months gap with no active portfolio
```

### Staggered vs Non-Staggered

**Monthly (staggered):**
- Default PyBondLab behavior
- Multiple overlapping cohorts
- Smoother return series
- Higher turnover

**Non-monthly (non-staggered):**
- Discrete rebalancing dates
- No overlapping cohorts
- Lower turnover
- More realistic for large portfolios

## Rebalancing Month Selection

### Quarterly
- **[3, 6, 9, 12]**: Standard fiscal quarters (most common)
- **[2, 5, 8, 11]**: Offset quarters (alternative)
- **[1, 4, 7, 10]**: January-based quarters

### Semi-Annual
- **[6, 12]**: Mid-year and year-end
- **[3, 9]**: Quarter-end spacing
- **[1, 7]**: Start and mid-year

### Annual
- **6 (June)**: Mid-year, academic year (common)
- **12 (December)**: Calendar year-end
- **1 (January)**: Year start, budget cycles

### Custom
- **Starting month**: First rebalancing month
- Schedule repeats at custom interval
- Example: frequency=4, month=1 → Jan, May, Sep, Jan, ...

## Performance Considerations

### Transaction Costs

Assuming 20% turnover per rebalance and 50 bps transaction costs:

| Frequency | Rebalances/Year | Annual Cost |
|-----------|-----------------|-------------|
| Monthly | 12 | 120 bps |
| Bi-monthly | 6 | 60 bps |
| Quarterly | 4 | 40 bps |
| Semi-annual | 2 | 20 bps |
| Annual | 1 | 10 bps |

**Cost savings** from quarterly vs monthly: ~80 bps/year

### Strategy Performance

Empirical findings:
- Monthly rebalancing captures short-term signals better
- Quarterly/annual reduce turnover with minimal performance loss
- Optimal frequency depends on:
  - Signal persistence
  - Transaction costs
  - Market liquidity
  - Implementation constraints

## Frequency Selection Guide

**Use MONTHLY when:**
- Short-term signals (momentum, reversals)
- Low transaction costs
- Research baseline/comparison
- Liquid markets

**Use QUARTERLY when:**
- Moderate signal persistence
- Fiscal quarter alignment
- Institutional implementation
- Balance of costs and performance

**Use ANNUAL when:**
- Long-term signals (value, quality)
- High transaction costs
- Buy-and-hold strategies
- Minimal turnover requirement

**Use CUSTOM when:**
- Fine-tuning cost-return trade-off
- Non-standard reporting cycles
- Research robustness checks
- Avoiding crowded rebalancing dates

## Common Patterns

### Pattern 1: Monthly Holding, Quarterly Rebalancing
```python
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rebalance_month=[3, 6, 9, 12]
)
```
**Result**: Overlapping 6-month portfolios, rebalanced quarterly

### Pattern 2: Non-Overlapping Quarterly
```python
strategy = pbl.SingleSort(
    holding_period=3,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rebalance_month=[3, 6, 9, 12]
)
```
**Result**: Clean 3-month portfolios, no overlap

### Pattern 3: Annual with Long Holding
```python
strategy = pbl.SingleSort(
    holding_period=12,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=6
)
```
**Result**: Buy-and-hold for 12 months, rebalance annually

## Literature Standards

### Equity Research
- **Monthly**: Standard for momentum, short-term reversals
- **Annual**: Value, profitability factors (Fama-French)

### Bond Research
- **Monthly**: Common for comparability
- **Quarterly**: More realistic for corporate bonds
- **Semi-annual/Annual**: Large portfolios, illiquid bonds

### Recent Trends
- Move toward quarterly/annual for realism
- Recognition of transaction costs in illiquid markets
- Robustness checks across frequencies

## Advanced Usage

### Multiple Strategies with Different Frequencies

```python
# Short-term momentum: monthly
strategy_st = pbl.Momentum(
    holding_period=3,
    lookback_period=3,
    skip=1,
    num_portfolios=10,
    rebalance_frequency='monthly'
)

# Long-term value: annual
strategy_lt = pbl.SingleSort(
    holding_period=12,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=6
)

results_st = pbl.StrategyFormation(data, strategy=strategy_st).fit()
results_lt = pbl.StrategyFormation(data, strategy=strategy_lt).fit()
```

### Sensitivity Analysis

Test strategy across multiple frequencies:

```python
frequencies = ['monthly', 'quarterly', 'semi-annual', 'annual']
results = {}

for freq in frequencies:
    strategy = pbl.SingleSort(
        holding_period=12,
        sort_var='RATING_NUM',
        num_portfolios=5,
        rebalance_frequency=freq,
        rebalance_month=6 if freq == 'annual' else [3,6,9,12]
    )
    results[freq] = pbl.StrategyFormation(data, strategy=strategy).fit()

# Compare Sharpe ratios
for freq, res in results.items():
    sharpe = res.hml_vw.mean() / res.hml_vw.std()
    print(f"{freq}: {sharpe:.3f}")
```

## Validation

### Check Rebalancing Dates

After running StrategyFormation, verify rebalancing occurs on expected dates by examining portfolio returns:

```python
results = pbl.StrategyFormation(data, strategy=strategy).fit()

# Check dates
print(results.ret_vw.index)

# For non-monthly: gaps should appear between rebalancing periods
```

## Next Steps

- **Strategies**: See `../strategies/` for portfolio formation methods
- **Filtering**: See `../filtering/` for data cleaning
- **Formation Options**: See `../formation_options/` for advanced features
- **Turnover**: See `../turnover/` for turnover diagnostics

## References

- Jegadeesh and Titman (1993): Monthly rebalancing standard
- Fama and French (1993): Annual for value factors
- Novy-Marx (2013): Sensitivity to rebalancing frequency
- Dickerson et al. (2023): Quarterly for corporate bonds

---

**Package**: PyBondLab v0.2.0
**Authors**: Giulio Rossetti, Alex Dickerson
