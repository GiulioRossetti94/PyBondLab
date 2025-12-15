# Formation Options Examples

This folder contains examples demonstrating advanced StrategyFormation parameters for portfolio customization.

## Examples

### `characteristics_tracking.py` - Track Portfolio Characteristics
Monitor portfolio characteristics over time (duration, OAS, YTM, etc.).

**Covered:**
- Specifying characteristics to track
- Time-series of portfolio characteristics
- Cross-sectional analysis
- Practical applications (duration-neutral, adjusted returns)

**Run time**: ~10 seconds

### `subset_filtering.py` - Filter Bond Universe
Filter bonds by rating or characteristics before portfolio formation.

**Covered:**
- Rating filtering (IG, NIG, custom range)
- Characteristic filtering (duration, size)
- Multiple simultaneous filters
- Comparison across filters

**Run time**: ~15 seconds

## Quick Start

```bash
cd examples/formation_options
python characteristics_tracking.py
python subset_filtering.py
```

## Characteristics Tracking

### Usage

```python
import PyBondLab as pbl

strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5
)

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    chars=['DURATION', 'OAS', 'YTM'],  # Track these characteristics
    turnover=False
).fit()

# Access characteristics
duration_by_portfolio = results.chars['DURATION']
```

### Common Characteristics

- **DURATION**: Interest rate sensitivity
- **OAS**: Option-adjusted spread
- **YTM**: Yield to maturity
- **RATING_NUM**: Credit quality
- **BOND_VALUE**: Market value/size
- **AGE**: Time since issuance
- **AMOUNT_OUTSTANDING**: Issuance size

### Applications

1. **Duration-Neutral Strategies**: Track and hedge duration differences
2. **Characteristic-Adjusted Returns**: Control for characteristic exposure
3. **Portfolio Monitoring**: Verify sorting works as intended
4. **Risk Management**: Monitor exposure drift

## Subset Filtering

### Rating Filters

```python
# Investment grade only
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='IG'                        # Ratings 1-10
).fit()

# Non-investment grade (high yield)
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='NIG'                       # Ratings 11-22
).fit()

# Custom range
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating=(5, 15)                     # Ratings 5-15
).fit()
```

### Characteristic Filters

```python
# Filter by duration
subset_filter = {
    'DURATION': (3, 8)                 # Duration between 3 and 8 years
}

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    subset_filter=subset_filter
).fit()

# Multiple filters
subset_filter = {
    'DURATION': (3, 8),
    'BOND_VALUE': (1e6, 1e9)           # Size between $1M and $1B
}

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='IG',                       # Plus rating filter
    subset_filter=subset_filter
).fit()
```

### Common Use Cases

| Filter | Purpose | Example |
|--------|---------|---------|
| `rating='IG'` | Investment grade analysis | Focus on safer bonds |
| `rating='NIG'` | High yield analysis | Focus on risky bonds |
| `DURATION: (4, 6)` | Control rate sensitivity | Narrow duration range |
| `BOND_VALUE: (p25, p75)` | Liquid bonds only | Exclude micro-cap bonds |
| `AGE: (1, 120)` | Exclude new/old bonds | Standard maturity range |

## Key Differences

### Characteristics Tracking vs Subset Filtering

**Characteristics Tracking** (`chars` parameter):
- Monitors portfolio characteristics over time
- Does not exclude data
- For analysis and risk management
- Example: Track duration exposure

**Subset Filtering** (`rating`, `subset_filter` parameters):
- Excludes bonds from universe before sorting
- Reduces sample size
- For focusing analysis
- Example: IG-only analysis

### Subset Filtering vs Return Filtering

**Subset Filtering** (formation_options):
- Removes bonds from universe
- Applied before portfolio formation
- Based on characteristics
- Example: `rating='IG'` removes all NIG bonds

**Return Filtering** (filtering folder):
- Adjusts or removes specific returns
- Applied to return data
- Based on return values
- Example: Trim returns > 30%

## Complete Example

```python
import PyBondLab as pbl

# Strategy with all options
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5
)

# Comprehensive configuration
results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    # Subset filtering
    rating='IG',                       # Investment grade only
    subset_filter={
        'DURATION': (3, 8),            # Duration range
        'BOND_VALUE': (1e6, 1e9)       # Size range
    },
    # Characteristics tracking
    chars=['DURATION', 'OAS', 'YTM'],  # Track these
    # Other options
    turnover=True,
    verbose=True
).fit()

# Access results
portfolio_returns = results.ret_vw
hml_spread = results.hml_vw
duration_by_portfolio = results.chars['DURATION']
turnover_stats = results.turnover
```

## Best Practices

1. **Start broad, then narrow**: Run full sample first, then apply filters
2. **Document filters**: Note filters in research for reproducibility
3. **Check sample size**: Ensure sufficient observations after filtering
4. **Compare filtered vs unfiltered**: Understand filter impact
5. **Use standard filters**: IG/NIG split is common in literature

## References

- Fama and French (1993): Separate analysis by size/rating
- Bai, Bali, Wen (2019): Duration-neutral bond factors
- Jostova et al. (2013): IG vs HY momentum differences

---

**Package**: PyBondLab v0.2.0
**Authors**: Giulio Rossetti, Alex Dickerson
