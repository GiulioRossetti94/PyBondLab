# Strategy Examples

This folder contains examples demonstrating different portfolio formation strategies in PyBondLab.

## Examples

### Single Sorting

**`singlesort_basic.py`** - Basic single-variable portfolio sorting
- Create 5 portfolios sorted by credit rating
- Access equal-weighted and value-weighted returns
- Compute high-minus-low spreads
- **Run time**: ~5 seconds

**`singlesort_breakpoints.py`** - Custom breakpoints and universe filtering
- Unequal portfolio sizes (30-40-30 split)
- Breakpoint universe filtering (e.g., NYSE-only breakpoints)
- Decile portfolios
- **Run time**: ~10 seconds

### Double Sorting

**`doublesort_unconditional.py`** - Independent bivariate sorts
- 3×3 portfolio grid (rating × size)
- Interpreting 2D portfolio structure
- Computing spreads in each dimension
- **Run time**: ~10 seconds

**`doublesort_conditional.py`** - Sequential bivariate sorts
- Sort by rating, then size within rating groups
- Controlling for first variable
- Conditional vs unconditional comparison
- Use with correlated characteristics
- **Run time**: ~15 seconds

### Signal-Based Strategies

**`momentum.py`** - Past return momentum
- 6-1-6 specification (6mo formation, 1mo skip, 6mo holding)
- Winner-minus-loser portfolios
- Alternative specifications (short-term, long-term)
- Skip period importance
- **Run time**: ~10 seconds

**`ltreversal.py`** - Long-term reversal (contrarian)
- 36-13-6 specification
- Signal: Long-term minus recent returns
- Reversal vs momentum comparison
- Mean reversion patterns
- **Run time**: ~15 seconds

## Quick Start

All examples use synthetic data and are self-contained:

```bash
cd examples/strategies
python singlesort_basic.py
python doublesort_unconditional.py
python momentum.py
```

## Strategy Selection Guide

**Use SingleSort when:**
- Sorting on single characteristic (rating, size, yield)
- Testing univariate relationships
- Building factor portfolios

**Use DoubleSort when:**
- Testing two characteristics simultaneously
- Conditional: Control for one variable, test another
- Unconditional: Joint effects of two variables

**Use Momentum when:**
- Past returns predict future returns
- Short to medium-term continuation patterns
- Typical horizons: 3-12 months

**Use LTreversal when:**
- Testing mean reversion
- Long-term contrarian strategies
- Typical horizons: 24-60 months

## Common Patterns

### Data Requirements

All strategies require:
```python
data = pd.DataFrame({
    'date': pd.to_datetime(...),
    'ID': bond_identifiers,
    'ret': returns,
    'sort_var': characteristic_for_sorting,
    'VW': market_values  # Optional, for value-weighting
})
```

### Basic Workflow

```python
import PyBondLab as pbl

# 1. Define strategy
strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5
)

# 2. Run formation
results = pbl.StrategyFormation(data, strategy=strategy).fit()

# 3. Access results
ew_returns = results.ret_ew
vw_returns = results.ret_vw
hml = results.hml_vw
```

## Parameters

### Common to All Strategies

- `holding_period`: Portfolio holding duration (months)
- `num_portfolios`: Number of portfolios to form
- `skip`: Gap between formation and holding (optional)
- `rebalance_frequency`: 'monthly', 'quarterly', 'annual', or int
- `rebalance_month`: Month(s) for rebalancing

### Strategy-Specific

**SingleSort:**
- `sort_var`: Variable name for sorting
- `breakpoints`: Custom percentile breakpoints
- `breakpoint_universe_func`: Universe filter for breakpoints

**DoubleSort:**
- `sort_var2`: Second sorting variable
- `num_portfolios2`: Portfolios for second sort
- `how`: 'unconditional' or 'conditional'

**Momentum:**
- `lookback_period`: Formation window length
- `skip`: Typical 1 month (avoid microstructure effects)

**LTreversal:**
- `lookback_period`: Total lookback (e.g., 36 months)
- `skip`: Recent period to exclude (e.g., 13 months)

## Next Steps

- **Filtering**: See `../filtering/` for data cleaning examples
- **Rebalancing**: See `../rebalancing/` for frequency options
- **Formation Options**: See `../formation_options/` for advanced features
- **Turnover**: See `../turnover/` for portfolio tracking

## References

- Jegadeesh and Titman (1993): Momentum strategies
- DeBondt and Thaler (1985): Long-term reversal
- Fama and French (1993): Size and value sorts
- Dickerson et al. (2023): Bond market applications

---

**Package**: PyBondLab v0.2.0
**Authors**: Giulio Rossetti, Alex Dickerson
