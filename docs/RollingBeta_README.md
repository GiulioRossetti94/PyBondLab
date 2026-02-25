# RollingBeta User Guide

`RollingBeta` estimates rolling-window factor betas on bond panels. It supports two computation engines: a fast Numba-compiled engine (~30x speedup) and a pure NumPy engine with additional features (controls, gap detection).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Reference](#api-reference)
   - [RollingBeta Parameters](#rollingbeta-parameters)
   - [compute() Method](#compute-method)
3. [Engines](#engines)
   - [Engine Selection](#engine-selection)
   - [Feature Comparison](#feature-comparison)
4. [Output Columns](#output-columns)
5. [Examples](#examples)
   - [Single Factor Beta](#single-factor-beta)
   - [Multi-Factor Betas](#multi-factor-betas)
   - [With Controls](#with-controls)
   - [Gap Detection](#gap-detection)
   - [Multiple Return Columns](#multiple-return-columns)
   - [Sort by Beta](#sort-by-beta)
6. [Complete Example](#complete-example)
7. [Summary](#summary)

---

## Quick Start

```python
import pandas as pd
from PyBondLab import RollingBeta, SingleSort, StrategyFormation

# Load factor time series (must have 'date' column + factor columns)
factors = pd.DataFrame({
    'date': pd.date_range('2010-01', periods=120, freq='ME'),
    'MKT': np.random.randn(120) * 0.05,
})

# Estimate rolling betas
beta_est = RollingBeta(factors=factors, window=36, min_periods=24)
panel = beta_est.compute(bond_data, ret_cols='ret')

# Sort bonds by market beta
strategy = SingleSort(holding_period=6, sort_var='MKT_beta_ret', num_portfolios=5)
results = StrategyFormation(panel, strategy).fit()
```

---

## API Reference

### RollingBeta Parameters

```python
from PyBondLab import RollingBeta

RollingBeta(
    factors: pd.DataFrame,              # Factor time series (must have 'date' column)
    controls: pd.DataFrame = None,      # Control variables (numpy engine only)
    window: int = 36,                   # Rolling window size
    min_periods: int = 24,              # Minimum observations before estimation starts
    add_constant: bool = True,          # Include intercept in regression
    no_gap: bool = False,               # Require consecutive months (numpy engine only)
    compute_volatility: bool = True,    # Compute total and idiosyncratic volatility
    compute_r2: bool = True,            # Compute adjusted R-squared
    engine: str = 'auto',              # 'auto', 'numba', or 'numpy'
    ridge: float = 1e-12,             # Ridge regularization (numba multi-factor only)
    verbose: bool = True,              # Print progress information
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `factors` | DataFrame | *required* | Factor time series with `date` column and one or more factor columns |
| `controls` | DataFrame | `None` | Control variables included in regression but whose betas are not primary output. Requires `engine='numpy'` |
| `window` | int | `36` | Rolling window size (number of valid observations) |
| `min_periods` | int | `24` | Minimum observations before beta estimation starts. Window expands from `min_periods` to `window`, then rolls forward |
| `add_constant` | bool | `True` | Include intercept (alpha) in regression |
| `no_gap` | bool | `False` | Require consecutive calendar months in window. Sets beta to NaN for non-consecutive windows. Requires `engine='numpy'` |
| `compute_volatility` | bool | `True` | Compute total volatility (`sigma_total`) and idiosyncratic volatility (`sigma_idio`) |
| `compute_r2` | bool | `True` | Compute adjusted R-squared for each observation |
| `engine` | str | `'auto'` | Computation engine: `'auto'` (best available), `'numba'` (fast), `'numpy'` (full features) |
| `ridge` | float | `1e-12` | Ridge regularization parameter for multi-factor regression (numba engine only, prevents singular matrices) |
| `verbose` | bool | `True` | Print initialization summary and progress |

### compute() Method

```python
panel = beta_est.compute(
    data: pd.DataFrame,           # Bond panel data
    ret_cols: str | list = 'ret', # Return column(s) to regress
    id_col: str = 'ID',           # Bond identifier column
    date_col: str = 'date',       # Date column
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | *required* | Bond panel with ID, date, and return columns |
| `ret_cols` | str or list | `'ret'` | Return column(s) to compute betas for. Multiple columns supported |
| `id_col` | str | `'ID'` | Column name for bond identifier |
| `date_col` | str | `'date'` | Column name for date |

**Returns:** `pd.DataFrame` — Original panel with added beta, volatility, and R² columns.

---

## Engines

### Engine Selection

| Engine | Speed | Controls | Gap Detection | Multi-Factor |
|--------|-------|----------|---------------|--------------|
| `numba` | ~30x faster | No | No | Yes (with ridge) |
| `numpy` | Baseline | Yes | Yes | Yes |

With `engine='auto'` (default):
1. If `controls` or `no_gap` is specified → uses `numpy`
2. If numba is installed → uses `numba`
3. Otherwise → uses `numpy`

### Feature Comparison

| Feature | numba | numpy |
|---------|-------|-------|
| Single-factor OLS | Closed-form moments (fastest) | Cumulative sums |
| Multi-factor OLS | Ring buffer with rolling moments | Cumulative outer products |
| Ridge regularization | Yes (`ridge` parameter) | No |
| Controls | Not supported (raises error) | Supported |
| Gap detection (`no_gap`) | Not supported (raises error) | Supported |

---

## Output Columns

For each factor and return column, the following columns are added to the panel:

| Column | Description | When |
|--------|-------------|------|
| `{factor}_beta_{ret_col}` | Rolling beta estimate | Always |
| `sigma_total_{ret_col}` | Total return volatility (std of returns in window) | `compute_volatility=True` |
| `sigma_idio_{ret_col}` | Idiosyncratic volatility (std of residuals) | `compute_volatility=True` |
| `adj_R2_{ret_col}` | Adjusted R-squared | `compute_r2=True` |

**Example:** With `factors` containing `['MKT', 'HML']` and `ret_cols='ret'`:
- `MKT_beta_ret`, `HML_beta_ret`, `sigma_total_ret`, `sigma_idio_ret`, `adj_R2_ret`

---

## Examples

### Single Factor Beta

```python
import numpy as np
import pandas as pd
from PyBondLab import RollingBeta

# Market factor
factors = pd.DataFrame({
    'date': pd.date_range('2005-01', periods=240, freq='ME'),
    'MKT': np.random.randn(240) * 0.04,
})

beta_est = RollingBeta(factors=factors, window=36, min_periods=24)
panel = beta_est.compute(bond_data, ret_cols='ret')

# New columns: MKT_beta_ret, sigma_total_ret, sigma_idio_ret, adj_R2_ret
print(panel[['ID', 'date', 'MKT_beta_ret', 'sigma_total_ret']].head())
```

### Multi-Factor Betas

```python
# Three-factor model
factors = pd.DataFrame({
    'date': dates,
    'MKT': mkt_returns,
    'SMB': smb_returns,
    'HML': hml_returns,
})

beta_est = RollingBeta(factors=factors, window=36, min_periods=24)
panel = beta_est.compute(bond_data, ret_cols='ret')

# New columns: MKT_beta_ret, SMB_beta_ret, HML_beta_ret, sigma_total_ret, ...
```

### With Controls

```python
# Include term structure controls (betas not primary output)
controls = pd.DataFrame({
    'date': dates,
    'TERM': term_spread,
    'DEF': default_spread,
})

beta_est = RollingBeta(
    factors=factors,
    controls=controls,      # Requires numpy engine
    window=36,
    min_periods=24,
)
panel = beta_est.compute(bond_data, ret_cols='ret')
```

### Gap Detection

```python
# Require consecutive months in each window
beta_est = RollingBeta(
    factors=factors,
    window=36,
    min_periods=24,
    no_gap=True,  # Requires numpy engine
)
panel = beta_est.compute(bond_data, ret_cols='ret')
# Bonds with non-consecutive months in their window get NaN betas
```

### Multiple Return Columns

```python
# Compute betas for both raw and excess returns
panel = beta_est.compute(bond_data, ret_cols=['ret', 'ret_excess'])

# Creates: MKT_beta_ret, MKT_beta_ret_excess, sigma_total_ret, sigma_total_ret_excess, ...
```

### Sort by Beta

```python
from PyBondLab import SingleSort, StrategyFormation

# Estimate betas
beta_est = RollingBeta(factors=factors, window=36)
panel = beta_est.compute(bond_data, ret_cols='ret')

# Form quintile portfolios sorted by market beta
strategy = SingleSort(holding_period=6, sort_var='MKT_beta_ret', num_portfolios=5)
results = StrategyFormation(panel, strategy).fit()

ew_ls, vw_ls = results.get_long_short()
print(f"EW long-short mean: {ew_ls.mean():.4f}")
print(f"VW long-short mean: {vw_ls.mean():.4f}")
```

---

## Complete Example

```python
import numpy as np
import pandas as pd
from PyBondLab import RollingBeta, SingleSort, StrategyFormation
from PyBondLab.pbl_test import generate_synthetic_data

# ================================================================
# Generate sample data and factors
# ================================================================
data = generate_synthetic_data(n_dates=120, n_bonds=500, seed=42)

dates = sorted(data['date'].unique())
factors = pd.DataFrame({
    'date': dates,
    'MKT': np.random.default_rng(42).normal(0, 0.04, len(dates)),
})

# ================================================================
# Estimate rolling market betas
# ================================================================
beta_est = RollingBeta(
    factors=factors,
    window=36,
    min_periods=24,
    compute_volatility=True,
    compute_r2=True,
)
panel = beta_est.compute(data, ret_cols='ret')

print(f"Panel shape: {panel.shape}")
print(f"Beta coverage: {panel['MKT_beta_ret'].notna().mean():.1%}")

# ================================================================
# Sort by market beta
# ================================================================
strategy = SingleSort(holding_period=1, sort_var='MKT_beta_ret', num_portfolios=5)
results = StrategyFormation(panel, strategy).fit()

ew_ls, vw_ls = results.get_long_short()
print(f"\nMarket Beta Factor:")
print(f"  EW mean: {ew_ls.mean():.4f}")
print(f"  VW mean: {vw_ls.mean():.4f}")

# ================================================================
# Sort by idiosyncratic volatility
# ================================================================
strategy_ivol = SingleSort(holding_period=1, sort_var='sigma_idio_ret', num_portfolios=5)
results_ivol = StrategyFormation(panel, strategy_ivol).fit()

ew_ivol, vw_ivol = results_ivol.get_long_short()
print(f"\nIdiosyncratic Volatility Factor:")
print(f"  EW mean: {ew_ivol.mean():.4f}")
print(f"  VW mean: {vw_ivol.mean():.4f}")
```

---

## Summary

| Feature | Description |
|---------|-------------|
| Single-factor beta | Fast closed-form OLS via numba |
| Multi-factor betas | Rolling moments with ring buffer (numba) or cumulative sums (numpy) |
| Volatility | Total (`sigma_total`) and idiosyncratic (`sigma_idio`) |
| Adjusted R² | Per-observation goodness of fit |
| Controls | Additional regressors (numpy engine only) |
| Gap detection | Require consecutive months (numpy engine only) |
| Multiple returns | Compute betas for several return columns at once |
| Engine auto-selection | Automatically picks fastest available engine |

The `RollingBeta` class integrates seamlessly with `StrategyFormation` — compute betas, then sort by any output column (`{factor}_beta_{ret}`, `sigma_idio_{ret}`, etc.).
