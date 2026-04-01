# NamingConfig User Guide

`NamingConfig` is a quality-of-life feature for generating consistent, readable factor names across PyBondLab outputs. It replaces verbose internal names like `EWEA_ALL_1` with user-friendly names like `cs` or `cs_ig*`.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Reference](#api-reference)
3. [Naming Conventions](#naming-conventions)
4. [Examples](#examples)
   - [Basic Naming](#basic-naming)
   - [Rating Suffixes](#rating-suffixes)
   - [Sign Correction](#sign-correction)
   - [Weighting Prefix](#weighting-prefix)
   - [DoubleSort Naming](#doublesort-naming)
   - [WithinFirmSort Naming](#withinfirmsort-naming)
   - [Factor Turnover](#factor-turnover)
5. [Backward Compatibility](#backward-compatibility)
6. [Complete Example](#complete-example)

---

## Quick Start

```python
from PyBondLab import StrategyFormation, SingleSort, NamingConfig

# Run a strategy
strategy = SingleSort(holding_period=1, sort_var='credit_spread', num_portfolios=5)
result = StrategyFormation(data, strategy, turnover=True).fit()

# Use NamingConfig for readable names
cfg = NamingConfig()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"Factor name: {ew_ls.name}")  # Output: credit_spread

# With sign correction
cfg_sign = NamingConfig(sign_correct=True)
ew_ls, vw_ls = result.get_long_short(naming=cfg_sign)
print(f"Factor name: {ew_ls.name}")  # Output: credit_spread* (if flipped)
```

---

## API Reference

### NamingConfig

```python
from PyBondLab import NamingConfig

NamingConfig(
    lowercase: bool = True,           # Use lowercase names
    sign_correct: bool = False,       # Flip negative factors, add '*' suffix
    use_signal_name: bool = True,     # Use signal column name as base
    weighting_prefix: bool = False,   # Add 'ew_' or 'vw_' prefix
    include_rating_suffix: bool = True,  # Add '_ig' or '_nig' for rated strategies
    include_wf_suffix: bool = True,   # Add '_wf' for WithinFirmSort
    doublesort_sep: str = '_',        # Separator for DoubleSort names
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowercase` | bool | `True` | Convert signal names to lowercase (`CS` → `cs`) |
| `sign_correct` | bool | `False` | Flip negative factors to positive, add `*` suffix. Applied independently for EW and VW. |
| `use_signal_name` | bool | `True` | Use the signal column name as base (`cs` vs generic `factor`) |
| `weighting_prefix` | bool | `False` | Add `ew_` or `vw_` prefix to distinguish weightings |
| `include_rating_suffix` | bool | `True` | Add `_ig` or `_nig` for rating-filtered strategies |
| `include_wf_suffix` | bool | `True` | Add `_wf` for WithinFirmSort strategies |
| `doublesort_sep` | str | `_` | Separator for DoubleSort factor names |

---

## Naming Conventions

### Factor Names

| Strategy | Base | With Rating | With Sign Correction | With Weighting Prefix |
|----------|------|-------------|---------------------|----------------------|
| SingleSort | `cs` | `cs_ig` | `cs*` | `ew_cs`, `vw_cs` |
| DoubleSort | `cs_duration` | `cs_duration_ig` | `cs_duration*` | `ew_cs_duration` |
| WithinFirmSort | `cs_wf` | `cs_wf_ig` | `cs_wf*` | `ew_cs_wf` |

### Portfolio Names

| Strategy | Portfolio 1 | Portfolio 5 | Long/Short |
|----------|-------------|-------------|------------|
| SingleSort | `cs1` | `cs5` | - |
| DoubleSort (3×3) | `cs1_dur1` | `cs3_dur3` | - |
| WithinFirmSort | `cs_low` | `cs_high` | - |

### Factor Turnover Names

| Strategy | Factor Turnover Name |
|----------|---------------------|
| SingleSort | `cs_turnover` |
| DoubleSort | `cs_duration_turnover` |
| WithinFirmSort | `cs_wf_turnover` |

---

## Examples

### Basic Naming

```python
from PyBondLab import StrategyFormation, SingleSort, NamingConfig

# Run strategy
strategy = SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5)
result = StrategyFormation(data, strategy, turnover=True).fit()

# Default naming
cfg = NamingConfig()
ew_ls, vw_ls = result.get_long_short(naming=cfg)

print(f"EW: {ew_ls.name}")  # Output: signal1
print(f"VW: {vw_ls.name}")  # Output: signal1
```

### Rating Suffixes

```python
# Strategy with investment grade filter
result = StrategyFormation(data, strategy, rating='IG').fit()

cfg = NamingConfig()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"Factor: {ew_ls.name}")  # Output: signal1_ig

# Non-investment grade
result = StrategyFormation(data, strategy, rating='NIG').fit()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"Factor: {ew_ls.name}")  # Output: signal1_nig

# Tuple rating (custom range)
result = StrategyFormation(data, strategy, rating=(1, 10)).fit()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"Factor: {ew_ls.name}")  # Output: signal1_ig
```

### Sign Correction

Sign correction flips negative factors to positive and adds a `*` suffix. **This is applied independently for EW and VW**, so they may have different signs.

```python
cfg = NamingConfig(sign_correct=True)
ew_ls, vw_ls = result.get_long_short(naming=cfg)

print(f"EW mean: {ew_ls.mean():.4f}")  # Always positive (or zero)
print(f"VW mean: {vw_ls.mean():.4f}")  # Always positive (or zero)

# Names indicate if sign was flipped
print(f"EW: {ew_ls.name}")  # e.g., signal1* (if original was negative)
print(f"VW: {vw_ls.name}")  # e.g., signal1 (if original was positive)
```

**Example scenario:**
- Original EW mean: -0.5%, VW mean: +0.2%
- After `sign_correct=True`: EW = +0.5% (name: `signal1*`), VW = +0.2% (name: `signal1`)

### Weighting Prefix

Use `weighting_prefix=True` to distinguish EW and VW factors:

```python
cfg = NamingConfig(weighting_prefix=True)
ew_ls, vw_ls = result.get_long_short(naming=cfg)

print(f"EW: {ew_ls.name}")  # Output: ew_signal1
print(f"VW: {vw_ls.name}")  # Output: vw_signal1

# Combined with rating
result = StrategyFormation(data, strategy, rating='IG').fit()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"EW: {ew_ls.name}")  # Output: ew_signal1_ig
```

### DoubleSort Naming

```python
from PyBondLab import DoubleSort

strategy = DoubleSort(
    holding_period=1,
    sort_var='credit_spread',
    sort_var2='duration',
    num_portfolios=5,
    num_portfolios2=5,
)
result = StrategyFormation(data, strategy, turnover=True).fit()

cfg = NamingConfig()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"Factor: {ew_ls.name}")  # Output: credit_spread_duration

# Portfolio turnover with naming
ew_turn, vw_turn = result.get_turnover(level='portfolio', naming=cfg)
print(f"Columns: {list(ew_turn.columns[:3])}")
# Output: ['credit_spread1_duration1', 'credit_spread1_duration2', ...]
```

### WithinFirmSort Naming

```python
from PyBondLab import WithinFirmSort

strategy = WithinFirmSort(
    holding_period=1,
    sort_var='credit_spread',
    firm_id_col='PERMNO',
)
result = StrategyFormation(data, strategy).fit()

cfg = NamingConfig()
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"Factor: {ew_ls.name}")  # Output: credit_spread_wf

# Portfolio names for WithinFirmSort
ew_turn, vw_turn = result.get_turnover(level='portfolio', naming=cfg)
print(f"Columns: {list(ew_turn.columns)}")
# Output: ['credit_spread_low', 'credit_spread_high']
```

### Factor Turnover

Factor turnover computes the average turnover of the long and short legs:
- **SingleSort**: `(P_N + P_1) / 2`
- **DoubleSort**: Average turnover across conditioning groups for long and short legs

```python
# Get factor-level turnover (Series)
ew_turn, vw_turn = result.get_turnover(level='factor')
print(f"EW turnover shape: {ew_turn.shape}")  # (T,) - one value per date
print(f"Mean turnover: {ew_turn.mean():.2f}")

# With naming
ew_turn, vw_turn = result.get_turnover(level='factor', naming=cfg)
print(f"EW name: {ew_turn.name}")  # Output: signal1_turnover

# Get portfolio-level turnover (DataFrame)
ew_turn, vw_turn = result.get_turnover(level='portfolio', naming=cfg)
print(f"Columns: {list(ew_turn.columns)}")
# Output: ['signal11', 'signal12', 'signal13', 'signal14', 'signal15']
```

---

## Backward Compatibility

When `naming=None` (default), the original legacy names are preserved:

```python
# Legacy behavior (no naming parameter)
ew_ls, vw_ls = result.get_long_short()
print(f"EW: {ew_ls.name}")  # Output: EWEA_ALL_1

# New behavior (with naming)
ew_ls, vw_ls = result.get_long_short(naming=NamingConfig())
print(f"EW: {ew_ls.name}")  # Output: signal1
```

All existing code continues to work without changes.

---

## Complete Example

```python
import pandas as pd
from PyBondLab import StrategyFormation, SingleSort, DoubleSort, NamingConfig
from PyBondLab.pbl_test import generate_synthetic_data

# Generate sample data
data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)

# ================================================================
# Example 1: SingleSort with all naming features
# ================================================================
print("=" * 60)
print("Example 1: SingleSort with NamingConfig")
print("=" * 60)

strategy = SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5)
result = StrategyFormation(
    data=data,
    strategy=strategy,
    turnover=True,
    chars=['char1'],
    rating='IG',
).fit()

# Configuration: lowercase, sign correction, weighting prefix
cfg = NamingConfig(
    lowercase=True,
    sign_correct=True,
    weighting_prefix=True,
)

# Get long-short returns with naming
ew_ls, vw_ls = result.get_long_short(naming=cfg)
print(f"\nLong-short factors:")
print(f"  EW: {ew_ls.name} (mean: {ew_ls.mean():.4f})")
print(f"  VW: {vw_ls.name} (mean: {vw_ls.mean():.4f})")

# Get factor turnover
ew_turn, vw_turn = result.get_turnover(level='factor', naming=cfg)
print(f"\nFactor turnover:")
print(f"  EW: {ew_turn.name} (mean: {ew_turn.mean():.2f})")
print(f"  VW: {vw_turn.name} (mean: {vw_turn.mean():.2f})")

# Get characteristics with naming
ew_chars, vw_chars = result.get_characteristics(naming=cfg)
print(f"\nCharacteristics columns:")
print(f"  {list(ew_chars['char1'].columns)}")

# ================================================================
# Example 2: DoubleSort with naming
# ================================================================
print("\n" + "=" * 60)
print("Example 2: DoubleSort with NamingConfig")
print("=" * 60)

strategy2 = DoubleSort(
    holding_period=1,
    sort_var='signal1',
    sort_var2='signal2',
    num_portfolios=3,
    num_portfolios2=3,
)
result2 = StrategyFormation(data, strategy2, turnover=True).fit()

cfg2 = NamingConfig()
ew_ls, vw_ls = result2.get_long_short(naming=cfg2)
print(f"\nDoubleSorted factor: {ew_ls.name}")

# Factor turnover averages across conditioning groups
ew_turn, vw_turn = result2.get_turnover(level='factor', naming=cfg2)
print(f"Factor turnover name: {ew_turn.name}")

# ================================================================
# Example 3: Compare named vs legacy
# ================================================================
print("\n" + "=" * 60)
print("Example 3: Named vs Legacy comparison")
print("=" * 60)

strategy3 = SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5)
result3 = StrategyFormation(data, strategy3).fit()

# Legacy (no naming)
ew_legacy, _ = result3.get_long_short()
print(f"\nLegacy name: {ew_legacy.name}")

# Named
ew_named, _ = result3.get_long_short(naming=NamingConfig())
print(f"Named: {ew_named.name}")

# Both return the same data
print(f"Data identical: {(ew_legacy - ew_named).abs().max() < 1e-10}")
```

---

## Summary

| Feature | Description | Example |
|---------|-------------|---------|
| Lowercase | `CS` → `cs` | `NamingConfig(lowercase=True)` |
| Rating suffix | Investment grade / non-investment grade | `_ig`, `_nig` |
| Sign correction | Flip negative, add `*` | `cs*` (if flipped) |
| Weighting prefix | Distinguish EW/VW | `ew_cs`, `vw_cs` |
| DoubleSort | Two signals | `cs_duration` |
| WithinFirmSort | Within-firm | `cs_wf` with `_low`, `_high` portfolios |
| Factor turnover | `get_turnover(level='factor')` | `(P_N + P_1) / 2` |

The `NamingConfig` system is fully optional and backward compatible. Pass `naming=None` (default) to preserve legacy behavior.
