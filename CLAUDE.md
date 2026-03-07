# PyBondLab-Dev

## Branch: `RELEASE`

Package for portfolio formation in corporate bond markets with numba-optimized performance.

---

## Project Structure

| File | Purpose |
|------|---------|
| `PyBondLab/PyBondLab.py` | Main `StrategyFormation` class (optimized) |
| `PyBondLab/numba_core.py` | All numba kernels |
| `PyBondLab/precompute.py` | Precomputes ranks, weights |
| `PyBondLab/utils_turnover.py` | Turnover computation (fast path) |
| `PyBondLab/utils_portfolio.py` | Portfolio utilities |
| `PyBondLab/utils_within_firm.py` | WithinFirmSort core functions |
| `PyBondLab/batch.py` | `BatchStrategyFormation` (multi-signal) |
| `PyBondLab/data_uncertainty.py` | `DataUncertaintyAnalysis` wrapper |
| `PyBondLab/naming.py` | `NamingConfig` dataclass for factor naming |
| `PyBondLab/extract.py` | `extract_panel()` unified panel extraction |
| `PyBondLab/StrategyClass.py` | Strategy definitions (SingleSort, DoubleSort, WithinFirmSort) |
| `PyBondLab/StrategyResultsClass.py` | Result container classes |
| `PyBondLab/config.py` | Configuration dataclasses |
| `PyBondLab/FilterClass.py` | Data filters (do NOT modify behavior) |
| `PyBondLab/AnomalyAssayer.py` | AnomalyAssayer with `save_idx` parameter |
| `PyBondLab/pbl_test.py` | Test suite (12 tests, baseline validation) |

---

## Test Suite

```bash
python -m PyBondLab.pbl_test
```

- 12 tests, tolerance `1e-10`, baseline in `PyBondLab/baseline_results/`
- Covers SingleSort (HP=1,3), DoubleSort (cond/uncond), banding (1,2), turnover, chars

---

## Fast Path / Slow Path Behavior

All classes auto-detect whether the fast (numba) path can be used. When conditions aren't met,
they **silently fall back to the slow (pandas) path** which always produces correct results.
Users never get wrong results - only slower execution.

### Two types of filtering (important distinction)

1. **`filters` (trim/price/bounce/wins)** - Used by `StrategyFormation` and `DataUncertaintyAnalysis`.
   These create EA/EP result pairs. **Always use slow path** (fast path disabled for these).
2. **`rating` / `subset_filter`** - Used by `BatchStrategyFormation`.
   These filter at formation date only (no look-ahead bias). **Supported in fast path**.

### StrategyFormation fast path (SingleSort only)

Auto-enabled when ALL: `turnover=False`, `chars=None`, `banding=None`, no `filters`, monthly rebalancing.
If any condition fails → slow path (correct results, just slower).

### BatchStrategyFormation fast path (multi-signal)

Ultra-fast numba batch path when ALL: `turnover=False`, `chars=None`, `banding=None`.
`rating` and `subset_filter` are supported in both fast and slow paths.

### Non-Staggered Rebalancing

Fast path for `rebalance_frequency != 'monthly'` with turnover/chars/banding support (~21-103x speedup).
Returns computed EVERY month (not just at rebalancing dates). Weights renormalized on bond dropout.

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `dynamic_weights` | `False` (config) / `True` (tests/batch) | VW from d-1 (True) or formation date (False) |
| `banding` | `None` | Integer (1 or 2). Threshold = banding/nport |
| `holding_period` | 1 | HP>1 = staggered cohorts |
| `rebalance_frequency` | `'monthly'` | Also: `'quarterly'`, `'semi-annual'`, `'annual'` |

---

## Known Limitations

| Feature | Status | User Impact |
|---------|--------|-------------|
| StrategyFormation fast path + filters | Disabled | Falls back to slow path silently - correct results, just slower |
| WithinFirmSort HP>1 | Disabled | Raises ValueError at init - user gets clear error message |
| WithinFirmSort chars | Implemented | Uses `compute_within_firm_chars_aggregation` numba kernel |
| DUA fast path for non-staggered | Not implemented | Falls back to slow path with message - correct results |
| WithinFirmSort fast path | Disabled | Always uses slow path due to ranking discrepancies |

---

## API Quick Reference

### StrategyFormation
```python
from PyBondLab import StrategyFormation, SingleSort, DoubleSort
sf = StrategyFormation(data, strategy=SingleSort(holding_period=1, sort_var='signal', num_portfolios=5), turnover=True)
result = sf.fit()
ew_ls, vw_ls = result.get_long_short()
```

### DoubleSort
Use `sort_var2` (not `cond_sort_var`), `num_portfolios2`, `how='conditional'`.

### BatchStrategyFormation
```python
from PyBondLab import BatchStrategyFormation
batch = BatchStrategyFormation(data=data, signals=['s1','s2'], holding_period=1, num_portfolios=5, turnover=False)
results = batch.fit()
```

### DataUncertaintyAnalysis
```python
from PyBondLab import DataUncertaintyAnalysis
results = DataUncertaintyAnalysis(data=data, signals=['sig'], holding_periods=[1,3,6],
    filters={'trim': [0.2], 'price': [[1,5],[150,200]], 'bounce': [0.05], 'wins': [(99,'both')]},
    ratings=['IG','NIG',None], num_portfolios=5).fit()
results.summary()  # NW t-stats, means in %
```

### NamingConfig & extract_panel
```python
from PyBondLab import NamingConfig, extract_panel
panel = extract_panel(batch_results, naming=NamingConfig(sign_correct=True))
# Panel: date | factor | freq | leg (ls/l/s) | weighting (ew/vw) | return | turnover | chars...
```

### WithinFirmSort
- HP=1 only. 2 portfolios (HIGH/LOW). No banding.
- Hierarchical aggregation: within-firm VW → cap-weighted across firms → average across rating terciles.

---

## Important Rules

1. **Never change FilterClass.py behavior**
2. **Source of truth = baseline_results** - new code must match exactly
3. **JIT warmup matters** - first run includes compilation time
4. **Banding is integer** - not float, not boolean
5. **EP results** (`get_long_short_ex_post()`) only available when filters are applied
