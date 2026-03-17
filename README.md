# PyBondLab

[![PyPI version](https://img.shields.io/pypi/v/PyBondLab.svg)](https://pypi.org/project/PyBondLab/)
[![Python](https://img.shields.io/pypi/pyversions/PyBondLab.svg)](https://pypi.org/project/PyBondLab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance Python toolkit for portfolio sorting and empirical asset pricing, with a focus on corporate bonds. Part of the [Open Source Bond Asset Pricing](https://openbondassetpricing.com/) project.

> **Paper:** Dickerson, Robotti, and Rossetti (2025). *The Corporate Bond Factor Replication Crisis: A New Protocol.* [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)

---

## Installation

```bash
pip install PyBondLab
```

For numba-accelerated performance (recommended):

```bash
pip install PyBondLab[performance]
```

For WRDS data download support:

```bash
pip install PyBondLab[wrds]
```

For all optional dependencies:

```bash
pip install PyBondLab[all]
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/GiulioRossetti94/PyBondLab.git
cd PyBondLab
pip install -e ".[performance]"
```
</details>

---

## Quick Start

```python
import PyBondLab as pbl

# Sort bonds into quintile portfolios by credit spread
strategy = pbl.SingleSort(holding_period=1, sort_var='cs', num_portfolios=5)
results = pbl.StrategyFormation(data, strategy=strategy, turnover=True).fit()

# Long-short factor returns (equal- and value-weighted)
ew_ls, vw_ls = results.get_long_short()

# Turnover
ew_turn, vw_turn = results.get_turnover()
```

Your data needs columns: `date`, `ID` (bond identifier), `ret` (returns), `VW` (value weight). Use column mapping if your names differ:

```python
results = pbl.StrategyFormation(data, strategy=strategy).fit(
    IDvar='cusip', RETvar='ret_vw', VWvar='mcap_e'
)
```

---

## Features

### Single & Double Sorts

Sort the cross-section into portfolios by one or two characteristics.

```python
# Single sort: quintile portfolios
strategy = pbl.SingleSort(holding_period=1, sort_var='cs', num_portfolios=5)

# Double sort: conditional (dependent) 3x3
strategy = pbl.DoubleSort(
    holding_period=1,
    sort_var='cs', num_portfolios=3,
    sort_var2='duration', num_portfolios2=3,
    how='conditional'
)
```

Supports banding, custom breakpoints, characteristics tracking, and portfolio turnover.
See [docs/SingleSort_DoubleSort_README.md](docs/SingleSort_DoubleSort_README.md) for full API.

---

### Within-Firm Sorting

Isolate within-firm bond dispersion from cross-firm differences. Bonds are sorted into HIGH/LOW portfolios within each firm, then aggregated across firms using market-cap weighting within rating terciles.

```python
strategy = pbl.WithinFirmSort(
    holding_period=1,
    sort_var='cs',
    firm_id_col='PERMNO',
)
results = pbl.StrategyFormation(data, strategy=strategy).fit()
```

See [docs/WithinFirmSort_README.md](docs/WithinFirmSort_README.md) for methodology details.

---

### Batch Processing

Process many signals at once. Automatically selects the fastest execution path.

```python
from PyBondLab import BatchStrategyFormation

batch = BatchStrategyFormation(
    data=data,
    signals=['cs', 'ytm', 'tmat', 'mom6_1', 'val_hz'],
    holding_period=1,
    num_portfolios=5,
    turnover=False,
)
results = batch.fit()
ew_ls, vw_ls = results['cs'].get_long_short()
```

Within-firm batch:

```python
from PyBondLab import BatchWithinFirmSortFormation

batch = BatchWithinFirmSortFormation(
    data=data,
    signals=['cs', 'ytm', 'tmat'],
    firm_id_col='PERMNO',
    turnover=False,
)
results = batch.fit()
```

See [docs/BatchStrategyFormation_README.md](docs/BatchStrategyFormation_README.md) and [docs/BatchWithinFirmSortFormation_README.md](docs/BatchWithinFirmSortFormation_README.md).

---

### Rebalancing Frequencies

Monthly (default), quarterly, semi-annual, or annual. Non-monthly rebalancing computes returns every month while holding portfolio composition fixed between rebalancing dates.

```python
# Quarterly rebalancing
strategy = pbl.SingleSort(
    holding_period=3, sort_var='cs', num_portfolios=5,
    rebalance_frequency='quarterly',
)

# Annual rebalancing in June (Fama-French style)
strategy = pbl.SingleSort(
    holding_period=12, sort_var='BtM', num_portfolios=5,
    rebalance_frequency='annual',
    rebalance_month=7,  # Formation in July, returns start August
)
```

See [docs/NonStaggeredRebalancing_README.md](docs/NonStaggeredRebalancing_README.md).

---

### Custom Breakpoint Universes

Compute breakpoints on a subset (e.g., NYSE stocks) and apply them to the full cross-section.

```python
def nyse_filter(df):
    return (df['EXCHCD'] == 1) & (df['SHRCD'].isin([10, 11]))

strategy = pbl.DoubleSort(
    holding_period=12,
    sort_var='ME', sort_var2='BtM',
    num_portfolios=2, num_portfolios2=3,
    breakpoints=[50], breakpoints2=[30, 70],
    how='unconditional',
    rebalance_frequency='annual', rebalance_month=7,
    breakpoint_universe_func=nyse_filter,
    breakpoint_universe_func2=nyse_filter,
)
```

See [examples/FF3/](examples/FF3/) for a complete Fama-French replication.

---

### Data Uncertainty Analysis

Test factor robustness across data filtering configurations. Computes ex-ante and ex-post returns for each filter, with Newey-West t-statistics.

```python
from PyBondLab import DataUncertaintyAnalysis

results = DataUncertaintyAnalysis(
    data=data,
    signals=['cs', 'ytm'],
    holding_periods=[1, 3, 6],
    filters={
        'trim': [0.2, 0.5],
        'price': [[1, 5], [150, 200]],
        'bounce': [0.05, -0.05],
        'wins': [(99, 'both'), (95, 'both')],
    },
    ratings=['IG', 'NIG', None],
    num_portfolios=5,
).fit()

results.summary()        # Summary stats with NW t-statistics
results.to_excel('out.xlsx')
```

See [docs/DataUncertaintyAnalysis_README.md](docs/DataUncertaintyAnalysis_README.md).

---

### Anomaly Assaying

Test factor significance across specification choices (weighting, number of portfolios, rating subsets, breakpoint universes) following Novy-Marx and Velikov (2023).

```python
from PyBondLab import AssayAnomaly

report = AssayAnomaly(data=data, sort_var='cs', holding_periods=[1])
_, recap = report.summary_results()
print(recap)
```

See [docs/AnomalyAssay_README.md](docs/AnomalyAssay_README.md) and [docs/BatchAssayAnomaly_README.md](docs/BatchAssayAnomaly_README.md).

---

### Factor Naming & Panel Extraction

Consistent, readable factor names with optional sign correction:

```python
from PyBondLab import NamingConfig, extract_panel

# Extract all batch results into a single panel
panel = extract_panel(batch_results, naming=NamingConfig(sign_correct=True))
# Columns: date | factor | freq | leg | weighting | return | turnover | chars...
```

See [docs/NamingConfig_README.md](docs/NamingConfig_README.md).

---

## Data Filtering

Four look-ahead bias free filtering procedures for corporate bond research:

| Filter | Description | Example |
|--------|-------------|---------|
| **Trim** | Exclude extreme returns | `{'adj': 'trim', 'level': 0.2}` |
| **Price** | Exclude extreme prices | `{'adj': 'price', 'level': [20, 150]}` |
| **Bounce** | Exclude return reversals | `{'adj': 'bounce', 'level': 0.01}` |
| **Winsorize** | Cap tails at percentiles | `{'adj': 'wins', 'level': 98, 'location': 'both'}` |

```python
results = pbl.StrategyFormation(
    data, strategy=strategy,
    filters={'adj': 'trim', 'level': 0.2}
).fit()

ew_ea, vw_ea = results.get_long_short()           # Ex-ante returns
ew_ep, vw_ep = results.get_long_short_ex_post()   # Ex-post returns
```

---

## Additional Tools

| Tool | Description | Docs |
|------|-------------|------|
| `pbl.Momentum(lookback_period, skip)` | Momentum strategy from past returns | |
| `pbl.LTreversal(lookback_period, skip)` | Long-term reversal strategy | |
| `pbl.RollingBeta(factors, window)` | Rolling beta estimation (~30x with numba) | [docs](docs/RollingBeta_README.md) |
| `pbl.PreAnalysisStats(data, variables)` | Summary statistics before sorting | [docs](docs/PreAnalysisStats_README.md) |

---

## Requirements

- Python >= 3.11
- numpy < 2, pandas >= 1.5, statsmodels >= 0.14, scipy >= 1.10, pyarrow

Optional: `numba >= 0.57` (performance), `wrds` (data access)

---

## References

Dickerson, A., Robotti, C., and Rossetti, G. (2025). [The Corporate Bond Factor Replication Crisis: A New Protocol](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879). *Working Paper.*

Novy-Marx, R. and Velikov, M. (2023). [Assaying Anomalies](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4338007). *Working Paper.*

Data: [openbondassetpricing.com](https://openbondassetpricing.com/)

---

## Contact

- Giulio Rossetti -- giulio.rossetti.1@wbs.ac.uk
- Alex Dickerson -- alexander.dickerson1@unsw.edu.au

## License

MIT. See [LICENSE](LICENSE).
