# Winsorization Bias Analysis

## Overview

The `winsorization_bias_analysis.py` script analyzes the bias introduced by asymmetric return winsorization in bond factor portfolios. It compares factor returns computed with and without winsorization to quantify how data cleaning choices affect reported factor performance.

## What the Code Does

1. **Signal Classification**: Separates signals into left-tail (downside risk factors) and right-tail (momentum factors)
2. **Rating Stratification**: Runs analysis for All bonds, Investment Grade (IG), and Non-Investment Grade (NIG)
3. **Winsorization Comparison**: For each signal, computes:
   - Baseline returns (no winsorization)
   - Winsorized returns (0.5% left-tail or 99.5% right-tail)
4. **Bias Computation**: Calculates bias = winsorized - baseline for long, short, and long-short legs
5. **Alpha Computation**: Regresses factor returns on market factor (mktb) to get risk-adjusted returns
6. **Time Series Output**: Returns raw time series for further analysis

## Signal Groups

### Left-Tail Signals (0.50% winsorization)
These factors are affected by extreme negative returns:
- `b_dunc`, `b_dunc3`, `b_unc` (downside uncertainty)
- `ltr48_12`, `ltr30_6` (long-term reversal)
- `ivol_bbw`, `ivol_vp` (idiosyncratic volatility)
- `b_dvix_vp`, `b_psb_m`, `b_amd_m` (volatility-related)
- `var_95`, `es_90` (tail risk measures)

### Right-Tail Signals (99.50% winsorization)
These factors are affected by extreme positive returns:
- `mom3_1`, `mom6_1`, `mom12_1` (momentum)

## Output Structure

The `main()` function returns a dictionary with the following keys:

### Display Tables (for printing/LaTeX)

| Key Pattern | Description |
|-------------|-------------|
| `left_All`, `left_IG`, `left_NIG` | Return bias tables for left-tail signals |
| `right_All`, `right_IG`, `right_NIG` | Return bias tables for right-tail signals |
| `left_All_alpha`, `left_IG_alpha`, ... | Alpha bias tables (if mktb provided) |
| `right_All_alpha`, `right_IG_alpha`, ... | Alpha bias tables (if mktb provided) |

### Time Series DataFrames

| Key Pattern | Description |
|-------------|-------------|
| `ts_left_All`, `ts_left_IG`, `ts_left_NIG` | Dict of time series for left-tail signals |
| `ts_right_All`, `ts_right_IG`, `ts_right_NIG` | Dict of time series for right-tail signals |

Each `ts_*` key maps to a dictionary containing:

| Sub-Key | Description |
|---------|-------------|
| `long_wins` | DataFrame: Long leg returns with winsorization |
| `long_base` | DataFrame: Long leg returns without winsorization (baseline) |
| `short_wins` | DataFrame: Short leg returns with winsorization |
| `short_base` | DataFrame: Short leg returns without winsorization (baseline) |
| `ls_wins` | DataFrame: Long-short returns with winsorization |
| `ls_base` | DataFrame: Long-short returns without winsorization (baseline) |
| `bias_long` | DataFrame: Long leg bias = long_wins - long_base |
| `bias_short` | DataFrame: Short leg bias = short_wins - short_base |
| `bias_ls` | DataFrame: L-S bias = ls_wins - ls_base |

Each DataFrame has:
- **Index**: Dates (monthly)
- **Columns**: Factor names (e.g., `b_dunc`, `mom3_1`)

## Output Table Format

### Return Bias Tables

```
====================================================================================================
LEFT-TAIL ASYMMETRIC RETURN WINSORIZATION (0.50%) - All Bonds
====================================================================================================

  Legend: μ̃ = Winsorized mean (%), μ = Baseline mean (%), Bias = μ̃ - μ (%)
          Values in parentheses are NW t-statistics

               ──Long──                  ──Short─                  ───L-S──
  Factor          μ̃_L      μ_L   Bias_L      μ̃_S      μ_S   Bias_S     μ̃_LS     μ_LS  Bias_LS
  -------------------------------------------------------------------------------------------------
  b_dunc          0.58     0.51     0.07      0.50     0.45     0.05      0.08     0.06     0.02
                (3.20)   (2.80)   (1.50)    (2.10)   (1.90)   (0.80)    (1.50)   (1.20)   (0.50)
  b_dunc3         0.55     0.49     0.06      0.48     0.43     0.05      0.07     0.06     0.01
                (3.10)   (2.70)   (1.40)    (2.00)   (1.80)   (0.70)    (1.40)   (1.10)   (0.40)
  ...
```

### Column Definitions

| Column | Meaning |
|--------|---------|
| `μ̃_L` | Mean return of LONG portfolio with winsorization (%) |
| `μ_L` | Mean return of LONG portfolio without winsorization (baseline, %) |
| `Bias_L` | Bias in long leg = μ̃_L - μ_L (%) |
| `μ̃_S` | Mean return of SHORT portfolio with winsorization (%) |
| `μ_S` | Mean return of SHORT portfolio without winsorization (baseline, %) |
| `Bias_S` | Bias in short leg = μ̃_S - μ_S (%) |
| `μ̃_LS` | Mean long-short return with winsorization (%) |
| `μ_LS` | Mean long-short return without winsorization (baseline, %) |
| `Bias_LS` | Bias in long-short = μ̃_LS - μ_LS (%) |

### Alpha Bias Tables

Same structure but with α (alpha) instead of μ (mean):

| Column | Meaning |
|--------|---------|
| `α̃_L` | Alpha of LONG portfolio with winsorization (%) |
| `α_L` | Alpha of LONG portfolio without winsorization (%) |
| `Bias_L` | Alpha bias = α̃_L - α_L (%) |
| ... | (same pattern for Short and Long-Short) |

Alpha = intercept from regressing portfolio returns on mktb (market factor)

## Usage Examples

### Basic Usage

```python
from examples.winsorization_bias_analysis import main

# Run analysis (loads mktb automatically if available)
all_results = main(your_data)

# Or with custom mktb
all_results = main(your_data, mktb=your_mktb_series)
```

### Accessing Display Tables

```python
# Get return bias table for left-tail signals, IG bonds
display_table = all_results['left_IG']
print(display_table)

# Get alpha bias table
alpha_table = all_results['left_IG_alpha']
```

### Accessing Time Series

```python
# Get time series dict for left-tail, All bonds
ts = all_results['ts_left_All']

# Access individual DataFrames
ls_wins = ts['ls_wins']      # Long-short with winsorization
ls_base = ts['ls_base']      # Long-short baseline
ls_bias = ts['bias_ls']      # Pre-computed bias

# These are equivalent:
ls_bias_computed = ls_wins - ls_base
assert (ls_bias == ls_bias_computed).all().all()
```

### Computing Custom Statistics

```python
# Get time series
ts = all_results['ts_left_All']

# Individual factor analysis
b_dunc_wins = ts['ls_wins']['b_dunc']
b_dunc_base = ts['ls_base']['b_dunc']
b_dunc_bias = ts['bias_ls']['b_dunc']

# Compute statistics
print(f"Mean bias: {b_dunc_bias.mean() * 100:.2f}%")
print(f"Std bias: {b_dunc_bias.std() * 100:.2f}%")

# Rolling statistics
rolling_bias = b_dunc_bias.rolling(12).mean()

# Cumulative returns
cum_wins = (1 + b_dunc_wins).cumprod() - 1
cum_base = (1 + b_dunc_base).cumprod() - 1
```

### Exporting to Excel

```python
import pandas as pd

# Export display tables
with pd.ExcelWriter('winz_bias_results.xlsx') as writer:
    all_results['left_All'].to_excel(writer, sheet_name='Left_All')
    all_results['right_All'].to_excel(writer, sheet_name='Right_All')

    # Export time series
    ts = all_results['ts_left_All']
    ts['ls_wins'].to_excel(writer, sheet_name='TS_LS_Wins')
    ts['ls_base'].to_excel(writer, sheet_name='TS_LS_Base')
    ts['bias_ls'].to_excel(writer, sheet_name='TS_Bias_LS')
```

## Configuration

Edit these variables at the top of the script:

```python
# Paths (adjust to your environment)
BASE_REPO = Path(r"C:\Users\ASUS\Documents\GitHub\trace-data-pipeline-private")
BASE_STAGE2 = BASE_REPO / "stage2"
STAGE0_DATE_STAMP = "20251126"

# Column mapping (your data -> PyBondLab expected names)
COLUMN_MAPPING = {
    'ID': 'cusip',           # Bond identifier
    'VW': 'mcap_e',          # Market cap / value weight
    'RATING_NUM': 'spc_rat', # Rating (1-22)
    'ret': 'ret_vw'          # Return column
}

# Analysis settings
HOLDING_PERIOD = 1           # Monthly rebalancing
NUM_PORTFOLIOS = 10          # Decile portfolios
```

## Methodology

### Long and Short Legs

- **Long (L)**: Portfolio 10 (P_N) - highest signal values
- **Short (S)**: Portfolio 1 (P_1) - lowest signal values
- **Long-Short (LS)**: L - S (or P_N - P_1)

### Winsorization

- **Left-tail (0.50%)**: Clips returns below the 0.5th percentile
- **Right-tail (99.50%)**: Clips returns above the 99.5th percentile

### Bias Interpretation

- **Positive bias**: Winsorization inflates returns (removes large negative returns)
- **Negative bias**: Winsorization deflates returns (removes large positive returns)

For left-tail signals (affected by negative returns):
- Long leg bias > 0: Removing extreme losses inflates long portfolio
- Short leg bias > 0: Removing extreme losses inflates short portfolio

For right-tail signals (momentum):
- Long leg bias < 0: Removing extreme gains deflates momentum long portfolio
- Short leg bias: Usually smaller effect

### T-Statistics

All t-statistics use Newey-West HAC standard errors with lag = T^0.25 to account for autocorrelation and heteroskedasticity.

## Notes for LaTeX Integration

The display tables are designed for easy conversion to LaTeX:

1. **Two-row format**: Each factor has means row + t-stats row
2. **Parentheses**: T-stats are pre-formatted with parentheses
3. **Consistent width**: All columns are 8 characters wide
4. **Grouping**: Columns are grouped into Long, Short, Long-Short sections

To convert to LaTeX:
1. Parse the DataFrame rows
2. Odd rows = means (numeric)
3. Even rows = t-stats (string with parentheses)
4. Use `\multicolumn` for section headers

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `statsmodels`: Newey-West standard errors, OLS regression
- `PyBondLab`: DataUncertaintyAnalysis for portfolio formation
