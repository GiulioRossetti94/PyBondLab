"""
Double Sort with Real Data

Demonstrates bivariate sorting (rating × size) using real bond data.
Independent 3×3 sort on credit rating and bond size.

Data: OSBAP or WRDS Enhanced TRACE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import PyBondLab as pbl
from data_loader import load_bond_data

# =============================================================================
# Load Data
# =============================================================================

print("="*80)
print("Double Sort (Rating × Size) - Real Data")
print("="*80)
print()

data = load_bond_data(verbose=True)
print()

# =============================================================================
# Strategy: Double Sort (Unconditional)
# =============================================================================

print("Forming portfolios...")

strategy = pbl.DoubleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    sort_var2='BOND_VALUE',
    num_portfolios=3,
    num_portfolios2=3,
    how='unconditional',
    verbose=False
)

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    turnover=False
).fit()

# =============================================================================
# Results
# =============================================================================

ew_ls, vw_ls = results.get_long_short()

print()
print("Results Summary:")
print("-"*80)
print(f"  Strategy: Rating × Size (3×3 unconditional)")
print(f"  Sample period: {ew_ls.index[0].strftime('%Y-%m')} to {ew_ls.index[-1].strftime('%Y-%m')}")
print(f"  N observations: {len(ew_ls)}")
print()
print(f"Equal-Weighted Long-Short:")
print(f"  Mean:   {ew_ls.mean()*100:>7.3f}% per month")
print(f"  Std:    {ew_ls.std()*100:>7.3f}%")
print(f"  Sharpe: {ew_ls.mean() / ew_ls.std() * np.sqrt(12):>7.2f} (annualized)")
print(f"  t-stat: {ew_ls.mean() / ew_ls.std() * np.sqrt(len(ew_ls)):>7.2f}")
print()
print(f"Value-Weighted Long-Short:")
print(f"  Mean:   {vw_ls.mean()*100:>7.3f}% per month")
print(f"  Std:    {vw_ls.std()*100:>7.3f}%")
print(f"  Sharpe: {vw_ls.mean() / vw_ls.std() * np.sqrt(12):>7.2f} (annualized)")
print(f"  t-stat: {vw_ls.mean() / vw_ls.std() * np.sqrt(len(vw_ls)):>7.2f}")
print()

print("="*80)
print("Complete!")
print("="*80)
