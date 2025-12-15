"""
Rating Sort with Real Data

Demonstrates credit rating sorts using real bond data.
Sorts bonds by credit rating to form long-short portfolios.

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
print("Rating Sort Strategy - Real Data")
print("="*80)
print()

data = load_bond_data(verbose=True)
print()

# =============================================================================
# Strategy: Rating Sort (5 Portfolios, 6-Month Holding)
# =============================================================================

print("Forming portfolios...")

strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    verbose=False
)

results = pbl.StrategyFormation(
    data,
    strategy=strategy,
    turnover=True
).fit()

# =============================================================================
# Results
# =============================================================================

# Long-short returns
ew_ls, vw_ls = results.get_long_short()

print()
print("Results Summary:")
print("-"*80)
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

# Turnover
if hasattr(results.ea, 'turnover') and results.ea.turnover is not None:
    avg_turnover = results.ea.turnover.vw_turnover_df.mean().mean()
    print(f"Average Turnover: {avg_turnover:.3f} ({avg_turnover*100:.1f}%)")
    print()

print("="*80)
print("Complete!")
print("="*80)
