"""
Investment Grade vs Non-Investment Grade Comparison

Compares rating sort performance across IG and NIG samples.
Demonstrates subset filtering by credit quality.

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
print("Investment Grade vs Non-Investment Grade - Real Data")
print("="*80)
print()

data = load_bond_data(verbose=True)
print()

# =============================================================================
# Strategy
# =============================================================================

strategy = pbl.SingleSort(
    holding_period=6,
    sort_var='RATING_NUM',
    num_portfolios=5,
    verbose=False
)

# =============================================================================
# Investment Grade (Ratings 1-10)
# =============================================================================

print("Investment Grade sample:")
results_ig = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='IG',
    turnover=False
).fit()

ew_ig, vw_ig = results_ig.get_long_short()

# =============================================================================
# Non-Investment Grade (Ratings 11-22)
# =============================================================================

print("Non-Investment Grade sample:")
results_nig = pbl.StrategyFormation(
    data,
    strategy=strategy,
    rating='NIG',
    turnover=False
).fit()

ew_nig, vw_nig = results_nig.get_long_short()

# =============================================================================
# Comparison
# =============================================================================

print()
print("Comparison: IG vs NIG")
print("="*80)
print(f"{'Metric':<30} {'IG (VW)':<15} {'NIG (VW)':<15}")
print("-"*80)

print(f"{'Mean (%/month)':<30} {vw_ig.mean()*100:>14.3f} {vw_nig.mean()*100:>14.3f}")
print(f"{'Std (%/month)':<30} {vw_ig.std()*100:>14.3f} {vw_nig.std()*100:>14.3f}")
print(f"{'Sharpe (annualized)':<30} {vw_ig.mean()/vw_ig.std()*np.sqrt(12):>14.2f} {vw_nig.mean()/vw_nig.std()*np.sqrt(12):>14.2f}")
print(f"{'t-stat':<30} {vw_ig.mean()/vw_ig.std()*np.sqrt(len(vw_ig)):>14.2f} {vw_nig.mean()/vw_nig.std()*np.sqrt(len(vw_nig)):>14.2f}")
print()

# Correlation
common_idx = vw_ig.index.intersection(vw_nig.index)
if len(common_idx) > 0:
    corr = vw_ig.loc[common_idx].corr(vw_nig.loc[common_idx])
    print(f"{'Correlation':<30} {corr:>14.3f}")
    print()

print("Note: Rating spreads often differ between IG and NIG segments")
print()
print("="*80)
print("Complete!")
print("="*80)
