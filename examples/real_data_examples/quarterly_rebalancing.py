"""
Quarterly Rebalancing with Real Data

Demonstrates quarterly rebalancing to reduce turnover.
Compares monthly vs quarterly rebalancing frequencies.

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
print("Quarterly Rebalancing - Real Data")
print("="*80)
print()

data = load_bond_data(verbose=True)
print()

# =============================================================================
# Monthly Rebalancing
# =============================================================================

print("Monthly rebalancing:")

strategy_monthly = pbl.SingleSort(
    holding_period=1,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='monthly',
    verbose=False
)

results_monthly = pbl.StrategyFormation(
    data,
    strategy=strategy_monthly,
    turnover=True
).fit()

ew_monthly, vw_monthly = results_monthly.get_long_short()

# =============================================================================
# Quarterly Rebalancing
# =============================================================================

print("Quarterly rebalancing:")

strategy_quarterly = pbl.SingleSort(
    holding_period = 3,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_frequency='quarterly',
    rebalance_month=[1, 4, 7, 11],
    verbose=False
)

results_quarterly = pbl.StrategyFormation(
    data,
    strategy=strategy_quarterly,
    turnover=True
).fit()

ew_quarterly, vw_quarterly = results_quarterly.get_long_short()

# =============================================================================
# Annual Rebalancing
# =============================================================================

print("Quarterly rebalancing:")

strategy_annual = pbl.SingleSort(
    holding_period = 12,
    sort_var='RATING_NUM',
    num_portfolios=5,
    rebalance_month=6,
    verbose=False
)

results_annual = pbl.StrategyFormation(
    data,
    strategy=strategy_annual,
    turnover=True
).fit()

ew_annual, vw_annual = results_annual.get_long_short()

# =============================================================================
# Comparison
# =============================================================================

print()
print("Comparison: Monthly vs Quarterly vs Annual")
print("="*80)
print(f"{'Metric':<30} {'Monthly (VW)':<15} {'Quarterly (VW)':<15} {'Annual (VW)':<15}")
print("-"*80)

print(f"{'Mean (%/month)':<30} {vw_monthly.mean()*100:>14.3f} {vw_quarterly.mean()*100:>14.3f} {vw_annual.mean()*100:>14.3f}")
print(f"{'Std (%/month)':<30} {vw_monthly.std()*100:>14.3f} {vw_quarterly.std()*100:>14.3f} {vw_annual.std()*100:>14.3f}")
print(f"{'Sharpe (annualized)':<30} {vw_monthly.mean()/vw_monthly.std()*np.sqrt(12):>14.2f} {vw_quarterly.mean()/vw_quarterly.std()*np.sqrt(12):>14.2f} {vw_annual.mean()/vw_annual.std()*np.sqrt(12):>14.2f}")
print()

# Turnover
if hasattr(results_monthly.ea, 'turnover') and results_monthly.ea.turnover is not None:
    turn_monthly = results_monthly.ea.turnover.vw_turnover_df.mean().mean()
    print(f"{'Turnover (monthly)':<30} {turn_monthly:>14.3f} ({turn_monthly*100:.1f}%)")

if hasattr(results_quarterly.ea, 'turnover') and results_quarterly.ea.turnover is not None:
    turn_quarterly = results_quarterly.ea.turnover.vw_turnover_df.mean().mean()
    print(f"{'Turnover (quarterly)':<30} {turn_quarterly:>14.3f} ({turn_quarterly*100:.1f}%)")
    
if hasattr(results_annual.ea, 'turnover') and results_annual.ea.turnover is not None:
    turn_annual = results_annual.ea.turnover.vw_turnover_df.mean().mean()
    print(f"{'Turnover (annual)':<30} {turn_annual:>14.3f} ({turn_annual*100:.1f}%)")

print()
print("Note: Lower rebalancing frequency reduces turnover but may reduce performance")
print()
print("="*80)
print("Complete!")
print("="*80)
