"""
02 - Double-Sort Portfolio Formation
=====================================
Demonstrates conditional and unconditional bivariate sorts using
DoubleSort, and shows how to extract the 2D portfolio grid.

PyBondLab features used:
  - StrategyFormation       (main engine)
  - DoubleSort              (bivariate sorting)
  - how='conditional'       (sequential dependent sort)
  - how='unconditional'     (independent 2D sort)
  - sort_var2               (second sort variable)
  - num_portfolios2         (second dimension size)
  - breakpoints2            (custom breakpoints for var2)
  - get_long_short()        (HML factor returns)
  - get_ptf()               (full 2D grid portfolios)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _config import load_panel

from PyBondLab import StrategyFormation, DoubleSort

data = load_panel()
print(f"Panel: {len(data):,} obs")

# =============================================================================
# 1. Conditional double sort: Momentum controlling for credit risk
# =============================================================================
result_cond = StrategyFormation(
    data=data,
    strategy=DoubleSort(holding_period=1, sort_var='mom6_1', sort_var2='cs',
                        num_portfolios=5, num_portfolios2=3, how='conditional'),
    dynamic_weights=True,
).fit()

ew_cond, vw_cond = result_cond.get_long_short()
print("\n-- Conditional: mom(6,1) | credit spread (5x3) --")
print(f"EW L-S:  mean={ew_cond.mean()*100:.2f}%/mo  "
      f"t={ew_cond.mean()/ew_cond.std()*np.sqrt(len(ew_cond)):.2f}")
print(f"VW L-S:  mean={vw_cond.mean()*100:.2f}%/mo  "
      f"t={vw_cond.mean()/vw_cond.std()*np.sqrt(len(vw_cond)):.2f}")

# =============================================================================
# 2. Unconditional double sort: value x size
# =============================================================================
result_uncond = StrategyFormation(
    data=data,
    strategy=DoubleSort(holding_period=1, sort_var='val_ipr', sort_var2='sze',
                        num_portfolios=5, num_portfolios2=3, how='unconditional'),
    dynamic_weights=True,
).fit()

ew_uncond, vw_uncond = result_uncond.get_long_short()
print("\n-- Unconditional: value (IPR) x size (5x3) --")
print(f"EW L-S:  mean={ew_uncond.mean()*100:.2f}%/mo  "
      f"t={ew_uncond.mean()/ew_uncond.std()*np.sqrt(len(ew_uncond)):.2f}")
print(f"VW L-S:  mean={vw_uncond.mean()*100:.2f}%/mo  "
      f"t={vw_uncond.mean()/vw_uncond.std()*np.sqrt(len(vw_uncond)):.2f}")

# Full 2D grid
ew_grid, vw_grid = result_uncond.get_ptf()
print(f"\nPortfolio grid shape: {ew_grid.shape[1]} columns (5x3 = 15 cells)")

# =============================================================================
# 3. Conditional with staggered holding period (HP=3)
# =============================================================================
result_hp3 = StrategyFormation(
    data=data,
    strategy=DoubleSort(holding_period=3, sort_var='mom6_1', sort_var2='RATING_NUM',
                        num_portfolios=5, num_portfolios2=3, how='conditional'),
    dynamic_weights=True,
).fit()
ew_hp3, _ = result_hp3.get_long_short()
print("\n-- Conditional: mom(6,1) | rating (5x3, HP=3) --")
print(f"EW L-S:  mean={ew_hp3.mean()*100:.2f}%/mo  "
      f"t={ew_hp3.mean()/ew_hp3.std()*np.sqrt(len(ew_hp3)):.2f}")

# =============================================================================
# 4. Unconditional with turnover and characteristics
# =============================================================================
result_chars = StrategyFormation(
    data=data,
    strategy=DoubleSort(holding_period=1, sort_var='cs', sort_var2='tmat',
                        num_portfolios=3, num_portfolios2=3, how='unconditional'),
    turnover=True, chars=['cs', 'tmat', 'sze'], dynamic_weights=True,
).fit()
ew_ds, _ = result_chars.get_long_short()
ew_to, _ = result_chars.get_turnover()
ew_ch, _ = result_chars.get_characteristics()
print("\n-- Unconditional: credit spread x maturity (3x3) with turnover & chars --")
print(f"EW L-S:  mean={ew_ds.mean()*100:.2f}%/mo")
print(f"Avg EW turnover: {ew_to.mean().mean()*100:.1f}%")
print(f"Available chars: {list(ew_ch.keys())}")

# =============================================================================
# 5. Custom breakpoints on second variable
# =============================================================================
result_custom = StrategyFormation(
    data=data,
    strategy=DoubleSort(holding_period=1, sort_var='mom6_1', sort_var2='cs',
                        num_portfolios=5, num_portfolios2=3,
                        breakpoints2=[10, 90], how='unconditional'),
    dynamic_weights=True,
).fit()
ew_custom, _ = result_custom.get_long_short()
print("\n-- Unconditional: momentum x cs (10/90 breakpoints on cs) --")
print(f"EW L-S:  mean={ew_custom.mean()*100:.2f}%/mo  "
      f"t={ew_custom.mean()/ew_custom.std()*np.sqrt(len(ew_custom)):.2f}")

# =============================================================================
# 6. Conditional with IG subsample
# =============================================================================
result_ig = StrategyFormation(
    data=data,
    strategy=DoubleSort(holding_period=1, sort_var='dvol', sort_var2='tmat',
                        num_portfolios=5, num_portfolios2=3, how='conditional'),
    rating='IG', dynamic_weights=True,
).fit()
ew_ig, _ = result_ig.get_long_short()
print("\n-- Conditional: downside vol | maturity (IG only) --")
print(f"EW L-S:  mean={ew_ig.mean()*100:.2f}%/mo  "
      f"t={ew_ig.mean()/ew_ig.std()*np.sqrt(len(ew_ig)):.2f}")

print("\n[Done] Script 02 complete.")
