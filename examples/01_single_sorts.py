"""
01 - Single-Sort Portfolio Formation
=====================================
Demonstrates StrategyFormation with SingleSort, Momentum, and LTreversal
strategies across holding periods, weighting schemes, turnover, and
portfolio characteristics tracking.

PyBondLab features used:
  - StrategyFormation           (main formation engine)
  - SingleSort                  (pre-computed signal sort)
  - Momentum                    (built-in momentum signal with skip)
  - LTreversal                  (long-term reversal with skip)
  - FormationResults            (result container)
  - StrategyFormationConfig     (config dataclass)
  - get_long_short()            (EW/VW long-short returns)
  - get_turnover()              (portfolio turnover)
  - get_characteristics()       (characteristic tracking)
  - get_ptf()                   (full portfolio returns)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _config import load_panel, load_mktb

from PyBondLab import StrategyFormation, SingleSort, Momentum, LTreversal
from PyBondLab.config import StrategyFormationConfig, DataConfig, FormationConfig

# -- Load data (columns already renamed to PyBondLab standard names) ----------
data = load_panel()
mktb = load_mktb()
print(f"Panel: {len(data):,} obs, {data['ID'].nunique():,} bonds, "
      f"{data['date'].nunique()} months")

# =============================================================================
# 1. Basic SingleSort: Credit Spread (HP=1, quintiles)
# =============================================================================
strategy = SingleSort(sort_var='cs', holding_period=1, num_portfolios=5)
result = StrategyFormation(
    data=data, strategy=strategy,
    turnover=True, chars=['tmat', 'cs', 'sze'],
    dynamic_weights=True,  # no effect at HP=1 (VW identical either way); matters for HP>1
).fit()

ew_ls, vw_ls = result.get_long_short()
ew_ptf, vw_ptf = result.get_ptf()
ew_to, vw_to = result.get_turnover()
ew_chars, vw_chars = result.get_characteristics()

print("\n-- Credit spread sort (HP=1, 5 portfolios) --")
print(f"EW L-S:  mean={ew_ls.mean()*100:.2f}%/mo  std={ew_ls.std()*100:.2f}%  "
      f"t={ew_ls.mean()/ew_ls.std()*np.sqrt(len(ew_ls)):.2f}")
print(f"VW L-S:  mean={vw_ls.mean()*100:.2f}%/mo  std={vw_ls.std()*100:.2f}%  "
      f"t={vw_ls.mean()/vw_ls.std()*np.sqrt(len(vw_ls)):.2f}")
print(f"Avg EW turnover: {ew_to.mean().mean()*100:.1f}%")
print(f"VW chars (cs) spread P5-P1: "
      f"{vw_chars['cs'].iloc[:, -1].mean() - vw_chars['cs'].iloc[:, 0].mean():.1f}")

# =============================================================================
# 2. Staggered holding period (HP=3)
# =============================================================================
result_hp3 = StrategyFormation(
    data=data,
    strategy=SingleSort(sort_var='cs', holding_period=3, num_portfolios=5),
    turnover=True, dynamic_weights=True,
).fit()

ew3, vw3 = result_hp3.get_long_short()
print("\n-- Credit spread sort (HP=3, staggered cohorts) --")
print(f"EW L-S:  mean={ew3.mean()*100:.2f}%/mo  t={ew3.mean()/ew3.std()*np.sqrt(len(ew3)):.2f}")
print(f"VW L-S:  mean={vw3.mean()*100:.2f}%/mo  t={vw3.mean()/vw3.std()*np.sqrt(len(vw3)):.2f}")

# =============================================================================
# 3. Momentum (6,1) with built-in signal construction
# =============================================================================
result_mom = StrategyFormation(
    data=data,
    strategy=Momentum(lookback_period=6, skip=1, holding_period=1, num_portfolios=5),
    turnover=True, dynamic_weights=True,
).fit()

ew_mom, vw_mom = result_mom.get_long_short()
print("\n-- Momentum (6,1) --")
print(f"EW L-S:  mean={ew_mom.mean()*100:.2f}%/mo  t={ew_mom.mean()/ew_mom.std()*np.sqrt(len(ew_mom)):.2f}")
print(f"VW L-S:  mean={vw_mom.mean()*100:.2f}%/mo  t={vw_mom.mean()/vw_mom.std()*np.sqrt(len(vw_mom)):.2f}")

# =============================================================================
# 4. Long-Term Reversal (48,12)
# =============================================================================
result_ltr = StrategyFormation(
    data=data,
    strategy=LTreversal(lookback_period=48, skip=12, holding_period=1, num_portfolios=5),
    dynamic_weights=True,
).fit()

ew_ltr, vw_ltr = result_ltr.get_long_short()
print("\n-- Long-term reversal (48,12) --")
print(f"EW L-S:  mean={ew_ltr.mean()*100:.2f}%/mo  t={ew_ltr.mean()/ew_ltr.std()*np.sqrt(len(ew_ltr)):.2f}")
print(f"VW L-S:  mean={vw_ltr.mean()*100:.2f}%/mo  t={vw_ltr.mean()/vw_ltr.std()*np.sqrt(len(vw_ltr)):.2f}")

# =============================================================================
# 5. SingleSort with banding (threshold = 1/nport)
# =============================================================================
ami_strategy = SingleSort(sort_var='ami', holding_period=1, num_portfolios=5)

result_band = StrategyFormation(
    data=data, strategy=ami_strategy,
    turnover=True,
    banding_threshold=1/5,  # StrategyFormation uses float; BatchStrategyFormation uses banding=1 (int)
    dynamic_weights=True,
).fit()
ew_band, _ = result_band.get_long_short()
ew_to_band, _ = result_band.get_turnover()

result_noband = StrategyFormation(
    data=data, strategy=ami_strategy,
    turnover=True, dynamic_weights=True,
).fit()
ew_nb, _ = result_noband.get_long_short()
ew_to_nb, _ = result_noband.get_turnover()

print("\n-- Amihud illiquidity: banding vs no-banding --")
print(f"No band  - turnover: {ew_to_nb.mean().mean()*100:.1f}%  mean L-S: {ew_nb.mean()*100:.2f}%/mo")
print(f"Banding  - turnover: {ew_to_band.mean().mean()*100:.1f}%  mean L-S: {ew_band.mean()*100:.2f}%/mo")

# =============================================================================
# 6. Using StrategyFormationConfig (explicit configuration)
# =============================================================================
config = StrategyFormationConfig(
    data=DataConfig(rating='IG', chars=['tmat', 'cs']),
    # Note: compute_turnover in FormationConfig corresponds to turnover= in kwargs
    formation=FormationConfig(dynamic_weights=True, compute_turnover=True, verbose=False),
)
result_cfg = StrategyFormation(
    data=data,
    strategy=SingleSort(sort_var='dvol', holding_period=1, num_portfolios=5),
    config=config,
).fit()
ew_cfg, vw_cfg = result_cfg.get_long_short()
print(f"\n-- Downside vol sort (IG only, via config) --")
print(f"EW L-S:  mean={ew_cfg.mean()*100:.2f}%/mo  t={ew_cfg.mean()/ew_cfg.std()*np.sqrt(len(ew_cfg)):.2f}")

# =============================================================================
# 7. Custom breakpoints (30/70 split)
# =============================================================================
result_bp = StrategyFormation(
    data=data,
    strategy=SingleSort(sort_var='cs', holding_period=1, num_portfolios=3, breakpoints=[30, 70]),
    dynamic_weights=True,
).fit()
ew_bp, _ = result_bp.get_long_short()
print(f"\n-- Credit spread (30/70 breakpoints) --")
print(f"EW L-S:  mean={ew_bp.mean()*100:.2f}%/mo  t={ew_bp.mean()/ew_bp.std()*np.sqrt(len(ew_bp)):.2f}")

# =============================================================================
# 8. IG-only breakpoints, all bonds sorted
# =============================================================================
def bp_ig_only(df):
    return (df['RATING_NUM'] >= 1) & (df['RATING_NUM'] <= 10)

result_igbp = StrategyFormation(
    data=data,
    strategy=SingleSort(sort_var='cs', holding_period=1, num_portfolios=5,
                        breakpoint_universe_func=bp_ig_only),
    dynamic_weights=True,
).fit()
ew_igbp, _ = result_igbp.get_long_short()
print(f"\n-- Credit spread (IG-only breakpoints) --")
print(f"EW L-S:  mean={ew_igbp.mean()*100:.2f}%/mo  t={ew_igbp.mean()/ew_igbp.std()*np.sqrt(len(ew_igbp)):.2f}")

# =============================================================================
# 9. Subset filter (maturity 0-5 years)
# =============================================================================
result_short = StrategyFormation(
    data=data,
    strategy=SingleSort(sort_var='cs', holding_period=1, num_portfolios=5),
    subset_filter={'tmat': (0, 5)}, dynamic_weights=True,
).fit()
ew_short, _ = result_short.get_long_short()
print(f"\n-- Credit spread (short maturity 0-5y) --")
print(f"EW L-S:  mean={ew_short.mean()*100:.2f}%/mo  t={ew_short.mean()/ew_short.std()*np.sqrt(len(ew_short)):.2f}")

# =============================================================================
# 10. Non-staggered quarterly rebalancing
# =============================================================================
result_q = StrategyFormation(
    data=data,
    strategy=SingleSort(sort_var='cs', num_portfolios=5,
                        rebalance_frequency='quarterly',  # actual holding = 3 months
                        rebalance_month=[3, 6, 9, 12]),   # overrides default schedule
    dynamic_weights=True,
).fit()
ew_q, _ = result_q.get_long_short()
print(f"\n-- Credit spread (quarterly rebalancing) --")
print(f"EW L-S:  mean={ew_q.mean()*100:.2f}%/mo  t={ew_q.mean()/ew_q.std()*np.sqrt(len(ew_q)):.2f}")

print("\n[Done] Script 01 complete.")
