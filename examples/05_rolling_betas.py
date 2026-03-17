"""
05 - Rolling Beta Estimation & Beta-Sorted Portfolios
=======================================================
Demonstrates RollingBeta for estimating rolling-window factor betas
and then sorting bonds on estimated betas to form portfolios.

PyBondLab features used:
  - RollingBeta                 (rolling OLS beta estimation)
  - compute()                   (panel-level beta computation)
  - engine='numba'/'numpy'      (engine selection)
  - compute_volatility          (residual vol estimation)
  - compute_r2                  (adjusted R-squared)
  - StrategyFormation           (sort on estimated betas)
  - SingleSort                  (sort by beta)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
from _config import load_panel, load_mktb, DATA_DIR

from PyBondLab import RollingBeta, StrategyFormation, SingleSort

data = load_panel()
mktb = load_mktb()
print(f"Panel: {len(data):,} obs")

# =============================================================================
# 1. Univariate rolling market beta (36-month window)
# =============================================================================
factors_df = mktb.to_frame('MKTB').reset_index()

beta_est = RollingBeta(
    factors=factors_df, window=36, min_periods=24,
    add_constant=True, compute_volatility=True, compute_r2=True,
    engine='auto', verbose=True,
)
panel = beta_est.compute(data=data, date_col='date', id_col='ID', ret_cols='ret')

beta_col = 'MKTB_beta_ret'
print(f"\n-- Rolling beta estimation --")
print(f"New columns: {[c for c in panel.columns if c not in data.columns]}")
print(f"Beta coverage: {panel[beta_col].notna().sum():,} / {len(panel):,} obs")
print(f"Beta mean: {panel[beta_col].mean():.3f}  std: {panel[beta_col].std():.3f}")

# Compare with pre-computed betas if available
if 'b_mktb' in panel.columns:
    valid = panel[[beta_col, 'b_mktb']].dropna()
    corr = valid[beta_col].corr(valid['b_mktb'])
    print(f"Correlation with pre-computed b_mktb: {corr:.4f}")

# =============================================================================
# 2. Sort on estimated market beta
# =============================================================================
panel_valid = panel.dropna(subset=[beta_col]).copy()

result_beta = StrategyFormation(
    data=panel_valid,
    strategy=SingleSort(sort_var=beta_col, holding_period=1, num_portfolios=5),
    dynamic_weights=True,  # no effect at HP=1 (VW identical either way); matters for HP>1
).fit()

ew_beta, vw_beta = result_beta.get_long_short()
print(f"\n-- Market beta sorted portfolios --")
print(f"EW L-S:  mean={ew_beta.mean()*100:.2f}%/mo  "
      f"t={ew_beta.mean()/ew_beta.std()*np.sqrt(len(ew_beta)):.2f}")
print(f"VW L-S:  mean={vw_beta.mean()*100:.2f}%/mo  "
      f"t={vw_beta.mean()/vw_beta.std()*np.sqrt(len(vw_beta)):.2f}")

# =============================================================================
# 3. Multi-factor rolling betas (MKTB + credit/default)
# =============================================================================
import pandas as pd
factors_full = pd.read_parquet(DATA_DIR / "bbw_factors.parquet")
factors_full.index = pd.to_datetime(factors_full.index)
factors_full = factors_full.reset_index()

factor_cols = [c for c in ['MKTB', 'CRF', 'DRF', 'LRF'] if c in factors_full.columns][:2]
if len(factor_cols) >= 2:
    panel_multi = RollingBeta(
        factors=factors_full[['date'] + factor_cols], window=36, min_periods=24,
        compute_volatility=True, compute_r2=True, engine='auto', verbose=True,
    ).compute(data=data, date_col='date', id_col='ID', ret_cols='ret')

    new_cols = [c for c in panel_multi.columns if c not in data.columns]
    print(f"\n-- Multi-factor betas ({factor_cols}) --")
    print(f"New columns: {new_cols}")
    for fc in factor_cols:
        bc = f'{fc}_beta_ret'
        if bc in panel_multi.columns:
            print(f"  {bc}: mean={panel_multi[bc].mean():.3f}  std={panel_multi[bc].std():.3f}")

# =============================================================================
# 4. Idiosyncratic volatility sorted portfolios
# =============================================================================
ivol_col = 'sigma_idio_ret'
if ivol_col in panel.columns:
    result_ivol = StrategyFormation(
        data=panel.dropna(subset=[ivol_col]).copy(),
        strategy=SingleSort(sort_var=ivol_col, holding_period=1, num_portfolios=5),
        dynamic_weights=True,
    ).fit()
    ew_ivol, vw_ivol = result_ivol.get_long_short()
    print(f"\n-- Idiosyncratic volatility sorted portfolios --")
    print(f"EW L-S:  mean={ew_ivol.mean()*100:.2f}%/mo  "
          f"t={ew_ivol.mean()/ew_ivol.std()*np.sqrt(len(ew_ivol)):.2f}")
    print(f"VW L-S:  mean={vw_ivol.mean()*100:.2f}%/mo  "
          f"t={vw_ivol.mean()/vw_ivol.std()*np.sqrt(len(vw_ivol)):.2f}")

# =============================================================================
# 5. Compare numba vs numpy engines
# =============================================================================
beta_numba = RollingBeta(factors=factors_df, window=36, min_periods=24, engine='numba')
beta_numpy = RollingBeta(factors=factors_df, window=36, min_periods=24, engine='numpy')

t0 = time.perf_counter()
p_numba = beta_numba.compute(data=data, date_col='date', id_col='ID', ret_cols='ret')
t_numba = time.perf_counter() - t0

t0 = time.perf_counter()
p_numpy = beta_numpy.compute(data=data, date_col='date', id_col='ID', ret_cols='ret')
t_numpy = time.perf_counter() - t0

valid = p_numba[[beta_col]].join(p_numpy[[beta_col]], rsuffix='_np').dropna()
corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
print(f"\n-- Engine comparison --")
print(f"Numba: {t_numba:.1f}s  |  NumPy: {t_numpy:.1f}s  |  Speedup: {t_numpy/t_numba:.1f}x")
print(f"Cross-engine correlation: {corr:.6f}")

print("\n[Done] Script 05 complete.")
