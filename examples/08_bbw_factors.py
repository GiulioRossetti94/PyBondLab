"""
08 - BBW Factor Replication (Bai, Bali & Wen, 2019 JFE)
=========================================================
Replicates the four-factor model for corporate bonds:

  MKTB: Value-weighted bond market excess return
  DRF:  Default risk factor (5x5 rating x VaR, P5-P1 averaged across ratings)
  LRF:  Liquidity risk factor (5x5 rating x illiquidity, same approach)
  CRF:  Credit risk factor (NIG-IG spread averaged across signal bins & signals)

Methodology:
  For each signal s in {VaR, illiquidity, reversal}, form 5x5 unconditional
  double-sorted VW portfolios on RATING_NUM x s. Then:
    DRF  = avg across 5 rating groups of (top - bottom quintile on VaR)
    LRF  = avg across 5 rating groups of (top - bottom quintile on illiquidity)
    CRF  = avg across 5 signal bins and all 3 signals of (NIG - IG spread)
    MKTB = lagged-VW bond market return (excess of risk-free rate)

Signal mapping (Byungmin -> DRR panel):
  var   -> var_95  (95th-percentile Value-at-Risk)
  illiq -> ilq     (illiquidity; DRR panel lacks trade count for n>=5 filter)
  strev -> str     (short-term reversal = lagged bond return)

PyBondLab features used:
  - StrategyFormation      (main formation engine)
  - DoubleSort             (5x5 unconditional bivariate sort)
  - get_ptf()              (extract 2D portfolio grid, VW returns)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from _config import load_panel, MKTB_PATH

import PyBondLab as pbl

data = load_panel()
print(f"Panel: {len(data):,} obs, {data['ID'].nunique():,} bonds, "
      f"{data['date'].nunique()} months")

# =============================================================================
# Signal definitions
# =============================================================================
# Each entry: (DRR column name, factor label, sign correction for L-S)
# sign=1 means high signal -> high risk -> long; sign=-1 for reversal
SORT_SIGNALS = [
    ('var_95', 'DRF',  1),   # VaR -> default risk
    ('ilq',    'LRF',  1),   # illiquidity -> liquidity risk
    ('str',    'REV', -1),   # reversal: low past return -> higher expected return
]

# =============================================================================
# 1. Double sorts: RATING_NUM x signal (5x5, unconditional, VW)
# =============================================================================

factors = {}      # signal-level L-S (P5 - P1 on signal, avg across ratings)
factors_crf = {}  # credit component (RATING5 - RATING1, avg across signal bins)

for signal, label, sign in SORT_SIGNALS:
    print(f"\n-- Double sort: RATING_NUM x {signal} ({label}) --")

    cols = ['date', 'ID', 'VW', 'RATING_NUM', signal, 'ret']
    sub = data[cols].dropna(subset=[signal]).copy()
    print(f"  Obs: {len(sub):,}")

    strategy = pbl.DoubleSort(
        holding_period=1,
        sort_var='RATING_NUM',
        sort_var2=signal,
        num_portfolios=5,
        num_portfolios2=5,
        how='unconditional',
    )

    # dynamic_weights defaults to False here (formation-date VW = lagged market cap),
    # which matches the BBW methodology
    result = pbl.StrategyFormation(
        data=sub,
        strategy=strategy,
        rating=None,
    ).fit()

    _, vw_ptf = result.get_ptf()

    # Print column names for verification
    print(f"  Portfolio columns ({vw_ptf.shape[1]}): {sorted(vw_ptf.columns)[:4]} ...")

    sig_up = signal.upper()

    # --- L-S factor: P5 - P1 on signal, averaged across 5 rating groups ---
    ls_parts = []
    for r in range(1, 6):
        hi = f'RATING_NUM{r}_{sig_up}5'
        lo = f'RATING_NUM{r}_{sig_up}1'
        ls_parts.append(vw_ptf[hi] - vw_ptf[lo])

    factor_ts = pd.concat(ls_parts, axis=1).mean(axis=1) * sign
    factors[label] = factor_ts
    print(f"  {label} mean: {factor_ts.mean()*100:.3f}%/mo  "
          f"std: {factor_ts.std()*100:.3f}%/mo")

    # --- Credit component: RATING5 - RATING1 within each signal bin ---
    crf_parts = []
    for s in range(1, 6):
        nig = f'RATING_NUM5_{sig_up}{s}'
        ig = f'RATING_NUM1_{sig_up}{s}'
        crf_parts.append(vw_ptf[nig] - vw_ptf[ig])

    crf_ts = pd.concat(crf_parts, axis=1).mean(axis=1)
    factors_crf[label] = crf_ts
    print(f"  CRF component mean: {crf_ts.mean()*100:.3f}%/mo")

# =============================================================================
# 2. Construct DRF, LRF, CRF
# =============================================================================

DRF = factors['DRF'].to_frame('DRF')
LRF = factors['LRF'].to_frame('LRF')

# CRF = average of credit components across all 3 signals
CRF = pd.concat(factors_crf.values(), axis=1).mean(axis=1).to_frame('CRF')

# =============================================================================
# 3. MKTB: lagged-VW bond market excess return
# =============================================================================

print("\n-- Computing MKTB --")

mkt = data[['date', 'ID', 'VW', 'ret']].dropna(subset=['VW', 'ret']).copy()
mkt['VW_lag'] = mkt.groupby('ID')['VW'].shift(1)
mkt = mkt.dropna(subset=['VW_lag'])

# Normalize lagged VW to portfolio weights within each month
mkt['w'] = mkt.groupby('date')['VW_lag'].transform(lambda x: x / x.sum())
MKTB = (mkt['w'] * mkt['ret']).groupby(mkt['date']).sum().to_frame('MKTB')

print(f"  MKTB mean: {MKTB['MKTB'].mean()*100:.3f}%/mo  "
      f"std: {MKTB['MKTB'].std()*100:.3f}%/mo")

# =============================================================================
# 4. Combine BBW factors
# =============================================================================

bbw = MKTB.join(DRF).join(CRF).join(LRF)
bbw = bbw[['MKTB', 'DRF', 'CRF', 'LRF']]
bbw.index = pd.to_datetime(bbw.index)

print(f"\n-- Replicated BBW factors: {len(bbw)} months --")
print((bbw.describe() * 100).round(3).to_string())

# =============================================================================
# 5. Compare with reference BBW factors
# =============================================================================

print("\n-- Comparison with reference BBW factors --")

bbw_ref = pd.read_parquet(MKTB_PATH)
bbw_ref.index = pd.to_datetime(bbw_ref.index)

merged = bbw.join(bbw_ref, rsuffix='_ref').dropna()

print(f"  Overlapping months: {len(merged)}")
print(f"\n  {'Factor':<8} {'Corr':>8} {'Mean Diff':>12} {'Repl Mean%':>12} {'Ref Mean%':>12}")
print("  " + "-" * 54)
for f in ['MKTB', 'DRF', 'CRF', 'LRF']:
    r = merged[f].corr(merged[f'{f}_ref'])
    diff = (merged[f] - merged[f'{f}_ref']).mean()
    m1 = merged[f].mean() * 100
    m2 = merged[f'{f}_ref'].mean() * 100
    print(f"  {f:<8} {r:>8.4f} {diff:>12.6f} {m1:>12.3f} {m2:>12.3f}")
