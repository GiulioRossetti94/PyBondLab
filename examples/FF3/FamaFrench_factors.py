#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fama-French 3 Factor Replication with PyBondLab
=================================================

Replicates SMB and HML using PyBondLab's DoubleSort strategy.

DATA REQUIREMENT:
-----------------
Run the data download script first to get WRDS data:

    python download_data.py <wrds_username>

This creates:
    data/monthly_panel_raw.parquet  (CRSP-Compustat monthly panel)
    data/ff_official.parquet        (official Ken French factors)

METHODOLOGY:
------------
- 2x3 independent sorts on Size (ME) and Book-to-Market (BtM)
- NYSE breakpoints for both dimensions
- Annual rebalancing in June (returns start July)
- 11-month holding period
- dynamic_weights=True (buy-and-hold VW, as in FF original)

EXPECTED CORRELATIONS (1970-2017):
-----------------------------------
  SMB: r ~ 0.993  vs official Ken French
  HML: r ~ 0.982  vs official Ken French

Can be run in Spyder (cell-by-cell) or from terminal:
    python FamaFrench_factors.py

Author: Giulio Rossetti
"""

# %% 1. Imports

import pandas as pd
import numpy as np
import PyBondLab as pbl
import os
import matplotlib
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from scipy import stats

try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')

# %% 2. Configuration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Date filter for comparison with official factors
DATE_START = '1970-01-01'
DATE_END = '2017-12-31'

# %% 3. Load data

print("=" * 60)
print("FAMA-FRENCH 3 FACTOR REPLICATION WITH PyBondLab")
print("=" * 60)

print("\nLoading data...")
panel = pd.read_parquet(os.path.join(DATA_DIR, 'monthly_panel_raw.parquet'))
ff_official = pd.read_parquet(os.path.join(DATA_DIR, 'ff_official.parquet'))

print(f"  Monthly panel: {len(panel):,} obs, {panel['permno'].nunique():,} securities")
print(f"  Date range: {panel['jdate'].min()} to {panel['jdate'].max()}")
print(f"  BtM available: {panel['BtM'].notna().sum():,} / {len(panel):,}")

# %% 4. Prepare data for PyBondLab

# Rename columns to match PyBondLab expectations
merged = panel.rename(columns={
    'permno': 'PERMNO',
    'shrcd': 'SHRCD',
    'exchcd': 'EXCHCD',
    'me': 'ME',
    'wt': 'VW',
    'count_comp': 'count',
})

# PyBondLab needs a 'date' column (datetime, end-of-month)
merged['date'] = merged['jdate']

# Filter to common shares on major exchanges (same universe as Freda)
merged = merged[
    (merged['SHRCD'].isin([10, 11])) &
    (merged['EXCHCD'].isin([1, 2, 3]))
].copy()

# Filter: positive VW weights
merged = merged[merged['VW'] > 0].copy()

# Drop duplicate PERMNO-date rows (from PERMCO aggregation)
merged = merged.drop_duplicates(subset=['PERMNO', 'date'])

# RATING_NUM is required by PyBondLab but not used when rating=None
merged['RATING_NUM'] = 11

# Match FF eligibility: exclude negative BtM and stocks with count < 1
n_neg_btm = (merged['BtM'] <= 0).sum()
n_low_count = (merged['count'] < 1).sum()
merged.loc[merged['BtM'] <= 0, 'BtM'] = np.nan
merged.loc[merged['count'] < 1, 'BtM'] = np.nan
print(f"  Set BtM=NaN: {n_neg_btm} rows with BtM<=0, {n_low_count} rows with count<1")

# Shift BtM back by one ffyear so it's available at June formation date
# BtM is assigned by ffyear (which changes in July), so ffyear-1 makes it
# available in June of that year -- matching FF's June sort timing.
june_chars = merged[merged['date'].dt.month == 7][['PERMNO', 'ffyear', 'BtM', 'count']].copy()
june_chars['ffyear'] = june_chars['ffyear'] - 1
merged = merged.drop(columns=['BtM', 'count']).merge(
    june_chars, on=['PERMNO', 'ffyear'], how='left')
print(f"  Shifted BtM by -1 ffyear for June formation")

# Filter to dates where BtM data exists
btm_start = merged.loc[merged['BtM'].notna(), 'date'].min()
btm_end = merged.loc[merged['BtM'].notna(), 'date'].max()
merged = merged[(merged['date'] >= btm_start) & (merged['date'] <= btm_end)].copy()

print(f"\nPrepared for PyBondLab: {len(merged):,} obs, {merged['PERMNO'].nunique():,} securities")

# %% 5. Define NYSE breakpoint filter


def nyse_filter(df):
    """
    NYSE breakpoint universe following Fama-French methodology.
    - EXCHCD == 1 (NYSE only)
    - SHRCD in [10, 11] (ordinary common shares)
    - BtM > 0 (positive book equity)
    - ME > 0 (positive market equity)
    - count >= 1 (at least 1 year in Compustat)
    """
    return (
        (df['EXCHCD'] == 1) &
        (df['BtM'] > 0) &
        (df['ME'] > 0) &
        (df['SHRCD'].isin([10, 11])) &
        (df['count'] >= 1)
    )


# %% 6. Define DoubleSort strategy

strategy = pbl.DoubleSort(
    sort_var='ME',                         # Primary sort: Size
    sort_var2='BtM',                       # Secondary sort: Book-to-Market
    num_portfolios=2,                      # 2 size portfolios (Small/Big)
    num_portfolios2=3,                     # 3 BtM portfolios (Low/Med/High)
    breakpoints=[50],                      # Median split for size
    breakpoints2=[30, 70],                 # 30th/70th percentiles for BtM
    how='unconditional',                   # Independent sorts
    rebalance_frequency='annual',          # Annual rebalancing
    rebalance_month=6,                     # Formation in June (returns start July)
    breakpoint_universe_func=nyse_filter,  # NYSE breakpoints for size
    breakpoint_universe_func2=nyse_filter, # NYSE breakpoints for BtM
    verbose=True
)

# %% 7. Form portfolios

print("\nForming portfolios...")

results = pbl.StrategyFormation(
    data=merged,
    strategy=strategy,
    rating=None,
    dynamic_weights=True,                  # Buy-and-hold VW (as in FF original)
    chars=['ME', 'BtM', 'EXCHCD', 'SHRCD', 'count'],
    turnover=True,
    verbose=True
).fit(IDvar='PERMNO', RETvar='retadj')

print("Portfolio formation complete!")

# %% 8. Extract portfolio returns

ew_portfolios, vw_portfolios = results.get_ptf()
ew_to, vw_to = results.get_turnover()

print(f"\nPortfolio columns: {sorted(vw_portfolios.columns.tolist())}")

# %% 9. Construct SMB and HML factors

# Identify portfolio columns
small_cols = [c for c in vw_portfolios.columns if c.startswith('ME1')]
big_cols = [c for c in vw_portfolios.columns if c.startswith('ME2')]

# SMB = (1/3)(SL + SM + SH) - (1/3)(BL + BM + BH)
smb_vw = vw_portfolios[small_cols].mean(axis=1) - vw_portfolios[big_cols].mean(axis=1)

# HML = (1/2)(SH + BH) - (1/2)(SL + BL)
small_high = [c for c in vw_portfolios.columns if c.startswith('ME1_BTM3')]
small_low = [c for c in vw_portfolios.columns if c.startswith('ME1_BTM1')]
big_high = [c for c in vw_portfolios.columns if c.startswith('ME2_BTM3')]
big_low = [c for c in vw_portfolios.columns if c.startswith('ME2_BTM1')]

hml_vw = ((vw_portfolios[small_high].mean(axis=1) +
            vw_portfolios[big_high].mean(axis=1)) / 2 -
           (vw_portfolios[small_low].mean(axis=1) +
            vw_portfolios[big_low].mean(axis=1)) / 2)

print(f"\nSMB mean: {smb_vw.mean():.4f},  std: {smb_vw.std():.4f}")
print(f"HML mean: {hml_vw.mean():.4f},  std: {hml_vw.std():.4f}")

# %% 10. Compare with official Fama-French factors

# Prepare PBL factors
smb_df = pd.DataFrame({'SMB_PBL': smb_vw})
smb_df.index = smb_df.index + MonthEnd(0)
smb_df = smb_df.reset_index(names='date')

hml_df = pd.DataFrame({'HML_PBL': hml_vw})
hml_df.index = hml_df.index + MonthEnd(0)
hml_df = hml_df.reset_index(names='date')

# Merge with official
ff_official['date'] = pd.to_datetime(ff_official['date'])
ff_official['smb'] = ff_official['smb'].astype(float)
ff_official['hml'] = ff_official['hml'].astype(float)

comp_smb = smb_df.merge(ff_official[['date', 'smb']], on='date', how='inner')
comp_hml = hml_df.merge(ff_official[['date', 'hml']], on='date', how='inner')

# Filter date range
comp_smb = comp_smb[(comp_smb['date'] >= DATE_START) &
                     (comp_smb['date'] <= DATE_END)].set_index('date')
comp_hml = comp_hml[(comp_hml['date'] >= DATE_START) &
                     (comp_hml['date'] <= DATE_END)].set_index('date')

r_smb, p_smb = stats.pearsonr(comp_smb['SMB_PBL'], comp_smb['smb'])
r_hml, p_hml = stats.pearsonr(comp_hml['HML_PBL'], comp_hml['hml'])

print(f"\n{'=' * 60}")
print(f"Correlations with official FF factors ({DATE_START[:4]}-{DATE_END[:4]})")
print(f"{'=' * 60}")
print(f"  SMB: r = {r_smb:.4f}  (p = {p_smb:.2e})")
print(f"  HML: r = {r_hml:.4f}  (p = {p_hml:.2e})")
print(f"  SMB mean diff: {(comp_smb['SMB_PBL'] - comp_smb['smb']).mean():.6f}")
print(f"  HML mean diff: {(comp_hml['HML_PBL'] - comp_hml['hml']).mean():.6f}")

# %% 11. Plot

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('FF3 Factor Replication: PyBondLab vs Official Ken French', fontsize=16)

# SMB cumulative returns
cum_smb = (1 + comp_smb[['smb', 'SMB_PBL']]).cumprod()
axes[0, 0].plot(cum_smb.index, cum_smb['smb'], 'r-', lw=2, label='Official')
axes[0, 0].plot(cum_smb.index, cum_smb['SMB_PBL'], 'b--', lw=1.5, alpha=0.8,
                label=f'PBL (r={r_smb:.4f})')
axes[0, 0].set_title('SMB Cumulative Returns', fontsize=13)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# HML cumulative returns
cum_hml = (1 + comp_hml[['hml', 'HML_PBL']]).cumprod()
axes[0, 1].plot(cum_hml.index, cum_hml['hml'], 'r-', lw=2, label='Official')
axes[0, 1].plot(cum_hml.index, cum_hml['HML_PBL'], 'b--', lw=1.5, alpha=0.8,
                label=f'PBL (r={r_hml:.4f})')
axes[0, 1].set_title('HML Cumulative Returns', fontsize=13)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# SMB scatter
axes[1, 0].scatter(comp_smb['smb'], comp_smb['SMB_PBL'],
                   alpha=0.3, s=10, color='blue', label=f'r={r_smb:.4f}')
lims_smb = [comp_smb['smb'].min(), comp_smb['smb'].max()]
axes[1, 0].plot(lims_smb, lims_smb, 'r--', lw=1)
axes[1, 0].set_xlabel('Official SMB')
axes[1, 0].set_ylabel('PBL SMB')
axes[1, 0].set_title('SMB: PBL vs Official (monthly)')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# HML scatter
axes[1, 1].scatter(comp_hml['hml'], comp_hml['HML_PBL'],
                   alpha=0.3, s=10, color='blue', label=f'r={r_hml:.4f}')
lims_hml = [comp_hml['hml'].min(), comp_hml['hml'].max()]
axes[1, 1].plot(lims_hml, lims_hml, 'r--', lw=1)
axes[1, 1].set_xlabel('Official HML')
axes[1, 1].set_ylabel('PBL HML')
axes[1, 1].set_title('HML: PBL vs Official (monthly)')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'ff3_replication.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {os.path.join(SCRIPT_DIR, 'ff3_replication.png')}")
plt.show()

print("\nDone.")
