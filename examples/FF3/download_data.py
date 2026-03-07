#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download WRDS Data for Fama-French Factor Replication
======================================================

Downloads CRSP and Compustat data from WRDS, processes it following the
Fama-French methodology (based on Qingyi (Freda) Song Drechsler's
implementation), and saves the data for use with FamaFrench_factors.py.

Usage:
    python download_data.py <wrds_username>

Output files (saved to ./data/):
    monthly_panel_raw.parquet   CRSP monthly panel with BtM from Compustat
    ff_official.parquet         Official Ken French factors from WRDS

Requires: wrds, pandas, numpy
"""

# %% 1. Imports

import pandas as pd
import numpy as np
import wrds
import os
import sys
from pandas.tseries.offsets import MonthEnd, YearEnd

# %% 2. Configuration

if len(sys.argv) >= 2:
    WRDS_USERNAME = sys.argv[1]
else:
    WRDS_USERNAME = ''  # <-- enter your WRDS username here

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# %% 3. Connect to WRDS

print("=" * 60)
print("DOWNLOADING WRDS DATA FOR FF REPLICATION")
print("=" * 60)

conn = wrds.Connection(wrds_username=WRDS_USERNAME)

# %% 4. Download Compustat data

print("\nDownloading Compustat data...")

comp = conn.raw_sql("""
    select gvkey, datadate, at, pstkl, txditc,
           pstkrv, seq, pstk
    from comp.funda
    where indfmt='INDL'
      and datafmt='STD'
      and popsrc='D'
      and consol='C'
      and datadate >= '01/01/1959'
""", date_cols=['datadate'])

comp['year'] = comp['datadate'].dt.year

# Preferred stock = coalesce(pstkrv, pstkl, pstk, 0)
comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])
comp['txditc'] = comp['txditc'].fillna(0)

# Book equity = SEQ + TXDITC - PS
comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)

# Number of years in Compustat
comp = comp.sort_values(by=['gvkey', 'datadate'])
comp['count'] = comp.groupby(['gvkey']).cumcount()

comp = comp[['gvkey', 'datadate', 'year', 'be', 'count']]

print(f"  Compustat: {len(comp):,} obs, {comp['gvkey'].nunique():,} firms")

# %% 5. Download CRSP monthly data

print("Downloading CRSP monthly data...")

crsp_m = conn.raw_sql("""
    select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
           a.ret, a.retx, a.shrout, a.prc
    from crsp.msf as a
    left join crsp.msenames as b
      on a.permno=b.permno
      and b.namedt<=a.date
      and a.date<=b.nameendt
    where a.date between '01/01/1959' and '12/31/2017'
      and b.exchcd between 1 and 3
""", date_cols=['date'])

crsp_m[['permco', 'permno', 'shrcd', 'exchcd']] = \
    crsp_m[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

crsp_m['jdate'] = crsp_m['date'] + MonthEnd(0)

print(f"  CRSP: {len(crsp_m):,} obs, {crsp_m['permno'].nunique():,} securities")

# %% 6. Download delisting returns and compute adjusted returns

print("Downloading delisting returns...")

dlret = conn.raw_sql("""
    select permno, dlret, dlstdt
    from crsp.msedelist
""", date_cols=['dlstdt'])

dlret.permno = dlret.permno.astype(int)
dlret['jdate'] = dlret['dlstdt'] + MonthEnd(0)

crsp = pd.merge(crsp_m, dlret, how='left', on=['permno', 'jdate'])
crsp['dlret'] = crsp['dlret'].fillna(0)
crsp['ret'] = crsp['ret'].fillna(0)

# Adjusted return factors in delisting returns
crsp['retadj'] = (1 + crsp['ret']) * (1 + crsp['dlret']) - 1

# Market equity (in thousands)
crsp['me'] = crsp['prc'].abs() * crsp['shrout']
crsp = crsp.drop(['dlret', 'dlstdt', 'prc', 'shrout'], axis=1)
crsp = crsp.sort_values(by=['jdate', 'permco', 'me'])

# %% 7. Aggregate market cap at PERMCO level

print("Aggregating market cap at PERMCO level...")

crsp_summe = crsp.groupby(['jdate', 'permco'])['me'].sum().reset_index()
crsp_maxme = crsp.groupby(['jdate', 'permco'])['me'].max().reset_index()

crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['jdate', 'permco', 'me'])
crsp1 = crsp1.drop(['me'], axis=1)

crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['jdate', 'permco'])
crsp2 = crsp2.sort_values(by=['permno', 'jdate']).drop_duplicates()

# %% 8. Compute lagged ME and weights (July-June fiscal years)

print("Computing lagged ME and weights...")

crsp2['year'] = crsp2['jdate'].dt.year
crsp2['month'] = crsp2['jdate'].dt.month

# July-June fiscal year alignment
crsp2['ffdate'] = crsp2['jdate'] + MonthEnd(-6)
crsp2['ffyear'] = crsp2['ffdate'].dt.year
crsp2['ffmonth'] = crsp2['ffdate'].dt.month
crsp2['1+retx'] = 1 + crsp2['retx']
crsp2 = crsp2.sort_values(by=['permno', 'date'])

# Cumulative return by stock-fiscal year
crsp2['cumretx'] = crsp2.groupby(['permno', 'ffyear'])['1+retx'].cumprod()
crsp2['lcumretx'] = crsp2.groupby(['permno'])['cumretx'].shift(1)

# Lag market cap
crsp2['lme'] = crsp2.groupby(['permno'])['me'].shift(1)
crsp2['count'] = crsp2.groupby(['permno']).cumcount()
crsp2['lme'] = np.where(crsp2['count'] == 0,
                         crsp2['me'] / crsp2['1+retx'],
                         crsp2['lme'])

# Baseline ME (as of July = ffmonth 1)
mebase = crsp2[crsp2['ffmonth'] == 1][['permno', 'ffyear', 'lme']].rename(
    columns={'lme': 'mebase'})

crsp3 = pd.merge(crsp2, mebase, how='left', on=['permno', 'ffyear'])
crsp3['wt'] = np.where(crsp3['ffmonth'] == 1,
                        crsp3['lme'],
                        crsp3['mebase'] * crsp3['lcumretx'])

# %% 9. Prepare June characteristics and merge with December ME

print("Preparing June characteristics...")

decme = crsp2[crsp2['month'] == 12][['permno', 'year', 'me']].copy()
decme['year'] = decme['year'] + 1
decme = decme.rename(columns={'me': 'dec_me'})

crsp3_jun = crsp3[crsp3['month'] == 6]
crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno', 'year'])
crsp_jun = crsp_jun[['permno', 'date', 'jdate', 'shrcd', 'exchcd',
                      'retadj', 'me', 'wt', 'cumretx', 'mebase',
                      'lme', 'dec_me']]
crsp_jun = crsp_jun.sort_values(by=['permno', 'jdate']).drop_duplicates()

# %% 10. Download CCM link table and merge Compustat-CRSP

print("Downloading CCM link table...")

ccm = conn.raw_sql("""
    select gvkey, lpermno as permno, linktype, linkprim,
           linkdt, linkenddt
    from crsp.ccmxpf_linktable
    where substr(linktype,1,1)='L'
      and (linkprim ='C' or linkprim='P')
""", date_cols=['linkdt', 'linkenddt'])

ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

# Merge Compustat with CCM
ccm1 = pd.merge(comp[['gvkey', 'datadate', 'be', 'count']],
                 ccm, how='left', on=['gvkey'])
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['yearend'] + MonthEnd(6)

# Keep only active links
ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) &
             (ccm1['jdate'] <= ccm1['linkenddt'])]
ccm2 = ccm2[['gvkey', 'permno', 'datadate', 'yearend', 'jdate', 'be', 'count']]

# Link Compustat and CRSP
ccm_jun = pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['beme'] = ccm_jun['be'] * 1000 / ccm_jun['dec_me']

print(f"  Merged Compustat-CRSP: {len(ccm_jun):,} obs")

# %% 11. Prepare monthly panel for PyBondLab

print("Preparing monthly panel...")

crsp3 = crsp3[['date', 'permno', 'shrcd', 'exchcd', 'retadj',
                'me', 'wt', 'cumretx', 'ffyear', 'jdate']]

# Merge June BtM and Compustat count into monthly panel
_june_chars = ccm_jun[['permno', 'jdate', 'beme', 'count', 'shrcd', 'exchcd']].copy()
_june_chars['ffyear'] = _june_chars['jdate'].dt.year
_june_chars = _june_chars.rename(columns={'beme': 'BtM'})

pbl_data = pd.merge(
    crsp3,
    _june_chars[['permno', 'ffyear', 'BtM', 'count']].rename(columns={'count': 'count_comp'}),
    how='left', on=['permno', 'ffyear'])

# %% 12. Download official FF factors

print("Downloading official Ken French factors...")

_ff = conn.get_table(library='ff', table='factors_monthly')
_ff = _ff[['date', 'smb', 'hml']]
_ff['date'] = pd.to_datetime(_ff['date']) + MonthEnd(0)
_ff['smb'] = _ff['smb'].astype(float)
_ff['hml'] = _ff['hml'].astype(float)

# %% 13. Save

print(f"\nSaving to {DATA_DIR}...")

pbl_data.to_parquet(os.path.join(DATA_DIR, 'monthly_panel_raw.parquet'), index=False)
_ff.to_parquet(os.path.join(DATA_DIR, 'ff_official.parquet'), index=False)

print(f"  monthly_panel_raw.parquet  ({len(pbl_data):,} rows)")
print(f"  ff_official.parquet        ({len(_ff):,} rows)")

# %% 14. Close connection

conn.close()

print("\nDone. Now run FamaFrench_factors.py to replicate FF3.")
