#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fama-French Data Download Module
=================================

This module downloads and processes CRSP and Compustat data for Fama-French
factor construction. It implements the exact methodology from ff3_v2.ipynb
(Freda Song Drechsler's implementation) to ensure data consistency.

The module provides:
1. Compustat fundamental data download (book equity calculation)
2. CRSP monthly stock data download (returns, market equity, delistings)
3. CRSP-Compustat linking (CCM)
4. Data merging and alignment for July-June fiscal years
5. Save/load functions for processed data

Author: Giulio Rossetti
Date: 2025-11-21
"""

import pandas as pd
import numpy as np
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import os
from typing import Optional, Tuple


def download_compustat_data(
    conn: wrds.Connection,
    start_date: str = '01/01/1959',
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download Compustat fundamental data and compute book equity.

    This follows Fama-French methodology:
    - Book Equity = Shareholder Equity + Deferred Taxes - Preferred Stock
    - BE = SEQ + TXDITC - PS
    - Where PS = coalesce(PSTKRV, PSTKL, PSTK, 0)

    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection
    start_date : str, default '01/01/1959'
        Start date for data download (MM/DD/YYYY format)
    end_date : str, optional
        End date for data download. If None, uses current date.

    Returns
    -------
    pd.DataFrame
        Compustat data with columns: gvkey, datadate, year, be, count
        - gvkey: Firm identifier
        - datadate: Fiscal year end date
        - year: Fiscal year
        - be: Book equity (millions)
        - count: Number of years firm has been in Compustat
    """
    print("Downloading Compustat data...")

    # Build date filter
    date_filter = f"and datadate >= '{start_date}'"
    if end_date is not None:
        date_filter += f" and datadate <= '{end_date}'"

    # Download fundamental annual data
    comp = conn.raw_sql(f"""
        select gvkey, datadate, at, pstkl, txditc,
        pstkrv, seq, pstk
        from comp.funda
        where indfmt='INDL'
        and datafmt='STD'
        and popsrc='D'
        and consol='C'
        {date_filter}
    """, date_cols=['datadate'])

    comp['year'] = comp['datadate'].dt.year

    # Create preferred stock variable
    # Preferred Stock = coalesce(Redemption Value, Liquidation Value, Par Value, 0)
    comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
    comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
    comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])
    comp['txditc'] = comp['txditc'].fillna(0)

    # Create book equity
    # BE = Shareholder Equity + Deferred Taxes - Preferred Stock
    comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
    comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)

    # Number of years in Compustat (for minimum history filter)
    comp = comp.sort_values(by=['gvkey', 'datadate'])
    comp['count'] = comp.groupby(['gvkey']).cumcount()

    # Keep only relevant columns
    comp = comp[['gvkey', 'datadate', 'year', 'be', 'count']]

    print(f"  Downloaded {len(comp)} Compustat observations")
    print(f"  Date range: {comp['datadate'].min()} to {comp['datadate'].max()}")
    print(f"  Unique firms: {comp['gvkey'].nunique()}")

    return comp


def download_crsp_data(
    conn: wrds.Connection,
    start_date: str = '01/01/1959',
    end_date: str = '12/31/2017'
) -> pd.DataFrame:
    """
    Download CRSP monthly stock data with delistings.

    Downloads:
    - Monthly stock file (msf): returns, prices, shares outstanding
    - Security names (msenames): share codes, exchange codes
    - Delisting returns (msedelist): delisting dates and returns

    Computes:
    - Market equity (ME = |price| × shares outstanding, in thousands)
    - Adjusted returns (incorporating delisting returns)

    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection
    start_date : str, default '01/01/1959'
        Start date for data download
    end_date : str, default '12/31/2017'
        End date for data download

    Returns
    -------
    pd.DataFrame
        CRSP monthly data with columns:
        - permno, permco: Security and company identifiers
        - date, jdate: Calendar date and end-of-month date
        - shrcd, exchcd: Share code and exchange code
        - ret, retx: Total and ex-dividend returns
        - retadj: Return adjusted for delisting
        - me: Market equity (thousands of dollars)
    """
    print("\nDownloading CRSP monthly data...")

    # Download monthly stock file with security characteristics
    crsp_m = conn.raw_sql(f"""
        select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
        a.ret, a.retx, a.shrout, a.prc
        from crsp.msf as a
        left join crsp.msenames as b
        on a.permno=b.permno
        and b.namedt<=a.date
        and a.date<=b.nameendt
        where a.date between '{start_date}' and '{end_date}'
        and b.exchcd between 1 and 3
    """, date_cols=['date'])

    # Change variable format to int
    crsp_m[['permco', 'permno', 'shrcd', 'exchcd']] = \
        crsp_m[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

    # Line up date to be end of month
    crsp_m['jdate'] = crsp_m['date'] + MonthEnd(0)

    print(f"  Downloaded {len(crsp_m)} CRSP monthly observations")

    # Download delisting returns
    print("  Downloading delisting returns...")
    dlret = conn.raw_sql("""
        select permno, dlret, dlstdt
        from crsp.msedelist
    """, date_cols=['dlstdt'])

    dlret.permno = dlret.permno.astype(int)
    dlret['jdate'] = dlret['dlstdt'] + MonthEnd(0)

    # Merge with delisting returns
    crsp = pd.merge(crsp_m, dlret, how='left', on=['permno', 'jdate'])
    crsp['dlret'] = crsp['dlret'].fillna(0)
    crsp['ret'] = crsp['ret'].fillna(0)

    # Adjusted return factors in delisting returns
    crsp['retadj'] = (1 + crsp['ret']) * (1 + crsp['dlret']) - 1

    # Calculate market equity (in thousands)
    crsp['me'] = crsp['prc'].abs() * crsp['shrout']
    crsp = crsp.drop(['dlret', 'dlstdt', 'prc', 'shrout'], axis=1)
    crsp = crsp.sort_values(by=['jdate', 'permco', 'me'])

    print(f"  Total observations after delisting merge: {len(crsp)}")
    print(f"  Date range: {crsp['jdate'].min()} to {crsp['jdate'].max()}")
    print(f"  Unique securities: {crsp['permno'].nunique()}")

    return crsp


def aggregate_market_cap(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate market cap at the PERMCO level.

    For firms with multiple share classes (different PERMNOs but same PERMCO):
    - Keep the PERMNO with largest market cap
    - Assign the sum of all share classes' ME to that PERMNO

    This ensures we don't double-count firms in portfolio formation.

    Parameters
    ----------
    crsp : pd.DataFrame
        CRSP data with permno, permco, jdate, me

    Returns
    -------
    pd.DataFrame
        CRSP data with one row per permco-date, using representative permno
    """
    print("\nAggregating market cap at PERMCO level...")

    # Sum of ME across different PERMNOs belonging to same PERMCO
    crsp_summe = crsp.groupby(['jdate', 'permco'])['me'].sum().reset_index()

    # Largest market cap within a PERMCO/date
    crsp_maxme = crsp.groupby(['jdate', 'permco'])['me'].max().reset_index()

    # Join by jdate/maxme to find the PERMNO with largest ME
    crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['jdate', 'permco', 'me'])

    # Drop ME column and replace with sum ME
    crsp1 = crsp1.drop(['me'], axis=1)

    # Join with sum of ME to get correct market cap
    crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['jdate', 'permco'])

    # Sort by permno and date, drop duplicates
    crsp2 = crsp2.sort_values(by=['permno', 'jdate']).drop_duplicates()

    print(f"  Aggregated to {len(crsp2)} unique permco-date observations")

    return crsp2


def compute_lagged_me_and_weights(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged market equity and portfolio weights following FF methodology.

    Fama-French uses July-June fiscal years:
    - June ME (end of June) is used for portfolio formation in July
    - Portfolios are held from July t to June t+1
    - Weights are based on beginning-of-month ME, adjusted by cumulative returns

    This function:
    1. Creates July-June fiscal year indicators (ffdate, ffyear)
    2. Computes lagged ME (lme) = prior month's ME
    3. Computes baseline ME (mebase) = ME at start of fiscal year (July)
    4. Computes portfolio weights (wt):
       - July: use June ME (lagged ME)
       - Aug-June: use July ME × cumulative return since July

    Parameters
    ----------
    crsp : pd.DataFrame
        CRSP data with jdate, permno, me, retx

    Returns
    -------
    pd.DataFrame
        CRSP data with additional columns:
        - year, month: Calendar year and month
        - ffdate, ffyear, ffmonth: Fiscal year date, year, month (July-June)
        - cumretx: Cumulative return factor since start of fiscal year
        - lme: Lagged market equity
        - wt: Portfolio weight (ME for value-weighting)
    """
    print("\nComputing lagged ME and weights...")

    # Extract December ME for book-to-market calculation
    crsp['year'] = crsp['jdate'].dt.year
    crsp['month'] = crsp['jdate'].dt.month

    # July to June fiscal years
    # Shift dates back 6 months to create fiscal year alignment
    crsp['ffdate'] = crsp['jdate'] + MonthEnd(-6)
    crsp['ffyear'] = crsp['ffdate'].dt.year
    crsp['ffmonth'] = crsp['ffdate'].dt.month

    # Compute cumulative return within fiscal year
    crsp['1+retx'] = 1 + crsp['retx']
    crsp = crsp.sort_values(by=['permno', 'date'])

    # Cumulative return by stock-fiscal year
    crsp['cumretx'] = crsp.groupby(['permno', 'ffyear'])['1+retx'].cumprod()

    # Lag cumulative return
    crsp['lcumretx'] = crsp.groupby(['permno'])['cumretx'].shift(1)

    # Lag market cap
    crsp['lme'] = crsp.groupby(['permno'])['me'].shift(1)

    # For first observation of each permno, use me/(1+retx)
    crsp['count'] = crsp.groupby(['permno']).cumcount()
    crsp['lme'] = np.where(crsp['count'] == 0, crsp['me'] / crsp['1+retx'], crsp['lme'])

    # Baseline ME (as of start of fiscal year = July, which is ffmonth==1)
    mebase = crsp[crsp['ffmonth'] == 1][['permno', 'ffyear', 'lme']].rename(
        columns={'lme': 'mebase'}
    )

    # Merge back
    crsp = pd.merge(crsp, mebase, how='left', on=['permno', 'ffyear'])

    # Weight calculation for PyBondLab annual rebalancing:
    # PyBondLab reads VW at formation time (June) and uses it to weight returns
    # For proper FF methodology with June formation:
    # - June (month==6): VW = current ME (for portfolio sorting and formation)
    # - July-May (month!=6): VW = June ME (held constant, or adjusted by returns if dynamic)
    #
    # Note: The 'wt' column will be used as VW in PyBondLab
    # Since PyBondLab with annual rebalancing only looks at June for formation,
    # we want VW in June to equal June ME
    crsp['wt'] = np.where(
        crsp['month'] == 6,
        crsp['me'],                       # June: use current ME for formation
        crsp['lme']                       # Other months: use lagged ME (for consistency)
    )

    print(f"  Computed weights for {len(crsp)} observations")
    print(f"  June formation: VW = current ME")
    print(f"  Other months: VW = lagged ME")

    return crsp


def download_ccm_link(conn: wrds.Connection) -> pd.DataFrame:
    """
    Download CRSP-Compustat Merged (CCM) linking table.

    The CCM link table matches CRSP PERMNOs to Compustat GVKEYs.
    We use only reliable links:
    - Link type starts with 'L' (LINK, LU, LC, etc.)
    - Link primary code is 'C' (primary) or 'P' (primary, joiner)

    Parameters
    ----------
    conn : wrds.Connection
        Active WRDS database connection

    Returns
    -------
    pd.DataFrame
        CCM link table with columns: gvkey, permno, linktype, linkprim, linkdt, linkenddt
    """
    print("\nDownloading CCM link table...")

    ccm = conn.raw_sql("""
        select gvkey, lpermno as permno, linktype, linkprim,
        linkdt, linkenddt
        from crsp.ccmxpf_linktable
        where substr(linktype,1,1)='L'
        and (linkprim ='C' or linkprim='P')
    """, date_cols=['linkdt', 'linkenddt'])

    # If linkenddt is missing, set to today's date
    ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

    print(f"  Downloaded {len(ccm)} CCM links")
    print(f"  Unique GVKEYs: {ccm['gvkey'].nunique()}")
    print(f"  Unique PERMNOs: {ccm['permno'].nunique()}")

    return ccm


def merge_compustat_crsp(
    comp: pd.DataFrame,
    crsp: pd.DataFrame,
    ccm: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Compustat and CRSP data using CCM link table.

    Steps:
    1. Prepare Compustat: align book equity to June following fiscal year end
    2. Link Compustat to CRSP using CCM (time-varying links)
    3. Merge with CRSP monthly data
    4. Compute book-to-market ratio

    Parameters
    ----------
    comp : pd.DataFrame
        Compustat data (from download_compustat_data)
    crsp : pd.DataFrame
        CRSP data (from aggregate_market_cap and compute_lagged_me_and_weights)
    ccm : pd.DataFrame
        CCM link table (from download_ccm_link)

    Returns
    -------
    pd.DataFrame
        Merged CRSP-Compustat data with book-to-market ratios
    """
    print("\nMerging Compustat and CRSP...")

    # Merge Compustat with CCM
    ccm1 = pd.merge(comp[['gvkey', 'datadate', 'be', 'count']], ccm, how='left', on=['gvkey'])

    # Align accounting data to June following fiscal year end
    ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
    ccm1['jdate'] = ccm1['yearend'] + MonthEnd(6)

    # Set link date bounds (only use links that are active at jdate)
    ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]
    ccm2 = ccm2[['gvkey', 'permno', 'datadate', 'yearend', 'jdate', 'be', 'count']]

    print(f"  CCM-linked observations: {len(ccm2)}")

    # Get December market equity for book-to-market calculation
    decme = crsp[crsp['month'] == 12][['permno', 'year', 'me']].copy()
    decme['year'] = decme['year'] + 1  # Align to following year
    decme = decme.rename(columns={'me': 'dec_me'})

    # Get June characteristics
    crsp_jun = crsp[crsp['month'] == 6].copy()

    # Merge June CRSP with December ME
    crsp_jun = pd.merge(crsp_jun, decme, how='inner', on=['permno', 'year'])
    crsp_jun = crsp_jun[[
        'permno', 'date', 'jdate', 'shrcd', 'exchcd', 'retadj',
        'me', 'wt', 'cumretx', 'mebase', 'lme', 'dec_me'
    ]].sort_values(by=['permno', 'jdate']).drop_duplicates()

    # Merge CRSP June with Compustat
    ccm_jun = pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])

    # Compute book-to-market ratio
    # BE is in millions, dec_me is in thousands → multiply BE by 1000
    ccm_jun['beme'] = ccm_jun['be'] * 1000 / ccm_jun['dec_me']

    print(f"  Merged CRSP-Compustat observations: {len(ccm_jun)}")

    # Handle duplicate permno-jdate from multiple fiscal year ends
    # Keep most recent fiscal year (highest datadate)
    # This happens when firms change their fiscal year end
    dup_count = ccm_jun.duplicated(['permno', 'jdate']).sum()
    if dup_count > 0:
        print(f"  Found {dup_count} duplicate permno-jdate pairs (fiscal year changes)")
        print(f"  Keeping most recent fiscal year (highest datadate)...")
        ccm_jun = ccm_jun.sort_values(['permno', 'jdate', 'datadate'])
        ccm_jun = ccm_jun.groupby(['permno', 'jdate'], as_index=False).last()
        print(f"  After deduplication: {len(ccm_jun)} observations")

    print(f"  Date range: {ccm_jun['jdate'].min()} to {ccm_jun['jdate'].max()}")

    return ccm_jun


def create_portfolio_assignment(
    ccm_jun: pd.DataFrame,
    crsp: pd.DataFrame
) -> pd.DataFrame:
    """
    Create portfolio assignments and merge with monthly CRSP data.

    Portfolio formation:
    1. Compute NYSE breakpoints (size and book-to-market)
    2. Assign all stocks to size and BtM portfolios
    3. Merge assignments with monthly returns (July t to June t+1)

    Filters:
    - NYSE breakpoints: EXCHCD==1, SHRCD in [10,11], positive BtM, positive ME, count>=1
    - Portfolio universe: SHRCD in [10,11], positive weight, valid portfolio assignment

    Parameters
    ----------
    ccm_jun : pd.DataFrame
        June characteristics with CRSP-Compustat merge
    crsp : pd.DataFrame
        Monthly CRSP data

    Returns
    -------
    pd.DataFrame
        Monthly CRSP data with portfolio assignments and characteristics
    """
    print("\nCreating portfolio assignments...")

    # Select NYSE stocks for breakpoint computation
    nyse = ccm_jun[
        (ccm_jun['exchcd'] == 1) &
        (ccm_jun['beme'] > 0) &
        (ccm_jun['me'] > 0) &
        (ccm_jun['count'] >= 1) &
        ((ccm_jun['shrcd'] == 10) | (ccm_jun['shrcd'] == 11))
    ]

    print(f"  NYSE breakpoint universe: {len(nyse)} observations")

    # Size breakpoint: median ME
    nyse_sz = nyse.groupby(['jdate'])['me'].median().to_frame().reset_index()
    nyse_sz = nyse_sz.rename(columns={'me': 'sizemedn'})

    # Book-to-market breakpoints: 30th and 70th percentiles
    nyse_bm = nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
    nyse_bm = nyse_bm[['jdate', '30%', '70%']].rename(columns={'30%': 'bm30', '70%': 'bm70'})

    # Merge breakpoints
    nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])

    # Join breakpoints back to all stocks
    ccm1_jun = pd.merge(ccm_jun, nyse_breaks, how='left', on=['jdate'])

    # Assign size portfolio
    def sz_bucket(row):
        if pd.isna(row['me']):
            return ''
        elif row['me'] <= row['sizemedn']:
            return 'S'
        else:
            return 'B'

    # Assign book-to-market portfolio
    def bm_bucket(row):
        if 0 <= row['beme'] <= row['bm30']:
            return 'L'
        elif row['beme'] <= row['bm70']:
            return 'M'
        elif row['beme'] > row['bm70']:
            return 'H'
        else:
            return ''

    # Apply portfolio assignments
    ccm1_jun['szport'] = np.where(
        (ccm1_jun['beme'] > 0) & (ccm1_jun['me'] > 0) & (ccm1_jun['count'] >= 1),
        ccm1_jun.apply(sz_bucket, axis=1),
        ''
    )

    ccm1_jun['bmport'] = np.where(
        (ccm1_jun['beme'] > 0) & (ccm1_jun['me'] > 0) & (ccm1_jun['count'] >= 1),
        ccm1_jun.apply(bm_bucket, axis=1),
        ''
    )

    # Create indicator variables
    ccm1_jun['posbm'] = np.where(
        (ccm1_jun['beme'] > 0) & (ccm1_jun['me'] > 0) & (ccm1_jun['count'] >= 1),
        1, 0
    )
    ccm1_jun['nonmissport'] = np.where(ccm1_jun['bmport'] != '', 1, 0)

    # Store portfolio assignment as of June
    june = ccm1_jun[['permno', 'date', 'jdate', 'bmport', 'szport', 'posbm', 'nonmissport']].copy()
    june['ffyear'] = june['jdate'].dt.year

    # Merge back with monthly records
    crsp3 = crsp[['date', 'permno', 'shrcd', 'exchcd', 'retadj', 'me', 'wt', 'cumretx', 'ffyear', 'jdate']]
    ccm3 = pd.merge(
        crsp3,
        june[['permno', 'ffyear', 'szport', 'bmport', 'posbm', 'nonmissport']],
        how='left',
        on=['permno', 'ffyear']
    )

    # Filter to portfolio universe
    ccm4 = ccm3[
        (ccm3['wt'] > 0) &
        (ccm3['posbm'] == 1) &
        (ccm3['nonmissport'] == 1) &
        ((ccm3['shrcd'] == 10) | (ccm3['shrcd'] == 11))
    ]

    print(f"  Final portfolio universe: {len(ccm4)} stock-month observations")
    print(f"  Unique stocks: {ccm4['permno'].nunique()}")

    return ccm4


def prepare_data_for_pybondlab(
    ccm4: pd.DataFrame,
    ccm_jun: pd.DataFrame,
    start_year: int = 1970
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data in format suitable for both PyBondLab and ff3_v2.ipynb.

    Creates two datasets:
    1. Monthly returns with characteristics (for PyBondLab)
    2. June characteristics (for verification and analysis)

    Parameters
    ----------
    ccm4 : pd.DataFrame
        Monthly CRSP data with portfolio assignments
    ccm_jun : pd.DataFrame
        June characteristics with full CRSP-Compustat merge
    start_year : int, default 1970
        Filter data to this year and later

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - monthly_data: Monthly returns with characteristics
        - june_chars: June characteristics for portfolio formation
    """
    print(f"\nPreparing data for PyBondLab (filtering to {start_year}+)...")

    # Create date_m (YYYYMM format) and date_jun for PyBondLab compatibility
    ccm4['date_m'] = ccm4['jdate'].dt.year * 100 + ccm4['jdate'].dt.month
    ccm4['date_jun'] = ccm4['ffyear'] * 100 + 1  # July = month 1 of fiscal year

    # Filter to start year
    ccm4_filtered = ccm4[ccm4['jdate'].dt.year >= start_year].copy()

    # Merge with June characteristics to get BtM and count
    # ccm4_filtered already has: permno, date, shrcd, exchcd, retadj, me, wt, cumretx, ffyear, jdate, szport, bmport, posbm, nonmissport
    june_merge = ccm_jun[[
        'permno', 'jdate', 'beme', 'count'
    ]].copy()
    june_merge['ffyear'] = june_merge['jdate'].dt.year

    monthly_data = pd.merge(
        ccm4_filtered,
        june_merge[['permno', 'ffyear', 'beme', 'count']],
        how='left',
        on=['permno', 'ffyear'],
        suffixes=('', '_jun')
    )

    # Rename columns to match PyBondLab expectations
    monthly_data = monthly_data.rename(columns={
        'permno': 'PERMNO',
        'shrcd': 'SHRCD',
        'exchcd': 'EXCHCD',
        'me': 'ME',
        'beme': 'BtM',
        'wt': 'VW'
    })

    # Add dummy rating (not used for FF factors)
    monthly_data['RATING_NUM'] = 11

    # Select final columns
    monthly_data = monthly_data[[
        'PERMNO', 'date', 'date_m', 'date_jun', 'jdate', 'ffyear',
        'retadj', 'ME', 'VW', 'BtM', 'SHRCD', 'EXCHCD', 'RATING_NUM',
        'szport', 'bmport', 'count'
    ]].copy()

    # Prepare June characteristics
    june_chars = ccm_jun.copy()
    june_chars['ffyear'] = june_chars['jdate'].dt.year
    june_chars = june_chars[june_chars['jdate'].dt.year >= start_year]

    print(f"  Monthly data: {len(monthly_data)} observations")
    print(f"  Date range: {monthly_data['jdate'].min()} to {monthly_data['jdate'].max()}")
    print(f"  June characteristics: {len(june_chars)} observations")

    return monthly_data, june_chars


def download_and_process_all(
    wrds_username: str,
    start_date: str = '01/01/1959',
    end_date: str = '12/31/2017',
    start_year: int = 1970,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data download and processing pipeline.

    This is the main function that orchestrates the entire data download
    and processing workflow. It:
    1. Connects to WRDS
    2. Downloads Compustat, CRSP, and CCM data
    3. Processes and merges data
    4. Creates portfolio assignments
    5. Prepares data for analysis
    6. Optionally saves to files

    Parameters
    ----------
    wrds_username : str
        WRDS username for database connection
    start_date : str, default '01/01/1959'
        Start date for data download
    end_date : str, default '12/31/2017'
        End date for data download
    start_year : int, default 1970
        Filter final output to this year and later
    save_path : str, optional
        Directory path to save output files. If None, data is not saved.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - monthly_data: Monthly returns with characteristics (ready for PyBondLab)
        - june_chars: June characteristics (for verification)

    Examples
    --------
    >>> monthly, june = download_and_process_all(
    ...     wrds_username='myusername',
    ...     save_path='./data'
    ... )
    >>> monthly.to_csv('./data/crsp_monthly.csv', index=False)
    """
    print("="*80)
    print("FAMA-FRENCH DATA DOWNLOAD AND PROCESSING")
    print("="*80)
    print()

    # Connect to WRDS
    print("Connecting to WRDS...")
    conn = wrds.Connection(wrds_username=wrds_username)
    print("  Connected successfully")
    print()

    try:
        # Download Compustat
        comp = download_compustat_data(conn, start_date, end_date)

        # Download CRSP
        crsp = download_crsp_data(conn, start_date, end_date)

        # Aggregate market cap
        crsp = aggregate_market_cap(crsp)

        # Compute lagged ME and weights
        crsp = compute_lagged_me_and_weights(crsp)

        # Download CCM link
        ccm = download_ccm_link(conn)

        # Merge Compustat and CRSP
        ccm_jun = merge_compustat_crsp(comp, crsp, ccm)

        # Create portfolio assignments
        ccm4 = create_portfolio_assignment(ccm_jun, crsp)

        # Prepare final datasets
        monthly_data, june_chars = prepare_data_for_pybondlab(ccm4, ccm_jun, start_year)

        # Save if requested
        if save_path is not None:
            save_processed_data(monthly_data, june_chars, save_path)

        print()
        print("="*80)
        print("DATA DOWNLOAD AND PROCESSING COMPLETE")
        print("="*80)
        print()

        return monthly_data, june_chars

    finally:
        conn.close()
        print("WRDS connection closed")


def save_processed_data(
    monthly_data: pd.DataFrame,
    june_chars: pd.DataFrame,
    save_path: str
):
    """
    Save processed data to CSV and Parquet files.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Monthly returns with characteristics
    june_chars : pd.DataFrame
        June characteristics
    save_path : str
        Directory path to save files
    """
    print(f"\nSaving processed data to {save_path}...")

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save to CSV
    monthly_csv = os.path.join(save_path, 'crsp_monthly_ff.csv')
    june_csv = os.path.join(save_path, 'june_characteristics_ff.csv')

    monthly_data.to_csv(monthly_csv, index=False)
    june_chars.to_csv(june_csv, index=False)

    print(f"  Saved: {monthly_csv}")
    print(f"  Saved: {june_csv}")

    # Save to Parquet (more efficient for large files)
    monthly_parquet = os.path.join(save_path, 'crsp_monthly_ff.parquet')
    june_parquet = os.path.join(save_path, 'june_characteristics_ff.parquet')

    monthly_data.to_parquet(monthly_parquet, index=False)
    june_chars.to_parquet(june_parquet, index=False)

    print(f"  Saved: {monthly_parquet}")
    print(f"  Saved: {june_parquet}")


def load_processed_data(
    data_path: str,
    file_format: str = 'parquet'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved processed data.

    Parameters
    ----------
    data_path : str
        Directory path containing saved files
    file_format : str, default 'parquet'
        File format to load ('csv' or 'parquet')

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - monthly_data: Monthly returns with characteristics
        - june_chars: June characteristics
    """
    print(f"\nLoading processed data from {data_path}...")

    if file_format == 'parquet':
        monthly_file = os.path.join(data_path, 'crsp_monthly_ff.parquet')
        june_file = os.path.join(data_path, 'june_characteristics_ff.parquet')
        monthly_data = pd.read_parquet(monthly_file)
        june_chars = pd.read_parquet(june_file)
    elif file_format == 'csv':
        monthly_file = os.path.join(data_path, 'crsp_monthly_ff.csv')
        june_file = os.path.join(data_path, 'june_characteristics_ff.csv')
        monthly_data = pd.read_csv(monthly_file)
        june_chars = pd.read_csv(june_file)
        # Convert date columns
        monthly_data['date'] = pd.to_datetime(monthly_data['date'])
        monthly_data['jdate'] = pd.to_datetime(monthly_data['jdate'])
        june_chars['date'] = pd.to_datetime(june_chars['date'])
        june_chars['jdate'] = pd.to_datetime(june_chars['jdate'])
        june_chars['datadate'] = pd.to_datetime(june_chars['datadate'])
        june_chars['yearend'] = pd.to_datetime(june_chars['yearend'])
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    print(f"  Loaded monthly data: {len(monthly_data)} observations")
    print(f"  Loaded June characteristics: {len(june_chars)} observations")

    return monthly_data, june_chars


if __name__ == "__main__":
    """
    Example usage when run as a script.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python download_data_for_ff.py <wrds_username> [save_path]")
        print("\nExample:")
        print("  python download_data_for_ff.py myusername ./data")
        sys.exit(1)

    wrds_username = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else './data'

    # Run full pipeline
    monthly_data, june_chars = download_and_process_all(
        wrds_username=wrds_username,
        save_path=save_path
    )

    print("\nData download complete!")
    print(f"Files saved to: {save_path}")
    print("\nTo load the data later:")
    print(f"  from download_data_for_ff import load_processed_data")
    print(f"  monthly, june = load_processed_data('{save_path}')")
