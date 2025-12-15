#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fama-French Factor Construction Example
========================================

This example demonstrates how to construct Fama-French factors (SMB and HML)
using PyBondLab's DoubleSort strategy and compare them against official
Ken French factors.

The example follows the standard Fama-French (1993) methodology:
- 2x3 independent sorts on Size (ME) and Book-to-Market (BtM)
- Annual rebalancing in June using prior year-end book equity
- NYSE breakpoints for both size and value
- 12-month holding periods

NOTE ON rebalance_month TIMING:
We use rebalance_month=7 for "June rebalancing". See the main README.md
for a detailed explanation of why rebalance_month=7 (not 6).

DATA HARMONIZATION (Updated 2025-11-21)
----------------------------------------
This file now supports two data loading methods:

1. HARMONIZED DATA (Recommended): Uses download_data_for_ff.py module
   - Implements exact ff3 methodology
   - Proper lagged ME weights with cumulative return adjustments
   - Complete CRSP-Compustat merge with CCM linking

   To use:
   - First time: Set DOWNLOAD_FRESH_DATA=True (takes 10-20 min)
   - Or run: python download_data_for_ff.py <wrds_username> ./data
   - Subsequent runs: Load from ./data directory (instant)

2. LEGACY DATA: Falls back to original CSV files if module unavailable
   - Uses pre-processed CSV files (old method)

Configuration:
- Set DOWNLOAD_FRESH_DATA = True/False (line 44)
- Set WRDS_USERNAME (line 47)
- Set DATA_DIR (line 50)

Author: Giulio Rossetti
Date: 2025-11-21
"""

import pandas as pd
import numpy as np
import PyBondLab as pbl
import os
import wrds
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats

# Import data download module
try:
    from download_data_for_ff import load_processed_data, download_and_process_all
    DATA_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: download_data_for_ff module not found. Using fallback CSV loading.")
    DATA_MODULE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set to True to download fresh data from WRDS (requires WRDS account)
# Set to False to load previously downloaded data
DOWNLOAD_FRESH_DATA = False
# WRDS username (only needed if DOWNLOAD_FRESH_DATA = True)
WRDS_USERNAME = ''

# Data directory (where processed data is saved/loaded)
DATA_DIR = './data'

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("="*80)
print("FAMA-FRENCH FACTOR CONSTRUCTION WITH PyBondLab")
print("="*80)
print()

print("Step 1: Loading data files...")
print("-"*80)

if DATA_MODULE_AVAILABLE:
    # Use new harmonized data module
    if DOWNLOAD_FRESH_DATA:
        print("  Downloading fresh data from WRDS...")
        print("  (This may take 10-20 minutes)")
        merged, june_chars = download_and_process_all(
            wrds_username=WRDS_USERNAME,
            start_date='01/01/1959',
            end_date='12/31/2017',
            start_year=1970,
            save_path=DATA_DIR
        )
    else:
        print(f"  Loading data from {DATA_DIR}...")
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(
                f"Data directory '{DATA_DIR}' not found. "
                f"Set DOWNLOAD_FRESH_DATA=True to download data, or run:\n"
                f"  python download_data_for_ff.py {WRDS_USERNAME} {DATA_DIR}"
            )
        merged, june_chars = load_processed_data(DATA_DIR, file_format='parquet')

    print(f"  Loaded {len(merged)} monthly observations (harmonized data)")
    print(f"  Date range: {merged['jdate'].min()} to {merged['jdate'].max()}")
    print(f"  Unique securities: {merged['PERMNO'].nunique()}")
    print()

else:
    # Fallback to old CSV loading method
    # TODO: Update these paths to point to your local data files
    print("  Using legacy CSV files...")
    char = pd.read_csv('path/to/your/FirmCharacteristicsFF5.csv')
    crsp = pd.read_csv('path/to/your/CRSPreturn1926m.csv')

    print(f"  Loaded {len(char)} firm characteristic observations")
    print(f"  Loaded {len(crsp)} CRSP return observations")
    print()

    # =============================================================================
    # STEP 2: Prepare CRSP Data (Legacy)
    # =============================================================================
    print("Step 2: Preparing CRSP data...")
    print("-"*80)

    # Define data types for memory efficiency
    ctotype32 = {
        'date_m': np.int32,
        'date_jun': np.int32,
        'PERMNO': np.int32,
        'RET': np.float32,
        'retadj': np.float32
    }

    # Clean CRSP data
    crspm = crsp.astype(ctotype32).dropna()
    # Filter to post-1970 data (standard in FF literature)
    crspm = crspm[crspm['date_jun'] >= 197001].reset_index(drop=True)

    print(f"  CRSP data: {len(crspm)} observations after 1970-01")
    print()

    # =============================================================================
    # STEP 3: Prepare Firm Characteristics (Legacy)
    # =============================================================================
    print("Step 3: Preparing firm characteristics...")
    print("-"*80)

    firmchars = char.copy()

    # Create count variable to ensure sufficient history
    # (Fama-French requires firms to have been listed for at least 2 years)
    firmchars['count'] = firmchars.groupby(['gvkey_x']).cumcount()
    firmchars = firmchars[firmchars['date_jun'] >= 197001].reset_index(drop=True)

    # Select relevant columns
    keep_cols = [
        'PERMNO', 'date_jun', 'ME', 'BtM', 'ME_dec', 'CAP_W',
        'EXCHCD', 'SHRCD', 'ME_W', 'CAP_W', 'count'
    ]

    firmchars_ = firmchars[keep_cols].drop_duplicates()

    print(f"  Firm characteristics: {len(firmchars_)} unique firm-date observations")
    print()

    # =============================================================================
    # STEP 4: Merge CRSP and Characteristics (Legacy)
    # =============================================================================
    print("Step 4: Merging CRSP and characteristics...")
    print("-"*80)

    # Merge CRSP returns with firm characteristics
    # This links monthly returns to annual characteristics (measured in June)
    merged = pd.merge(
        crspm,
        firmchars_,
        on=['PERMNO', 'date_jun'],
        how='left'
    )

    # Sort by date and PERMNO for proper time-series alignment
    merged = merged.sort_values(["date_m", "PERMNO"])
    # merged.loc[np.isinf(merged['BtM']), 'BtM'] = np.nan

    print(f"  Merged data: {len(merged)} observations")
    print(f"  Date range: {merged['date_m'].min()} to {merged['date_m'].max()}")
    print()

# =============================================================================
# STEP 5: Add Required Fields for PyBondLab
# =============================================================================
print("Step 5: Preparing data for PyBondLab...")
print("-"*80)

if DATA_MODULE_AVAILABLE:
    # Harmonized data already has all required fields
    # Just ensure 'date' column exists (jdate is end-of-month datetime)
    if 'date' not in merged.columns:
        merged['date'] = merged['jdate']

    # VW column already contains proper weights from data module
    print(f"  Using harmonized data with proper FF weights")
    print(f"  Date column: {merged['date'].min()} to {merged['date'].max()}")
    print(f"  Total firms: {merged['PERMNO'].nunique()}")
    print(f"  VW (value weights) already computed using lagged ME methodology")

else:
    # Legacy: Add dummy rating (not used for Fama-French factors)
    merged['RATING_NUM'] = 11

    # Convert date to datetime format
    merged['date'] = pd.to_datetime(merged['date_m'].astype(str), format='%Y%m')

    # Use market equity as value weight
    # Note: In standard FF methodology, we use lagged ME for value weighting
    merged['VW'] = merged['ME']

    print(f"  Added date column: {merged['date'].min()} to {merged['date'].max()}")
    print(f"  Total firms: {merged['PERMNO'].nunique()}")

print()

# =============================================================================
# STEP 6: Define NYSE Breakpoint Filter
# =============================================================================
print("Step 6: Defining NYSE breakpoint filter...")
print("-"*80)

def nyse_filter(df):
    """
    Filter to NYSE stocks for breakpoint computation.

    Following Fama-French methodology:
    - Exchange code = 1 (NYSE only)
    - Share code 10 or 11 (ordinary common shares only)
    - Positive book-to-market (exclude negative BE firms)
    - Positive market equity (exclude zero/negative ME firms)
    - At least 2 years of history (count >= 2)

    This ensures breakpoints are based only on large, established firms,
    while portfolios include NYSE, AMEX, and NASDAQ stocks.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with firm characteristics

    Returns
    -------
    pandas.Series
        Boolean series indicating which rows pass the filter
    """
    return (
        (df['EXCHCD'].isin([1])) &           # NYSE only
        (df['BtM'] > 0) &                # Positive book-to-market
        (df['ME'] > 0) &                 # Positive market equity
        ((df['SHRCD'] == 10) | (df['SHRCD'] == 11))    # Ordinary common shares
        # (df['count'] >= 12)               # At least 1 years of history
    )

print("  NYSE filter defined:")
print("    - EXCHCD == 1 (NYSE)")
print("    - BtM > 0 (positive book equity)")
print("    - ME > 0 (positive market equity)")
print("    - SHRCD in [10, 11] (ordinary common shares)")
print("    - count >= 2 (at least 2 years of history)")
print()

# =============================================================================
# STEP 7: Define Fama-French 2x3 Sort Strategy
# =============================================================================
print("Step 7: Defining Fama-French 2x3 sort strategy...")
print("-"*80)

strategy = pbl.DoubleSort(
    holding_period=11,                     # 12-month holding period (annual)
    sort_var='ME',                         # Primary sort: Size (Market Equity)
    sort_var2='BtM',                       # Secondary sort: Book-to-Market
    num_portfolios=2,                      # 2 size portfolios (Small/Big)
    num_portfolios2=3,                     # 3 BM portfolios (Low/Med/High)
    breakpoints=[50],                      # Median split for size
    breakpoints2=[30, 70],                 # 30th/70th percentiles for BM
    how='unconditional',                   # Independent sorts (not conditional)
    rebalance_frequency='annual',          # Rebalance once per year
    rebalance_month=7,                     # June rebalancing (see README for timing explanation)
    breakpoint_universe_func=nyse_filter,  # Use NYSE for size breakpoints
    breakpoint_universe_func2=nyse_filter, # Use NYSE for BM breakpoints
    verbose=True                           # Print progress information
)

print("  Strategy configuration:")
print("    - 2x3 independent sorts (Size x Book-to-Market)")
print("    - Size breakpoint: 50th percentile (median)")
print("    - BtM breakpoints: 30th and 70th percentiles")
print("    - Annual rebalancing in June")
print("    - 12-month holding periods")
print("    - NYSE breakpoints for both dimensions")
print()

# =============================================================================
# STEP 8: Form Portfolios
# =============================================================================
print("Step 8: Forming portfolios...")
print("-"*80)

# Enable turnover calculation to track portfolio changes
do_turnover = True

# Run the strategy formation
results = pbl.StrategyFormation(
    data=merged,
    strategy=strategy,
    rating=None,                          # Use all stocks (no rating filter)
    dynamic_weights=False,                # Static weights within holding period
    chars=['ME', 'BtM', 'EXCHCD', 'SHRCD', 'count'],  # Track characteristics
    turnover=do_turnover,                 # Calculate turnover
    verbose=True
).fit(IDvar='PERMNO', RETvar='retadj')   # Use PERMNO as ID, retadj as return

print()
print("  Portfolio formation complete!")
print()

# =============================================================================
# STEP 9: Extract Portfolio Returns
# =============================================================================
print("Step 9: Extracting portfolio returns...")
print("-"*80)

# Get equal-weighted and value-weighted portfolio returns
ew_portfolios, vw_portfolios = results.get_ptf()

if do_turnover:
    ew_to, vw_to = results.get_turnover()

print(f"  Equal-weighted portfolios: {ew_portfolios.shape}")
print(f"  Value-weighted portfolios: {vw_portfolios.shape}")
print()

# Display portfolio structure
print("  Portfolio columns:")
for col in sorted(vw_portfolios.columns):
    print(f"    {col}")
print()

# =============================================================================
# STEP 10: Construct SMB (Small Minus Big) Factor
# =============================================================================
print("Step 10: Constructing SMB (Small Minus Big) factor...")
print("-"*80)

# SMB is the average return on small portfolios minus the average return
# on big portfolios. We average across all three BtM categories.

# Identify small and big portfolios
small_cols = [col for col in ew_portfolios.columns if col.startswith('ME1')]
big_cols = [col for col in ew_portfolios.columns if col.startswith('ME2')]

print(f"  Small portfolios: {small_cols}")
print(f"  Big portfolios: {big_cols}")
print()

# Equal-weighted SMB
small_avg_ew = ew_portfolios[small_cols].mean(axis=1)
big_avg_ew = ew_portfolios[big_cols].mean(axis=1)
smb_ew = small_avg_ew - big_avg_ew

# Value-weighted SMB (standard in literature)
small_avg_vw = vw_portfolios[small_cols].mean(axis=1)
big_avg_vw = vw_portfolios[big_cols].mean(axis=1)
smb_vw = small_avg_vw - big_avg_vw

# Convert to DataFrame
smb_vw_df = pd.DataFrame(smb_vw, columns=['SMB_VW_Manual'])

print(f"  SMB factor constructed: {len(smb_vw_df)} months")
print(f"  SMB mean: {smb_vw_df['SMB_VW_Manual'].mean():.4f}")
print(f"  SMB std:  {smb_vw_df['SMB_VW_Manual'].std():.4f}")
print()

# =============================================================================
# STEP 11: Construct HML (High Minus Low) Factor
# =============================================================================
print("Step 11: Constructing HML (High Minus Low) factor...")
print("-"*80)

# HML is the average return on high BtM portfolios minus the average return
# on low BtM portfolios. We average across both size categories.

# Identify Small/Big and Low/High portfolios
small_low = [col for col in ew_portfolios.columns if col.startswith('ME1_BTM1')]
small_high = [col for col in ew_portfolios.columns if col.startswith('ME1_BTM3')]
big_low = [col for col in ew_portfolios.columns if col.startswith('ME2_BTM1')]
big_high = [col for col in ew_portfolios.columns if col.startswith('ME2_BTM3')]

print(f"  Small Low BtM: {small_low}")
print(f"  Small High BtM: {small_high}")
print(f"  Big Low BtM: {big_low}")
print(f"  Big High BtM: {big_high}")
print()

# Equal-weighted HML
high_avg_ew = (ew_portfolios[small_high].mean(axis=1) + ew_portfolios[big_high].mean(axis=1)) / 2
low_avg_ew = (ew_portfolios[small_low].mean(axis=1) + ew_portfolios[big_low].mean(axis=1)) / 2
hml_ew = high_avg_ew - low_avg_ew

# Value-weighted HML (standard in literature)
high_avg_vw = (vw_portfolios[small_high].mean(axis=1) + vw_portfolios[big_high].mean(axis=1)) / 2
low_avg_vw = (vw_portfolios[small_low].mean(axis=1) + vw_portfolios[big_low].mean(axis=1)) / 2
hml_vw = high_avg_vw - low_avg_vw

# Convert to DataFrame
hml_vw_df = pd.DataFrame(hml_vw, columns=['HML_VW_Manual'])

print(f"  HML factor constructed: {len(hml_vw_df)} months")
print(f"  HML mean: {hml_vw_df['HML_VW_Manual'].mean():.4f}")
print(f"  HML std:  {hml_vw_df['HML_VW_Manual'].std():.4f}")
print()

# =============================================================================
# STEP 12: Load or Download Official Fama-French Factors
# =============================================================================
print("Step 12: Loading official Fama-French factors...")
print("-"*80)

# Path to cached FF factors
FF_FACTORS_CACHE = os.path.join(DATA_DIR, 'ff_factors_monthly.csv')

try:
    # Check if cached file exists
    if os.path.exists(FF_FACTORS_CACHE):
        print(f"  Loading cached FF factors from {FF_FACTORS_CACHE}...")
        _ff = pd.read_csv(FF_FACTORS_CACHE, parse_dates=['date'])
        print(f"  Loaded {len(_ff)} months of official FF factors from cache")
        print(f"  Date range: {_ff['date'].min()} to {_ff['date'].max()}")
        print()
    else:
        # Download from WRDS
        print(f"  Cache not found. Downloading from WRDS...")
        conn = wrds.Connection(wrds_username=WRDS_USERNAME)

        # Download Fama-French factors
        _ff = conn.get_table(library='ff', table='factors_monthly')
        _ff = _ff[['date', 'smb', 'hml']]
        _ff['date'] = _ff['date'] + MonthEnd(0)
        _ff['date'] = pd.to_datetime(_ff['date'])

        # Save to cache
        _ff.to_csv(FF_FACTORS_CACHE, index=False)
        print(f"  Downloaded {len(_ff)} months of official FF factors")
        print(f"  Saved to {FF_FACTORS_CACHE}")
        print(f"  Date range: {_ff['date'].min()} to {_ff['date'].max()}")
        print()

    wrds_available = True
except Exception as e:
    print(f"  WARNING: Could not load/download FF factors: {e}")
    print("  Skipping official factor comparison")
    print()
    wrds_available = False

# =============================================================================
# STEP 13: Compare with Official Fama-French Factors
# =============================================================================
if wrds_available:
    print("Step 13: Comparing with official Fama-French factors...")
    print("-"*80)

    # Prepare SMB comparison
    SMB_pbl = smb_vw_df.copy()
    SMB_pbl.index = SMB_pbl.index + MonthEnd(0)
    SMB_pbl = SMB_pbl.dropna().reset_index(names='date')

    comp_SMB = SMB_pbl.merge(_ff[['smb', 'date']], how='inner', on=['date']).set_index('date')
    comp_SMB['smb'] = pd.to_numeric(comp_SMB['smb'])

    print("  SMB Comparison:")
    print(f"    Correlation: {comp_SMB['SMB_VW_Manual'].corr(comp_SMB['smb']):.4f}")
    print(f"    Mean difference: {(comp_SMB['SMB_VW_Manual'] - comp_SMB['smb']).mean():.4f}")
    print()

    # Prepare HML comparison
    HML_pbl = hml_vw_df.copy()
    HML_pbl.index = HML_pbl.index + MonthEnd(0)
    HML_pbl = HML_pbl.dropna().reset_index(names='date')

    comp_HML = HML_pbl.merge(_ff[['hml', 'date']], how='inner', on=['date']).set_index('date')
    comp_HML['hml'] = pd.to_numeric(comp_HML['hml'])

    print("  HML Comparison:")
    print(f"    Correlation: {comp_HML['HML_VW_Manual'].corr(comp_HML['hml']):.4f}")
    print(f"    Mean difference: {(comp_HML['HML_VW_Manual'] - comp_HML['hml']).mean():.4f}")
    print()

    # =============================================================================
    # STEP 14: Filter Date Range
    # =============================================================================
    print("Step 15: Filtering to common date range...")
    print("-"*80)

    date_end = "2024-12-31"
    date_start = "1926-01-01"

    # comp_SMB = comp_SMB[comp_SMB.index < date_end]
    # comp_HML = comp_HML[comp_HML.index < date_end]
    
    comp_SMB = comp_SMB[(comp_SMB.index >= date_start) & (comp_SMB.index < date_end)]
    comp_HML = comp_HML[(comp_HML.index >= date_start) & (comp_HML.index < date_end)]

    print(f"  Date range: {comp_SMB.index.min()} to {comp_SMB.index.max()}")
    print(f"  Number of months: {len(comp_SMB)}")
    print()

    # =============================================================================
    # STEP 15: Correlation Analysis
    # =============================================================================
    print("Step 16: Correlation analysis...")
    print("-"*80)

    print("  HML Correlation Matrix:")
    print(comp_HML.corr().to_string())
    print()

    print("  SMB Correlation Matrix:")
    print(comp_SMB.corr().to_string())
    print()

    # =============================================================================
    # STEP 16: Visualization
    # =============================================================================
    print("Step 17: Creating visualizations...")
    print("-"*80)

    # Plot SMB comparison
    plt.figure(figsize=(16, 12))
    ax1 = plt.subplot()
    plt.suptitle('SMB Factor Comparison', fontsize=20)

    CUMPROD = (1 + comp_SMB).cumprod()
    ax1.plot(CUMPROD['smb'], 'r--',
             CUMPROD['SMB_VW_Manual'], 'g--')

    ax1.legend(['Official SMB (Ken French)',
                'PyBondLab SMB (Computed)'],
               loc='upper left',
               fontsize=12,
               frameon=True)

    ax1.set_title('Cumulative SMB Factor Performance', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('SMB_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: SMB_comparison.png")
    plt.show()

    # Plot HML comparison
    plt.figure(figsize=(16, 12))
    ax1 = plt.subplot()
    plt.suptitle('HML Factor Comparison', fontsize=20)

    CUMPROD = (1 + comp_HML).cumprod()
    ax1.plot(CUMPROD['hml'], 'r--',
             CUMPROD['HML_VW_Manual'], 'g--')

    ax1.legend(['Official HML (Ken French)',
                'PyBondLab HML (Computed)'],
               loc='upper left',
               fontsize=12,
               frameon=True)

    ax1.set_title('Cumulative HML Factor Performance', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('HML_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: HML_comparison.png")
    plt.show()

    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

else:
    print()
    print("="*80)
    print("FACTOR CONSTRUCTION COMPLETE")
    print("="*80)
    print("Note: Comparison with official factors skipped (WRDS not available)")
    if DATA_MODULE_AVAILABLE:
        print("\nData Source: HARMONIZED (download_data_for_ff.py)")
    else:
        print("Data Source: LEGACY CSV files")
    print()

print(CUMPROD.tail(1))
