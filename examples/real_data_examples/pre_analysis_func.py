import pandas as pd
import numpy as np

import PyBondLab as pbl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from PyBondLab.AnomalyAssayer import AnomalyResults

# =============================================================================
# Load Data File
# =============================================================================
# TODO: Update BASE_PATH to point to your local data directory
BASE_PATH = Path("path/to/your/data").expanduser()
VERSION = "v1"
TIMESTAMP = "29_04_2025"

# Data files
PANEL_FILE = f'drr_monthly_panel_{VERSION}_{TIMESTAMP}.parquet'
PANEL_PATH = BASE_PATH / PANEL_FILE

FACTOR_FILE = 'bbw4_month_end_data_return.parquet'
FACTOR_PATH = BASE_PATH / FACTOR_FILE

tbl1 = pd.read_parquet(PANEL_PATH)
tbl1['RATING_NUM'] = tbl1['sp_rating']
print(f"Loaded panel data with shape: {tbl1.shape}")

pre_stats = pbl.PreAnalysisStats(data=tbl1,
                                    variables=['bond_ret_me','bond_ret_mb','PRC_END','YTM'],
                                    date_col='date',
                                    id_col='cusip_id')

stats = pre_stats.compute()

# Get summary statistics for 'bond_ret_me'. 
# This return a DataFrame with descriptive statistics for the specified variable.
# Each row corresponds to the time-series mean of different statistic (mean, std, min, max, etc.) 
sum_stats = stats.summary('bond_ret_me')
print(stats.summary('bond_ret_me'))


# CS stats
cs_stats = stats.get_cs_stats('bond_ret_me')
cs_stats_price = stats.get_cs_stats('PRC_END')
cs_stats_ytm = stats.get_cs_stats('YTM')
print(cs_stats)
# =============================================================================
# subsetting by rating
# =============================================================================

pre_stats_IG = pbl.PreAnalysisStats(data=tbl1,
                                    variables=['bond_ret_me','bond_ret_mb','PRC_END','YTM'],
                                    date_col='date',
                                    id_col='cusip_id',
                                    rating='IG')


pre_stats_NIG = pbl.PreAnalysisStats(data=tbl1,
                                    variables=['bond_ret_me','bond_ret_mb','PRC_END','YTM'],
                                    date_col='date',
                                    id_col='cusip_id',
                                    rating='NIG')

stats_IG = pre_stats_IG.compute()
stats_NIG = pre_stats_NIG.compute()
sum_stats_IG = stats_IG.summary('bond_ret_me')

print(sum_stats)
print(sum_stats_IG)
print(stats_IG.compare(stats, variable='YTM'))

stats_IG.plot(variable='YTM',stats = ['mean','std','p75','p95'], title='YTM stats for IG bonds')
stats_NIG.plot(variable='YTM',stats = ['mean','std','p75','p95'], title='YTM stats for NIG bonds')
print(stats_NIG.compare(stats_IG, variable='YTM'))

# =============================================================================
# subsetting by rating and chars
# =============================================================================
# Characteristic filter
stats_subset = pbl.PreAnalysisStats(
    data=tbl1,
    variables='CREDIT_SPREAD',
    subset_filter={'bond_maturity': (5, 15), 'mod_dur': (2, 8)},
    date_col='date',
    id_col='cusip_id',
)

# Combined (IG + characteristic filter)
stats_combined = pbl.PreAnalysisStats(
    data=tbl1,
    variables='CREDIT_SPREAD',
    rating='IG',
    subset_filter={'bond_maturity': (5, 15)},
    date_col='date',
    id_col='cusip_id',
)

stats_subset_res = stats_subset.compute()
stats_combined_res = stats_combined.compute()

print(stats_subset_res.summary('CREDIT_SPREAD'))
print(stats_combined_res.summary('CREDIT_SPREAD'))

fig1 = stats_subset_res.plot(variable='CREDIT_SPREAD', stats=['mean', 'std'], title='Credit Spread (5-15 yr maturity)')
fig2 = stats_combined_res.plot(variable='CREDIT_SPREAD', stats=['mean', 'std'], title='Credit Spread IG Bonds (5-15 yr maturity)')