import numpy as np
import pandas as pd
import PyBondLab as pbl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from PyBondLab.AnomalyAssayer import AnomalyResults
from PyBondLab.visualization import set_latex
set_latex(use_tex=False, font_family="palatino")
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

F = pd.read_parquet(FACTOR_PATH)
f = F['MKTB']

# Create overall panel index
tbl1 = tbl1.reset_index(drop=True)
tbl1["index"] = np.arange(1, len(tbl1) + 1)
# =============================================================================
# Format the data
# =============================================================================
tbl1['date'] = pd.to_datetime(tbl1['date'])

# Your panel must be sorted by it's bond identifier and date #
tbl1 = tbl1.sort_values(['cusip_id','date'])

# WRDS data "starts" officially on "2002-08-31"
tbl1 = tbl1[tbl1['date'] >= "2002-08-31"]
tbl1['VW'] = tbl1['MARKET_CAP']

# =============================================================================
# Create random allocation for unpriced factor vol
# =============================================================================
tbl1['rand_ass'] = tbl1.groupby('date')['cusip_id'].transform(
    lambda x: np.random.uniform(size = len(x)))
# The portfolio returns are computed with the overlapping methodology
# of Jegadeesh and Titman (1993) when holding_period > 1
holding_period = 1            # holding period returns
n_portf        = 10            # number of portfolios, 5 is quintiles, 10 deciles
skip = 0
sort_var0 = 'rand_ass'

data = tbl1.copy()

# =============================================================================
# Assaying anomaly
# =============================================================================
sort_var1 = "var_5pct"

# Get L, LS, and S portfolios for different parameters
ASSAYING = pbl.AssayAnomaly(
    data             = data,
    sort_var         = sort_var1,
    subset_filter    = {
        "bond_maturity": [
            (0, 4.99),
            (5, 9.99),
            (10, np.inf)
        ]
    },
    IDvar            = "cusip_id",
    RETvar           = ["bond_ret_mb", "bond_ret_me"],
    RATINGvar        = "sp_rating",
    holding_periods  = [1, 3,6],     # instead of [1, 3]
    nport          = [3, 5, 10],         # instead of [5, 10]
    ratings          = [None, 'NIG', 'IG'],
    turnover=False,
    save_idx=False   
)
# get TS of portfolios
assying_TS = ASSAYING.df

# get cross sectional stasts
assying_CS,stats = ASSAYING.summary_results(factor=f, nw_lag=3)