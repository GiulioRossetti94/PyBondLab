#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:04:34 2024

@author: u1972481
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from pandas.tseries.offsets import *
import statsmodels.api as sm
import PyBondLab as pbl

import time
start_time = time.time()
# =============================================================================
# Load OSBAP File
# =============================================================================
url = r"C:\Users\Alex\Dropbox\DR_Bonds\data_2023\TRACE\OSBAP_Updates_2023_2025\WRDS_MMN_Corrected_Data_2024_July.csv"
url  = "https://openbondassetpricing.com/wp-content/uploads/2024/07/WRDS_MMN_Corrected_Data_2024_July.csv"
url = "~/Dropbox/NIG_momentum/bondret_new_yoshio_debug.csv"
# Read the CSV file directly from the URL into a pandas DataFrame
tbl1 = pd.read_csv(url)

# Create overall panel index
tbl1 = tbl1.reset_index()
tbl1['index'] = range(1,(len(tbl1)+1))

# =============================================================================
# Format the data
# =============================================================================
tbl1['DATE'] = pd.to_datetime(tbl1['DATE'])

# Your panel must be sorted by it's bond identifier and date #
# tbl1 = tbl1.sort_values(['cusip','date'])

# WRDS data "starts" officially on "2002-08-31"
# tbl1 = tbl1[tbl1['DATE'] >= "2002-09-30"]

# Column used for value weigted returns
# Here, we set the VW variable to the MMN-adjusted bond value
# Defined as MMN-adjusted bond price multiplied by amount outstanding
tbl1['VW'] = tbl1['PRICE_EOM'] * tbl1['AMOUNT_OUTSTANDING']

# NOTE: for sorts on variables like size 'BOND_VALUE' etc.
# be sure to sign-correct this variable such that the ave. returns are
# increasing in that variable.

# renaming. could be skipped but then specify the ID and Ret variable in the .fit() method
# rename your primary bond identifier (i.e., ISSUE_ID or CUSIP) as simply "ID"
# rename your main return variable to 'ret'
# rename your main price variable to 'PRICE'

# Establish unique IDs to conform with package
# N = len(np.unique(tbl1.cusip))
# ID = dict(zip(np.unique(tbl1.cusip).tolist(),np.arange(1,N+1)))
# tbl1['ID'] = tbl1.cusip.apply(lambda x: ID[x])

tbl1.rename(columns={
                    
                     'RET_L5M':"ret"},inplace=True) # Rename bond_ret to ret


# tbl1['date'] = pd.to_datetime(tbl1['DATE'])
# tbl1.drop(['DATE'], axis = 1, inplace = True)
datelist = pd.Series( tbl1['DATE'].unique()).sort_values()
datelist = list(datelist)
tbl1.columns

tbl1 = tbl1.sort_values(['ID','DATE'])
tbl1 = tbl1.loc[:, ~tbl1.columns.duplicated()]

load_data_time = time.time()
print(f"loading data time: {load_data_time - start_time }")

holding_period = 1             # holding period returns
n_portf        = 5            # number of portfolios, 5 is quintiles, 10 deciles
skip           = 0             # no skip period, i.e., observe at signal at t
                               # and trade in signal at t (no delay in trade)
varname = 'YIELD'

single_sort = pbl.SingleSort(holding_period, 
                             varname, 
                             n_portf,
                             skip)
strategy_time = time.time()
print(f"strategy time: {strategy_time -load_data_time }")

chars = ['YIELD', 'PRICE_L5M','TMT','T_SPREAD']
# parameters
    # we do not conduct any filtering, so we leave most of this blank
params = {'strategy': single_sort,
           'rating':None, 
           'chars':chars,            # list of chars for bin statistics
          'dynamic_weights':False,   # dynamic weights with horizon
          'turnover': True,
          'filters':{'adj':"trim","level":0.2}
}


data = tbl1.copy()
# Fit the strategy to the data. 
mod = pbl.StrategyFormation(data, **params).fit(DATEvar="DATE")

fit_time = time.time()
print(f"fit time: {fit_time - strategy_time }")

#
ewls,vwls = mod.get_long_short()    # get long-short portfolios

# get characteristics of the portfolios in each bin
char_ewls, char_vwls = mod.get_chars()

# get turnover of the strategy
ew_turnover,vw_turnover = mod.get_ptf_turnover()

# exmpost
ewls_ep,vwls_ep = mod.get_long_short_ex_post()
char_ewls_ep, char_vwls_ep = mod.get_chars_ex_post()
ew_turnover_ep,vw_turnover_ep = mod.get_ptf_turnover_ex_post()

print(mod.stats_bonds_adj())








