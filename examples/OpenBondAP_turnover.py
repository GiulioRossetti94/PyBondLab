# This script constructs portfolio turnover for a single sort strategy #
# Uses the OSBAP data
# Download the OSBAP data here:
# https://openbondassetpricing.com/wp-content/uploads/2024/07/WRDS_MMN_Corrected_Data_2024_July.csv

import numpy as np
import pandas as pd
import PyBondLab as pbl
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================================================================
# Load OSBAP File
# =============================================================================
url = "https://openbondassetpricing.com/wp-content/uploads/2024/07/WRDS_MMN_Corrected_Data_2024_July.csv"

# Read the CSV file directly from the URL into a pandas DataFrame
tbl1 = pd.read_csv(url)

# Create overall panel index
tbl1 = tbl1.reset_index()
tbl1['index'] = range(1,(len(tbl1)+1))
# =============================================================================
# Format the data
# =============================================================================
tbl1['date'] = pd.to_datetime(tbl1['date'])

# Your panel must be sorted by it's bond identifier and date #
tbl1 = tbl1.sort_values(['cusip','date'])

# WRDS data "starts" officially on "2002-08-31"
tbl1 = tbl1[tbl1['date'] >= "2002-08-31"]

# Column used for value weigted returns
# Here, we set the VW variable to the MMN-adjusted bond value
# Defined as MMN-adjusted bond price multiplied by amount outstanding
tbl1['VW'] = tbl1['BOND_VALUE']

# NOTE: for sorts on variables like size 'BOND_VALUE' etc.
# be sure to sign-correct this variable such that the ave. returns are
# increasing in that variable.

# renaming. could be skipped but then specify the ID and Ret variable in the .fit() method
# rename your primary bond identifier (i.e., ISSUE_ID or CUSIP) as simply "ID"
# rename your main return variable to 'ret'
# rename your main price variable to 'PRICE'
tbl1.rename(columns={"BONDPRC":"PRICE", # Rename MMN-adjusted BONDPRC to PRICE
                     "cusip":"ID",      # Rename cusip to ID
                     "rating":"RATING_NUM", # Rename rating to RATING_NUM
                     "bond_ret":"ret"},inplace=True) # Rename bond_ret to ret

# i.e., if you want to use duration-adjusted returns, rename 
# 'exretnc_bns' to 'ret'

# i.e., if your price, bond id and bond return variables were called:
#               PRC, BOND_ID and BOND_RET, we would have:
# tbl1.rename(columns={"PRC":"PRICE","BOND_ID":"ID","BOND_RET":"ret"},inplace=True)

#==============================================================================
#   USAGE: Single Sorts
#==============================================================================
# Play around with the numbers below
# set holding period to: 1 (monthly rebalance)
#                        3 (quarterly)
#                        6 (semi-annual)
#                       12 (annual)
# The portfolio returns are computed with the overlapping methodology
# of Jegadeesh and Titman (1993) when holding_period > 1

holding_period = 1             # holding period returns
n_portf        = 5            # number of portfolios, 5 is quintiles, 10 deciles
skip           = 0             # no skip period, i.e., observe at signal at t
                               # and trade in signal at t (no delay in trade)
# Recomended to keep skip = 0, this is standard.                              

# Indpendent of the package, create dataframe to store the sort output #
Strategies_LS = pd.DataFrame() # Long-Short (LS) strategies #

Sort_Vars = [
            'RATING_NUM',         # Rating
            'CS',             # Credit Spread (MMN-adjusted)
            'CS_6M_DELTA',    # 6m log change in CS (MMN-adjusted)
            'tmt',            # Time-to-Maturity
            'DURATION',       # Duration (computed with MMN-adjusted price)
            'BOND_RET'        # MMN-adjusted bond STREV signal
             ]

# For each variable in char_list, we compute the average characteristic for each portfolio bucket
char_list = ['bond_yield', 'cs']

char_results_ew = []
turnover_results_ew = []
for sort_var1 in Sort_Vars:
    print('Sorting on ' + str(sort_var1))
    
    # initialize the strategy
    single_sort = pbl.SingleSort(holding_period, 
                                 sort_var1, 
                                 n_portf,
                                 skip)
      
    # parameters
    # we do not conduct any filtering, so we leave most of this blank
    # to compute turnover, we set turnover = True
    params = {'strategy': single_sort,
              'rating':None, 
              'dynamic_weights':True,
              'chars': char_list,
              'turnover': True       
    }
    
    # Copy the dataframe for this iteration / sort
    data = tbl1.copy()
    
    # Fit the strategy to the data. 
    RESULTS = pbl.StrategyFormation(data, **params).fit()
    
    # extract the long-short strategies #
    _out = pd.concat(RESULTS.get_long_short(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) , 'VW_'+ str(sort_var1)]  
    Strategies_LS = pd.concat([Strategies_LS, _out], axis = 1)
    
    # save turnover results for equal-weighted portfolios
    ew_turnover,vw_turnover = RESULTS.get_ptf_turnover()
    turnover_results_ew.append(ew_turnover)

    # save characteristic results for equal-weighted portfolios
    ew_char, vw_char = RESULTS.get_chars()
    char_results_ew.append(ew_char)

    #
    
     
# Basic plots of cumulative returns #

# Calculate cumulative returns and plot
# Long-Short #
(1+Strategies_LS).cumprod().plot(figsize=(14, 8), linewidth=2)
plt.title('Cumulative Returns of Portfolios', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative ($) Value ', fontsize=14)
plt.grid(True)
plt.tight_layout()
# Customize the plot style
# plt.style.use('seaborn-darkgrid')
plt.show()


# Plotting bond_yield
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Avg Bond Yield for Equal-Weighted Portfolios')

for i, data in enumerate(char_results_ew):
    row, col = divmod(i, 2)
    data['bond_yield'].plot(ax=axes[row, col], title=f"portfolio sorted on: {Sort_Vars[i]}")
fig.subplots_adjust(hspace=0.5)

plt.show()

# Plotting cs
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Avg CS for Equal-Weighted Portfolios')

# Plotting cs
for i, data in enumerate(char_results_ew):
    row, col = divmod(i, 2)
    data['cs'].plot(ax=axes[row, col], title=f"portfolio sorted on: {Sort_Vars[i]}")
fig.subplots_adjust(hspace=0.5)

plt.show()

# Plotting Turnover
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Turnover')

for i, data in enumerate(turnover_results_ew):
    row, col = divmod(i, 2)
    data.plot(ax=axes[row, col], title=Sort_Vars[i])
fig.subplots_adjust(hspace=0.5)

plt.show()
################################ END ##########################################

