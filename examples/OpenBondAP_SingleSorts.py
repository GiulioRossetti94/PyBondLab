# This script constructs several single sorts #
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
Strategies_L  = pd.DataFrame() # Long (L) strategies #
Strategies_S  = pd.DataFrame() # Short (S) strategies #

Sort_Vars = [
            'RATING_NUM',         # Rating
            'CS',             # Credit Spread (MMN-adjusted)
            'CS_6M_DELTA',    # 6m log change in CS (MMN-adjusted)
            'tmt',            # Time-to-Maturity
            'DURATION',       # Duration (computed with MMN-adjusted price)
            'BOND_RET'        # MMN-adjusted bond STREV signal
             ]

for sort_var1 in Sort_Vars:
    print('Sorting on ' + str(sort_var1))
    
    # initialize the strategy
    single_sort = pbl.SingleSort(holding_period, 
                                 sort_var1, 
                                 n_portf,
                                 skip)
      
    # parameters
    # we do not conduct any filtering, so we leave most of this blank
    params = {'strategy': single_sort,
              'rating':None,         
    }
    
    # Copy the dataframe for this iteration / sort
    data = tbl1.copy()
    
    # Fit the strategy to the data. 
    RESULTS = pbl.StrategyFormation(data, **params).fit()
    
    # extract the long-short strategies #
    _out = pd.concat(RESULTS.get_long_short(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) , 'VW_'+ str(sort_var1)]  
    Strategies_LS = pd.concat([Strategies_LS, _out], axis = 1)
    
    # extract the long-only strategies #
    _out = pd.concat(RESULTS.get_long_leg(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) + str('_Long') , 'VW_'+ str(sort_var1) + str('_Long')]
    Strategies_L  = pd.concat([Strategies_L , _out], axis = 1)
    
    # extract the short-only strategies #
    _out = pd.concat(RESULTS.get_short_leg(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) + str('_Short') , 'VW_'+ str(sort_var1) + str('_Short')]
    Strategies_S  = pd.concat([Strategies_S , _out], axis = 1)
    
     
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
#plt.style.use('seaborn-darkgrid')
plt.show()

# Import factors for risk-adjustment #
# Use EW MKTB for simplicity #
MKTb = tbl1.groupby("date")[['exretn']].mean()
MKTb.columns = ['MKTB']

# Risk premia/alpha and t-statistics #
tstats                =  list()
tstats_alpha          =  list()
alpha                 =  list()

for i,s in enumerate(Strategies_LS.columns):
    print(s)
    regB = sm.OLS(Strategies_LS.iloc[:,i].values, 
                      pd.DataFrame(np.tile(1,(len(Strategies_LS.iloc[:,i]),1)) ).values,
                      missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':3})
       
    tstats.append(np.round(regB.tvalues[0],2))  
    
    _HML = Strategies_LS.iloc[:,i].to_frame()
    _HML = _HML.merge(MKTb,how="inner",left_index=True,right_index=True).dropna()
    regB = sm.OLS(_HML.iloc[:,0], 
                      sm.add_constant(_HML[['MKTB']]),                      
                      ).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    regB.summary()
        
    tstats_alpha.append(np.round(regB.tvalues.iloc[0] ,2)) 
    alpha.append( np.round(regB.params.iloc [0]*100,2))

# Premia #
Mean       = (pd.DataFrame(np.array(Strategies_LS.mean())) * 100).round(2).T  
Mean.index = ['Avg Ret(%)' ]
Mean.columns= Strategies_LS.columns
Tstats     = pd.DataFrame(np.array(tstats)).T
Tstats.index = [str('t-stat') ]
Tstats  = "(" + Tstats.astype(str)  + ")" 
Tstats .columns= Strategies_LS.columns
         
RiskPremia = pd.concat([Mean, Tstats], axis = 0)

print(RiskPremia)

# Alpha #
Alpha       = (pd.DataFrame(np.array(alpha)) ).round(2).T  
Alpha.index = ['Alpha (%)' ]
Alpha.columns= Strategies_LS.columns
TstatsA     = pd.DataFrame(np.array(tstats_alpha)).T
TstatsA.index = [str('t-stat') ]
TstatsA  = "(" + TstatsA.astype(str)  + ")" 
TstatsA .columns= Strategies_LS.columns
         
Alpha = pd.concat([Alpha, TstatsA], axis = 0)

print(Alpha)
################################ END ##########################################
