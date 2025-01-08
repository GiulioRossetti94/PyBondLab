# This script constructs a single sort on CS (credit spreads)
# It uses the banding procedure from Novy-Marx and Velikov RFS paper
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
                     'exretnc_bns':"ret"},inplace=True) # Rename 'exretnc_bns' to ret

# NOTE: For this example, we will use DURATION-ADJUSTED returns,
# we are NOT using total bond returns

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
n_portf        = 10            # number of portfolios, 5 is quintiles, 10 deciles
skip           = 0             # no skip period, i.e., observe at signal at t
                               # and trade in signal at t (no delay in trade)
# Recomended to keep skip = 0, this is standard. 
                             
# Indpendent of the package, create dataframe to store the sort output #
Strategies_LS = pd.DataFrame() # Long-Short (LS) strategies #
Strategies_L  = pd.DataFrame() # Long (L) strategies #
Strategies_S  = pd.DataFrame() # Short (S) strategies #
Strategies_LS_TO  = pd.DataFrame() # LS turnover #

# We will use the CS, credit spread factor as an example #
sort_var1 = 'CS'

banding = [0,1,2,3,4]

for b in banding:
    print('Sorting on CS with banding value set to ' + str(b))
    
    # initialize the strategy
    single_sort = pbl.SingleSort(holding_period, 
                                 sort_var1, 
                                 n_portf,
                                 skip)
      
    # parameters
    # we do not conduct any filtering, so we leave most of this blank
    params = {  'strategy': single_sort,
                'rating':None, 
                'banding_threshold': b,
                'turnover': True
    }

    
    # Copy the dataframe for this iteration / sort
    data = tbl1.copy()
    
    # Fit the strategy to the data. 
    RESULTS = pbl.StrategyFormation(data, **params).fit()
    
    # extract the long-short strategies #
    _out = pd.concat(RESULTS.get_long_short(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) +'b'+str(b) , 'VW_'+ str(sort_var1)+'b'+str(b)]  
    Strategies_LS = pd.concat([Strategies_LS, _out], axis = 1)
    
    # extract the long-only strategies #
    _out = pd.concat(RESULTS.get_long_leg(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) + str('_Long')+'b'+str(b) , 'VW_'+ str(sort_var1) + str('_Long')+'b'+str(b)]
    Strategies_L  = pd.concat([Strategies_L , _out], axis = 1)
    
    # extract the short-only strategies #
    _out = pd.concat(RESULTS.get_short_leg(), axis = 1) 
    _out.columns = ['EW_'+ str(sort_var1) + str('_Short')+'b'+str(b) , 'VW_'+ str(sort_var1) + str('_Short')+'b'+str(b)]
    Strategies_S  = pd.concat([Strategies_S , _out], axis = 1)
    
    # Get Turnover #
    ew_turn,vw_turn = RESULTS .get_ptf_turnover()
    _out_ew = ((ew_turn.iloc[:,0]+ew_turn.iloc[:,(n_portf-1)])/2).to_frame()
    _out_ew.columns = ['EW_'+ str(sort_var1)+'_TO'+'b'+str(b)]
    
    _out_vw = ((vw_turn.iloc[:,0]+vw_turn.iloc[:,(n_portf-1)])/2).to_frame()
    _out_vw.columns = ['VW_'+ str(sort_var1)+'_TO'+'b'+str(b)]
    Strategies_LS_TO = pd.concat([Strategies_LS_TO,  _out_ew, _out_vw], axis = 1)
        
     
# Basic plots of cumulative returns #

# Calculate cumulative returns and plot
# Long-Short #
# Identify EW and VW columns
ew_cols = [col for col in Strategies_LS.columns if col.startswith("EW")]
vw_cols = [col for col in Strategies_LS.columns if col.startswith("VW")]

# Create 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Panel A: EW
(1 + Strategies_LS[ew_cols]).cumprod().plot(ax=axes[0], grid=True)
axes[0].set_title("Panel A: EW")
axes[0].set_xlabel("Date", fontsize=14)
axes[0].set_ylabel("Cumulative ($) Value", fontsize=14)

# Panel B: VW
(1 + Strategies_LS[vw_cols]).cumprod().plot(ax=axes[1], grid=True)
axes[1].set_title("Panel B: VW")
axes[1].set_xlabel("Date", fontsize=14)
axes[1].set_ylabel("Cumulative ($) Value", fontsize=14)

# Main title (figure-level title)
fig.suptitle("Cumulative Returns of Portfolios", fontsize=16)

plt.tight_layout()
plt.show()

# Bar plot of average returns (premia) #
mean_returns = Strategies_LS.mean() * 100
std_returns = Strategies_LS.std() * 100
n = len(Strategies_LS)  
std_error = std_returns / np.sqrt(n)
conf_95 = 1.96 * std_error

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(
    mean_returns.index,         
    mean_returns.values,        
    yerr=conf_95.values,        
    capsize=4,                  
    color='skyblue',
    edgecolor='black'
)

ax.set_ylabel("Mean Return (%)", fontsize=12)
ax.set_title("Average Returns", fontsize=14)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# Bar plot of average turnover #
# Separate columns for EW and VW
ew_cols = [col for col in Strategies_LS_TO.columns if col.startswith("EW")]
vw_cols = [col for col in Strategies_LS_TO.columns if col.startswith("VW")]

# Calculate mean turnover for each subset, scale by 100 (convert to %)
mean_ew = Strategies_LS_TO[ew_cols].mean() * 100
mean_vw = Strategies_LS_TO[vw_cols].mean() * 100

# Create 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Panel A: EW
axes[0].bar(mean_ew.index, mean_ew.values, color='skyblue', edgecolor='black')
axes[0].set_title("Panel A: EW Turnover", fontsize=14)
axes[0].set_ylabel("Mean Turnover (%)", fontsize=12)

# Rotate x-axis labels to avoid overlap
axes[0].tick_params(axis='x', rotation=45)

# Panel B: VW
axes[1].bar(mean_vw.index, mean_vw.values, color='lightgreen', edgecolor='black')
axes[1].set_title("Panel B: VW Turnover", fontsize=14)

# Rotate x-axis labels to avoid overlap
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()

# A band of 1 seems to do the best! #

# Import factors for risk-adjustment #
# Use EW MKTB for simplicity #
MKTb = tbl1.groupby("date")[['ret']].mean()
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

print(RiskPremia.T)

# Alpha #
Alpha       = (pd.DataFrame(np.array(alpha)) ).round(2).T  
Alpha.index = ['Alpha (%)' ]
Alpha.columns= Strategies_LS.columns
TstatsA     = pd.DataFrame(np.array(tstats_alpha)).T
TstatsA.index = [str('t-stat') ]
TstatsA  = "(" + TstatsA.astype(str)  + ")" 
TstatsA .columns= Strategies_LS.columns
         
Alpha = pd.concat([Alpha, TstatsA], axis = 0)

print(Alpha.T)
################################ END ##########################################
