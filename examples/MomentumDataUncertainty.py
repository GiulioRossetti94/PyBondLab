# Momentum data uncertainty.
# The script replicates the results in Dickerson, Robotti, Rossetti 2024
# It computes a total of n momentum strategies with different data cleaning (ex ante and ex post) procedures
#

import numpy as np
import pandas as pd
import wrds
import PyBondLab as pbl

#==============================================================================
#   Load Data
#==============================================================================

# =============================================================================
# Option 1: access data directly from WRDS
# =============================================================================
# Assumes you have a valid WRDS account and have set-up your cloud access #
# See:
# https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/
wrds_username = 'phd18ad1' # Input your WRDS username
db = wrds.Connection(wrds_username = wrds_username )

tbl1 = db.raw_sql("""SELECT  DATE, ISSUE_ID,CUSIP, RATING_NUM, RET_L5M,AMOUNT_OUTSTANDING,
                                TMT, N_SP, PRICE_L5M                         
                        FROM wrdsapps.bondret
                  """)
# Required because the WRDS data comes with "duplicates" in the index
# does not affect data, but the "index" needs to be re-defined #                 
tbl1 = tbl1.reset_index()
tbl1['index'] = range(1,(len(tbl1)+1))

# =============================================================================
# Option 2: Load any bond dataset you have saved to file
# =============================================================================
# file = '' # Input your file name, assumed to be saved as .csv #
# tbl1         =  pd.read_csv(file)

tbl1['index'] = range(1,(len(tbl1)+1))
# =============================================================================
# Format the data
# =============================================================================
tbl1.columns = tbl1.columns.str.upper()
tbl1['date'] = pd.to_datetime(tbl1['DATE'])
tbl1['AMOUNT_OUTSTANDING'] = np.abs(tbl1['AMOUNT_OUTSTANDING'])
tbl1['PRICE_L5M'] = np.abs(tbl1['PRICE_L5M'])
tbl1 = tbl1.sort_values(['ISSUE_ID','DATE'])

# WRDS data "starts" officially on "2002-08-31"
tbl1 = tbl1[tbl1['date'] >= "2002-08-31"]

# Column used for value weigted returns
tbl1['VW'] = (tbl1['PRICE_L5M'] * tbl1['AMOUNT_OUTSTANDING'])/1000

# renaming. could be skipped but then specify the ID and Ret variable in the .fit() method
# rename your primary bond identifier (i.e., ISSUE_ID or CUSIP) as simply "ID"
# rename your return variable to 'ret'
# rename your price variable to 'PRICE'
tbl1.rename(columns={"PRICE_L5M":"PRICE","ISSUE_ID":"ID","RET_L5M":"ret"},inplace=True)

# Specify the universe of bonds based on Ratings: "NIG", "IG", None -> "NIG" + "IG"
# In this example, we specify NIG, which includes high-yield bonds only 
RATING = 'NIG'
#==============================================================================
#   USAGE: Single Sort Rating
#==============================================================================
# Initialize two momentum strategies (6,6) and (3,3) skipping one month (skip =1).
# K is the holding period and J is the formation period. 
# Bonds are sorted in deciles based on the momentum signal 

# Initialize the class
n_portf = 10
skip    = 1

# Initialize the class
#(3,3): signal is computed as the cumulative return over the J=3 previous months skipping the most recent observation (skip = 1). 
mom33 = pbl.Momentum(K = 3,J = 3, skip = skip ,nport = n_portf)
#(6,6): signal is computed as the cumulative return over the J=6 previous months skipping the most recent observation (skip = 1). 
mom66 = pbl.Momentum(K = 6,J = 6, skip = skip ,nport = n_portf)


#==============================================================================
#   Param Specs: Winsorization
#==============================================================================
BREAKPOINTS = pbl.load_breakpoints_WRDS()
BREAKPOINTS.index = pd.to_datetime(BREAKPOINTS.index)
winsorization_level_up      = np.arange(98.0,99.8,0.5)


param_wins = []    # create a list with all the param dictionaries related to winsorization

for x in [mom33,mom66]:
    for w in winsorization_level_up:
        for loc in ['right','left','both']:
            w = round(w,4)
            params = {'strategy':x,'rating':RATING,
                'filters': {'adj':'wins','level': w,'location':loc,'df_breakpoints':BREAKPOINTS}, # "df_breakpoints" can be omitted. the function will automatically compute the rolling percentile but it will slow everything down
                }   
            param_wins.append(params)

#==============================================================================
#   Param Specs: Price Exclusion
#==============================================================================

price_filter_up   = np.arange(150,1100,100) 
price_filter_down = np.arange(20,0,-2) 
price_filter      = np.concatenate([price_filter_up,price_filter_down])

param_price = []    # create a list with all the param dictionaries related to price exclusions

for x in [mom33,mom66]:
    for w in price_filter:
            w = round(w,4)
            params = {'strategy':x,'rating':RATING,
                'filters': {'adj':'price','level': w},
                }   
            param_price.append(params)

for x in [mom33,mom66]:
    for up, down in zip(price_filter_up, price_filter_down):
        params = {'strategy':x,'rating':RATING,
            'filters': {'adj':'price','level': [down, up]},   # if a list is passed, exclude bonds whose prices are < down and > up
            }   
        param_price.append(params)

del price_filter_up, price_filter_down
#==============================================================================
#   Param Specs: Return Exclusion
#==============================================================================

return_filter_up   = np.arange(0.20,1.00,0.05) 
return_filter_down = np.arange(-0.95,-0.15,0.05) 
return_filter      = np.round( np.concatenate([return_filter_up,return_filter_down]),2)

param_return = []   # create a list with all the param dictionaries related to return exclusions
for x in [mom33,mom66]:
    for w in return_filter:
            w = round(w,4)
            params = {'strategy':x,'rating':RATING,
                'filters': {'adj':'trim','level': w},
                }   
            param_return.append(params)
        
for x in [mom33,mom66]:
    for w in return_filter_up:
        w = round(w,4)
        params = {'strategy':x,'rating':RATING,
            'filters': {'adj':'trim','level': [-w,w]},  # if a list is passed, exclude bonds whose returns are < -w and > w
            }   
        param_return.append(params)

del return_filter_up, return_filter_down       
#==============================================================================
#   Param Specs: Bounce Exclusion
#============================================================================== 
bounce_filter_up        = np.arange(-0.10,0,0.01) 
bounce_filter_down      = np.arange( 0.01,0.11,0.01) 
bounce_filter           = np.concatenate([bounce_filter_up,bounce_filter_down])

    
param_bounce = []   # create a list with all the param dictionaries related to bounce back exclusions
for x in [mom33,mom66]:
    for w in bounce_filter:
            w = round(w,4)
            params = {'strategy':x,'rating':RATING,
                'filters': {'adj':'bounce','level': w},
                }   
            param_bounce.append(params)
        

for x in [mom33,mom66]:
    for idx, w in enumerate(bounce_filter_down):
        w = round(w,4)
        params = {'strategy':x,'rating':RATING,
            'filters': {'adj':'bounce','level': [-w,w]},
            }   
        param_bounce.append(params)
      
del bounce_filter_up, bounce_filter_down

#==============================================================================
#   Param Specs: 
#==============================================================================     
# Adds up all of the filtering dictionaries into a single container    
params_all = param_return + param_price + param_bounce + param_wins

print(len(params_all)) # Total number of strategies

# initialize empty lists to store the results

l_ew_ea = [] # Equal-Weight (ew) _ Ex ante (ea) filtered strategies
l_ew_ep = [] # Equal-Weight (ew) _ Ex post (ep) filtered strategies

l_vw_ea = [] # Value-Weight (vw) _ Ex ante (ea) filtered strategies
l_vw_ep = [] # Value-Weight (vw) _ Ex post (ea) filtered strategies


for i, params in enumerate(params_all):
    # create a copy of the original dataset
    tbl1_loop = tbl1.copy()
    
    # fit the strategy: input: the dataset, a dictionary with the parameters
    mom_res = pbl.StrategyFormation(tbl1_loop, **params).fit()
    print(i+1,'/',len(params_all),"\t",mom_res.name)

    # get the long short portfolios ex ante and ex post
    ew_ea, vw_ea = mom_res.get_long_short() 
    ew_ep, vw_ep = mom_res.get_long_short_ex_post()
       
    # Append to a large list
    l_ew_ea.append(ew_ea)
    l_ew_ep.append(ew_ep)

    l_vw_ea.append(vw_ea)
    l_vw_ep.append(vw_ep)
 
# =============================================================================
# Export to file
# =============================================================================
# Equal-Weight
pd.concat(l_ew_ea, axis = 1).to_csv(r'exante_momentum_uncertainty_ew.csv')
pd.concat(l_ew_ep, axis = 1).to_csv(r'expost_momentum_uncertainty_ew.csv')

# Value-Weight
pd.concat(l_vw_ea, axis = 1).to_csv(r'exante_momentum_uncertainty_ew.csv')
pd.concat(l_vw_ep, axis = 1).to_csv(r'expost_momentum_uncertainty_ew.csv')
# =============================================================================
