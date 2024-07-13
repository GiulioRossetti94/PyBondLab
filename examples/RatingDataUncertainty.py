# Rating data uncertainty.
# The script replicates the results in Dickerson, Robotti, Rossetti 2024
# as it relates to data uncertainty with corporate bond rating strategy

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
wrds_username = '' # Input your WRDS username
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
# In this example, we specify None, which includes ALL bonds only 
# i.e., we use ALL bonds, so we set RATING = None
RATING = None
#==============================================================================
#   USAGE: Single Sort Rating
#==============================================================================

# Initialize the class
holding_period = 1
n_portf        = 5
sort_var    = 'RATING_NUM'

# Initialize the class
rating_single_sort = pbl.SingleSort(holding_period,sort_var,n_portf )

#==============================================================================
#   Param Specs: Winsorization
#==============================================================================
BREAKPOINTS = pbl.load_breakpoints_WRDS()
BREAKPOINTS.index = pd.to_datetime(BREAKPOINTS.index)

# Pick any breakpoints that you like #
# If you use the pre-loaded file, pick breakpoiunts that are found in the file
winsorization_level_up      = np.arange(98.0,99.8,0.5)

param_wins = []

for w in winsorization_level_up:
    for loc in ['right','left','both']:
        w = round(w,4)
        params = {'strategy':rating_single_sort,'rating':RATING,
            'filters': {'adj':'wins','level': w,'location':loc,'df_breakpoints':BREAKPOINTS},
            }   
        param_wins.append(params)


#==============================================================================
#   Param Specs: Price Exclusion
#==============================================================================

# These filters exclude bonds based on their prices in the right tail, 
# i.e., bonds with very HIGH prices
price_filter_up   = np.arange(150,1100,100) 

# These filters exclude bonds based on their prices in the left tail, 
# i.e., bonds with very LOW prices
price_filter_down = np.arange(20,0,-2) 
price_filter      = np.concatenate([price_filter_up,price_filter_down])

param_price = []
for w in price_filter:
        w = round(w,4)
        params = {'strategy':rating_single_sort,'rating':RATING,
            'filters': {'adj':'price','level': w},
            }   
        param_price.append(params)

for up, down in zip(price_filter_up, price_filter_down):
    params = {'strategy':rating_single_sort,'rating':RATING,
        'filters': {'adj':'price','level': [down, up]},
        }   
    param_price.append(params)

del price_filter_up, price_filter_down
#==============================================================================
#   Param Specs: Return Exclusion
#==============================================================================

# These filters exclude bonds based on their returns in the right tail, 
# i.e., bonds with very HIGH returns
return_filter_up   = np.arange(0.20,1.00,0.05) 

# These filters exclude bonds based on their returns in the left tail, 
# i.e., bonds with very LOW returns
return_filter_down = np.arange(-0.95,-0.15,0.05) 
return_filter      = np.round( np.concatenate([return_filter_up,return_filter_down]),2)

param_return = []
for w in return_filter:
        w = round(w,4)
        params = {'strategy':rating_single_sort,'rating':RATING,
            'filters': {'adj':'trim','level': w},
            }   
        param_return.append(params)
        

for w in return_filter_up:
    w = round(w,4)
    params = {'strategy':rating_single_sort,'rating':RATING,
        'filters': {'adj':'trim','level': [-w,w]},
        }   
    param_return.append(params)

del return_filter_up, return_filter_down       
#==============================================================================
#   Param Specs: Bounce Exclusion
#============================================================================== 
# This is a illiquidity filter first used in:
# Are Capital Market Anomalies Common to Equity and Corporate Bond Markets? An Empirical Investigation
# https://www.jstor.org/stable/26590444      

# These filters exclude bonds based on their Rt x R-1 (i.e., product of returns)
# in the left tail
# i.e., bonds with very HIGH bounceback in the left tail
bounce_filter_up        = np.arange(-0.10,0,0.01) 

# These filters exclude bonds based on their Rt x R-1 (i.e., product of returns)
# in the right tail
# i.e., bonds with very HIGH bounceback in the right tail
bounce_filter_down      = np.arange( 0.01,0.11,0.01) 
bounce_filter           = np.concatenate([bounce_filter_up,bounce_filter_down])

    
param_bounce = []
for w in bounce_filter:
        w = round(w,4)
        params = {'strategy':rating_single_sort,'rating':RATING,
            'filters': {'adj':'bounce','level': w},
            }   
        param_bounce.append(params)
        

 
for idx, w in enumerate(bounce_filter_down):
    w = round(w,4)
    params = {'strategy':rating_single_sort,'rating':RATING,
        'filters': {'adj':'bounce','level': [-w,w]},
        }   
    param_bounce.append(params)
  
del bounce_filter_up, bounce_filter_down

#==============================================================================
#   Param Specs: 
#==============================================================================     
# Adds up all of the filtering dictionaries into a single container    
params_all = param_return + param_price + param_bounce + param_wins

print(len(params_all))

l_ew_ea = [] # Equal-Weight (ew) _ Ex ante (ea) filtered strategies
l_ew_ep = [] # Equal-Weight (ew) _ Ex post (ep) filtered strategies

l_vw_ea = [] # Value-Weight (vw) _ Ex ante (ea) filtered strategies
l_vw_ep = [] # Value-Weight (vw) _ Ex post (ea) filtered strategies


for i, params in enumerate(params_all):
    
    tbl1_loop = tbl1.copy() # Copies the main dataframe for each new iteration #
    
    # Estimates the sort on the data given the parameters in the loop
    rating_res = pbl.StrategyFormation(tbl1_loop, **params).fit()
    
    # printing strategy computed 
    print(i+1,'/',len(params_all),"\t",rating_res.name)
    
    # Output extraction #
    ew_ea = rating_res.ewls_ea_df
    ew_ep = rating_res.ewls_ep_df
    vw_ea = rating_res.vwls_ea_df
    vw_ep = rating_res.vwls_ep_df     
    
    # Append to a large list
    l_ew_ea.append(ew_ea)
    l_ew_ep.append(ew_ep)

    l_vw_ea.append(vw_ea)
    l_vw_ep.append(vw_ep)

# =============================================================================
# Export to file
# =============================================================================
# Equal-Weight
pd.concat(l_ew_ea, axis = 1).to_csv(r'exante_rating_uncertainty_ew.csv')
pd.concat(l_ew_ep, axis = 1).to_csv(r'expost_rating_uncertainty_ew.csv')

# Value-Weight
pd.concat(l_vw_ea, axis = 1).to_csv(r'exante_rating_uncertainty_ew.csv')
pd.concat(l_vw_ep, axis = 1).to_csv(r'expost_rating_uncertainty_ew.csv')
# =============================================================================
