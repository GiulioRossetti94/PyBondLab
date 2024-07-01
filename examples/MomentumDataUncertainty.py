# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:32:52 2024

@author: phd19gr
"""
# Momentum data uncertainty.
# The script replicates the results in Dickerson, Robotti, Rossetti 2024
# It computes a total of n momentum strategies with different data cleaning (ex ante and ex post) procedures
#

import numpy as np
import pandas as pd
import PyBondLab as pbl

#==============================================================================
#   Load Data
#==============================================================================
tbl1         =  pd.read_csv(r'~/Dropbox/NIG_momentum/bondret_new.csv')#.reset_index()

tbl1.columns = tbl1.columns.str.upper()
tbl1['date'] = pd.to_datetime(tbl1['DATE'])

tbl1['AMOUNT_OUTSTANDING'] = np.abs(tbl1['AMOUNT_OUTSTANDING'])
tbl1['PRICE_L5M'] = np.abs(tbl1['PRICE_L5M'])
tbl1 = tbl1.sort_values(['ISSUE_ID','DATE'])

# starting point "2002-08-31"
tbl1 = tbl1[tbl1['date'] >= "2002-08-31"]
# column used for value weigted returns
tbl1['VW'] = (tbl1['PRICE_L5M'] * tbl1['AMOUNT_OUTSTANDING'])/1000

# renaming. could be skipped but then specify the ID and Ret variable in the .fit() method
tbl1.rename({"ISSUE_ID":"ID","PRICE_L5M":"PRICE","RET_L5M":"ret"},axis=1)
tbl1.rename({"ISSUE_ID":"ID","PRICE_L5M":"PRICE"},axis=1,inplace=True)

# Specify the universe of bonds based on Ratings: "NIG", "IG", None -> "NIG" + "IG"
RATING = None
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
#   Parameter
#==============================================================================
# here we create a dictionary in which we specify the adjustments that we are going to apply to the bond data
# eg: this compute the Momentum (3,3) excluding all bonds whose returns is higher than 20%
# param = {'strategy': mom33,
#  'rating': None,
#  'filters': {'adj': 'trim', 'level': 0.2}}


#==============================================================================
#   Param Specs: Winsorization
#==============================================================================
# to speed up computations, we input a df with the rolling percentile of the pooled cross section of bonds
# rolling percentile is needed to winsorize the signal. 
# at time t, we winsorize returns given the percentile of the distribution of bond returns from the beginning
# of the sample up to date t

BREAKPOINTS = pd.read_csv('breakpoints_update.csv',index_col=0)
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

# create a list with all the params specifications     
params_all = param_return + param_price + param_bounce + param_wins

print(len(params_all)) # Total number of strategies

# initialize empty lists to store the results
l_ew_ea = []
l_ew_ep = []

l_vw_ea = []
l_vw_ep = []


for i, params in enumerate(params_all):
    # create a copy of the original dataset
    tbl1_loop = tbl1.copy()
    
    # fit the strategy: input: the dataset, a dictionary with the parameters
    mom_res = pbl.StrategyFormation(tbl1_loop, **params).fit()
    print(i+1,'/',len(params_all),"\t",mom_res.name)

    # get the long short portfolios ex ante and ex post
    ew_ea, vw_ea = mom_res.get_long_short_ex_ante() 
    ew_ep, vw_ep = mom_res.get_long_short_ex_post()
    
    
    ## get the portfolios in a different way
    # ew_ea = mom_res.ewls_ea_df
    # ew_ep = mom_res.ewls_ep_df
    # vw_ea = mom_res.vwls_ea_df
    # vw_ep = mom_res.vwls_ep_df     

    # store the results
    l_ew_ea.append(ew_ea)
    l_ew_ep.append(ew_ep)

    l_vw_ea.append(vw_ea)
    l_vw_ep.append(vw_ep)
 
# concatenate the results
dfEWEA = pd.concat(l_ew_ea,axis=1)   # df with all the Equal-weighted ex ante strategies
dfEWEP = pd.concat(l_ew_ep,axis=1)   # df with all the Value-weighted ex ante strategies

dfVWEA = pd.concat(l_vw_ea,axis=1)   # df with all the Equal-weighted ex post strategies
dfVWEP = pd.concat(l_vw_ep,axis=1)   # df with all the Value-weighted ex post strategies
    
# dfEWEA = pd.concat(l_ew_ea,axis=1).to_csv('~/Desktop/EWEAmom_ALL_10w.csv')
# dfEWEP = pd.concat(l_ew_ep,axis=1).to_csv('~/Desktop/EWEPmom_ALL_10w.csv')

# dfVWEA = pd.concat(l_vw_ea,axis=1).to_csv('~/Desktop/VWEAmom_ALL_10w.csv')
# dfVWEP = pd.concat(l_vw_ep,axis=1).to_csv('~/Desktop/VWEPmom_ALL_10w.csv')    
