# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:32:52 2024

@author: phd19gr
"""

import numpy as np
import pandas as pd
import PyBondLab as pbl

#==============================================================================
#   Load Data
#==============================================================================

tbl1         =  pd.read_csv(r'~/Dropbox/NIG_momentum/bondret_new.csv')#.reset_index()

tbl1.columns = tbl1.columns.str.upper()
tbl1['date'] = pd.to_datetime(tbl1['DATE'])
datelist = pd.Series( tbl1['date'].unique()).sort_values()
datelist = list(datelist)

tbl1['ret'] = tbl1['RET_L5M']
tbl1['AMOUNT_OUTSTANDING'] = np.abs(tbl1['AMOUNT_OUTSTANDING'])
tbl1['PRICE_L5M'] = np.abs(tbl1['PRICE_L5M'])
tbl1 = tbl1.sort_values(['ISSUE_ID','DATE'])
tbl1      = tbl1[tbl1['date'] >= "2002-08-31"]

tbl1['VW'] = (tbl1['PRICE_L5M'] * tbl1['AMOUNT_OUTSTANDING'])/1000

tbl1.rename({"ISSUE_ID":"ID","PRICE_L5M":"PRICE"},axis=1)

tbl1.rename({"ISSUE_ID":"ID","PRICE_L5M":"PRICE"},axis=1,inplace=True)
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
BREAKPOINTS = pd.read_csv('../breakpoints_update.csv',index_col=0)
BREAKPOINTS.index = pd.to_datetime(BREAKPOINTS.index)
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

price_filter_up   = np.arange(150,1100,100) 
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

return_filter_up   = np.arange(0.20,1.00,0.05) 
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
      
bounce_filter_up        = np.arange(-0.10,0,0.01) 
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
#   Param Specs: Trim Comb
#==============================================================================         
params_all = param_return + param_price + param_bounce + param_wins

print(len(params_all))

l_ew_ea = []
l_ew_ep = []

l_vw_ea = []
l_vw_ep = []

# params_all = param_wins

params_all = param_wins
for i, params in enumerate(params_all):
    
    tbl1_loop = tbl1.copy()
    
    rating_res = pbl.StrategyFormation(tbl1_loop, **params).fit()
    
    # printing strategy computed 
    print(i+1,'/',len(params_all),"\t",rating_res.name)
    
    ew_ea = rating_res.ewls_ea_df
    ew_ep = rating_res.ewls_ep_df
    vw_ea = rating_res.vwls_ea_df
    vw_ep = rating_res.vwls_ep_df     
  
    l_ew_ea.append(ew_ea)
    l_ew_ep.append(ew_ep)

    l_vw_ea.append(vw_ea)
    l_vw_ep.append(vw_ep)
    
# dfEWEA = pd.concat(l_ew_ea,axis=1).to_csv('EWEArating_ALL_5.csv')
# dfEWEP = pd.concat(l_ew_ep,axis=1).to_csv('EWEPrating_ALL_5.csv')

# dfVWEA = pd.concat(l_vw_ea,axis=1).to_csv('VWEArating_ALL_5.csv')
# dfVWEP = pd.concat(l_vw_ep,axis=1).to_csv('VWEPrating_ALL_5.csv')    
