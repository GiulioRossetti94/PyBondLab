# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:28:52 2024

@authors: Giulio Rossetti & Alex Dickerson
"""

import numpy as np
import pandas as pd
from .FilterClass import Filter
from .StrategyClass import *
from .iotools.PyBondLabResults import StrategyResults
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt

from PyBondLab.data.WRDS import load

def load_breakpoints_WRDS() -> pd.DataFrame:
    """
    Load the breakpoints (rolling percentiles) WRDS data
    """
    return load()

class StrategyFormation:
    def __init__(self, data: pd.DataFrame,strategy: Strategy, rating: str = None, chars: dict = None, dynamic_weights: bool = False,turnover: bool = False ,banding_threshold: float = None,filters: dict = None):
        
        self.data_raw = data.copy()
        self.data     = data.copy()
        # Check if 'date' column exists before creating datelist
        self.datelist = pd.Series(self.data['date'].unique()).sort_values().tolist() if 'date' in self.data.columns else None
        
        # STRATEGY PARAMETERS
        self.strategy = strategy
        self.nport    = strategy.nport
        
        # RATING, CHARS, WEIGHTS
        self.rating            = self._validate_rating(rating)    # validate and set ratings
        self.chars             = chars if chars else None         # used to compute stats for portfolio bins
        self.dynamic_weights   = dynamic_weights 
        self.turnover          = turnover                         # compute turnover. False by default to speed up computations
        self.banding_threshold = banding_threshold                # threshold for banding
        
        # PARAMETERS FOR FILTERS/ADJUSTMENTS
        self.filters                = self._validate_filters(filters)
        self.adj                    = self.filters.get('adj')
        self.w                      = self.filters.get('level')
        self.loc                    = self.filters.get('location')
        self.perc_breakpoints       = self.filters.get('df_breakpoints') if self.adj == 'wins' else None
        self.price_threshold        = self.filters.get('price_threshold', 25) if self.adj == 'price' else None
        
        # CREATE NAMES, INITIALIZE DFs,  SAVE PARAMETERS FOR IO OPERATIONS
        self.name = self._create_name(self.rating, self.strategy.str_name)    
        self.signal_names = self._get_signal_names()
        self._initialize_dfs(filters)  # initialize variables for storing results
        self.stored_params = self._store_params()

    def _store_params(self):
        """ store parameters in a dictionary """
        return {
            "strategy": self.strategy,
            "rating": self.rating,
            "chars": self.chars,
            "dynamic_weights": self.dynamic_weights,
            "turnover": self.turnover,
            "banding_threshold": self.banding_threshold,
            "filters": self.filters
        }

    def _validate_rating(self, rating):
        valid_ratings = ["NIG", "IG", None]
        if rating not in valid_ratings:
            raise ValueError(f"Invalid rating: {rating}. Valid options are {valid_ratings}")
        return rating
    
    def _validate_filters(self, filters):
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise ValueError("Filters should be passed as a dictionary")
        
        adj = filters.get('adj')
        valid_adj = ["trim", "wins", "price", "bounce"]
        
        if adj is not None and adj not in valid_adj:
            raise ValueError(f"Invalid filtering option: {adj}. Valid options are {valid_adj}")
        return filters

    def _create_name(self, rating, strategy_name):
        if rating is None:
            return f"ALL_{strategy_name}"
        return f"{rating}_{strategy_name}"

    def _initialize_dfs(self, filters):
        # ex ante dfs
        self.ewls_ea_long_df      = None
        self.vwls_ea_long_df      = None
        self.ewls_ea_short_df     = None
        self.vwls_ea_short_df     = None
        self.ewls_ea_df           = None
        self.vwls_ea_df           = None
        self.ewport_ea            = None
        self.vwport_ea            = None
        self.ewport_ea_df         = None
        self.vwport_ea_df         = None
        self.ewport_weight_hor_ea = None
        self.vwport_weight_hor_ea = None
        self.ewturnover_ea_df     = None
        self.vwturnover_ea_df     = None
        self.ew_chars_ea          = None
        self.vw_chars_ea          = None
        # es post dfs
        if filters:
            self.ewls_ep_df           = None 
            self.vwls_ep_df           = None
            self.ewls_ep_long_df      = None
            self.vwls_ep_long_df      = None
            self.ewls_ep_short_df     = None
            self.vwls_ep_short_df     = None
            self.ewport_ep            = None
            self.vwport_ep            = None
            self.ewport_ep_df         = None
            self.vwport_ep_df         = None
            self.ewport_weight_hor_ep = None
            self.vwport_weight_hor_ep = None
            self.ewturnover_ep_df     = None
            self.vwturnover_ep_df     = None
            self.ew_chars_ep          = None
            self.vw_chars_ep          = None         
    
    def _store_results(self, filters):
        # Base results (ex ante)
        self._stored_results = {
            "name": self.name,
            "ewls": self.ewls_ea_df,
            "vwls": self.vwls_ea_df,
            "ewl": self.ewls_ea_long_df,
            "ews": self.ewls_ea_short_df,
            "vwl": self.vwls_ea_long_df,
            "vws": self.vwls_ea_short_df,
            "ewport": self.ewport_ea,
            "vwport": self.vwport_ea,
            "ew_turnover": self.ewturnover_ea_df,
            "vw_turnover": self.vwturnover_ea_df,
            "ew_chars": self.ew_chars_ea,
            "vw_chars": self.vw_chars_ea
        }

        # If filters are used, add ex post results
        if filters:
            self._stored_results.update({
                "ewls_ep": self.ewls_ep_df,
                "vwls_ep": self.vwls_ep_df,
                "ewl_ep": self.ewls_ep_long_df,
                "ews_ep": self.ewls_ep_short_df,
                "vwl_ep": self.vwls_ep_long_df,
                "vws_ep": self.vwls_ep_short_df,
                "ewport_ep": self.ewport_ep,
                "vwport_ep": self.vwport_ep,
                "ew_turnover_ep": self.ewturnover_ep_df,
                "vw_turnover_ep": self.vwturnover_ep_df,
                "ew_chars_ep": self.ew_chars_ep,
                "vw_chars_ep": self.vw_chars_ep
            })


    def fit(self, *, IDvar=None, DATEvar=None, RETvar=None, PRICEvar=None, RATINGvar = None ,Wvar = None):
        if any([IDvar, DATEvar, RETvar, PRICEvar, RATINGvar, Wvar]):
            column_mapping = {
            IDvar: "ID",
            DATEvar: "date",
            RETvar: "ret",
            PRICEvar: "PRICE",
            RATINGvar: "RATING_NUM",
            Wvar: "VW"
            }
            # Loop through the mapping and drop columns if they exist
            for var, col in column_mapping.items():
                if var and col in self.data.columns:
                    self.data.drop(columns=col, inplace=True)
                    self.data_raw.drop(columns=col, inplace=True)
                    warnings.warn(f"Column '{col}' already exists. It will be overwritten.", UserWarning)

            if Wvar is None and "VW" not in self.data.columns:
                self.data['VW'] = 1
                self.data_raw['VW'] = 1
                warnings.warn("Column 'VW' does not exist. Setting VW = 1 (i.e., equal weights)", UserWarning)

            self.rename_id(IDvar=IDvar, DATEvar=DATEvar, RETvar=RETvar, RATINGvar=RATINGvar, PRICEvar=PRICEvar, Wvar = Wvar)
        
        required_columns = ['index','date','ID', 'ret','RATING_NUM','VW']    
        # reindex to avoid issues with groupby
        self.data['index'] = np.arange(1, len(self.data) + 1, dtype=np.int64)
        self.data_raw['index'] = np.arange(1, len(self.data_raw) + 1, dtype=np.int64)
        
        # if self.rating:
        #     required_columns.append('RATING_NUM')        
        if self.adj == 'price':
            required_columns.append('PRICE')
        if self.chars:
            required_columns += self.chars 
            
        missing_columns = [col for col in required_columns if col not in self.data.columns]    
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")         
        
        # force the IDs to be numbers. Needed to facilitate storing results
        N = len(np.unique(self.data["ID"]))
        self.unique_bonds = N
        ID = dict(zip(np.unique(self.data["ID"]).tolist(),np.arange(1, N + 1, dtype=np.int64)))
        
        self.data["ID"] = self.data["ID"].map(ID)
        self.data_raw["ID"] = self.data_raw["ID"].map(ID)

        # sort by id and date
        self.data = self.data.sort_values(['ID','date'])
        self.data_raw = self.data_raw.sort_values(['ID','date'])

        # select relevant columns
        signal_vars = [self.strategy.get_sort_var()]
        if self.strategy.DoubleSort == 1:
            signal_vars.append(self.strategy.sort_var2)
        
        for signal_col in signal_vars:
            if signal_col in self.data.columns and signal_col not in required_columns:
                required_columns.append(signal_col)

        self.data = self.data[required_columns]
        self.data_raw = self.data_raw[required_columns]

        self.required_columns = required_columns
        
        self.compute_signal()
        self.portfolio_formation()

        # STORE RESULTS FOR IO OPERATIONS
        self._store_results(filters=self.filters)
        
        self.stored_params.update({"nbonds": N,
                                   "start_date": self.datelist[0],
                                   "end_date": self.datelist[-1],
                                   "tot_periods": len(self.datelist)})
        
        self.sfit = StrategyResults(self.stored_params,self._stored_results)

        return self
    
    def summary(self,type = "ew"):
        if hasattr(self, 'sfit'):
            return self.sfit.summary(type = type)
        else:
            raise ValueError("Fit the model first")
    
    def rename_id(self, *, IDvar=None, DATEvar=None,RETvar=None,RATINGvar=None, PRICEvar=None, Wvar = None):
        """
        rename columns to ensure consistency with col names
        """
        mapping = {}
        
        if IDvar:
            mapping[IDvar] = 'ID'
        if DATEvar:
            mapping[DATEvar] = 'date'
        if RETvar:
            mapping[RETvar] = 'ret'
        if RATINGvar:
            mapping[RATINGvar] = 'RATING_NUM'
        if PRICEvar:
            mapping[PRICEvar] = 'PRICE'
        if Wvar:
            mapping[Wvar] = 'VW'
            
        self.data.rename(columns=mapping, inplace=True)
        self.data_raw.rename(columns=mapping, inplace=True)
        
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'])
        if not pd.api.types.is_datetime64_any_dtype(self.data_raw['date']):
            self.data_raw['date'] = pd.to_datetime(self.data_raw['date'])
            
        self.datelist = pd.Series(self.data['date'].unique()).sort_values().tolist()
                     
    def compute_signal(self):
        # compute signal
        self.data = self.strategy.compute_signal(self.data) 
        # print(self.strategy.__strategy_name__)
        if self.filters and self.adj in ["trim", "wins", "price", "bounce"]:
            filter_obj = Filter(self.data, self.adj, self.w, self.loc, self.perc_breakpoints,self.price_threshold)
            self.name += filter_obj.name_filt
            self.data = filter_obj.apply_filters()
            sort_var = self.strategy.get_sort_var(self.adj)
            
            if self.adj == 'wins':
                # get the winsorized returns for ex post winsorization
                self.data_winsorized_ex_post = filter_obj.data_winsorized_ex_post# this is a df w/ wisorized returns
            
            if 'signal' in sort_var:
                if self.strategy.__strategy_name__ == "MOMENTUM":
                    # formation period of J months
                    J = self.strategy.J
                    skip = self.strategy.skip
                    varname = f'ret_{self.adj}'
                    self.data['logret'] = np.log(self.data[varname] + 1)
                    self.data[f'signal_{self.adj}'] = self.data.groupby(['ID'], group_keys=False)['logret']\
                        .rolling(J, min_periods=J).sum().values
                    self.data[f'signal_{self.adj}'] = np.exp(self.data[f'signal_{self.adj}']) - 1
                    self.data[f'signal_{self.adj}'] = self.data.groupby("ID")[f'signal_{self.adj}'].shift(skip)
                elif self.strategy.__strategy_name__ == "LT-REVERSAL":
                    # using "skip" as a formation period
                    J = self.strategy.J
                    skip = self.strategy.skip
                    varname = f'ret_{self.adj}'
                    self.data[f'signal_{self.adj}'] = self.data.groupby(['ID'], group_keys=False)[varname]\
                        .apply(lambda x:  x.rolling(window = J).sum()) - \
                               self.data.groupby(['ID'], group_keys=False)[varname]\
                        .apply(lambda x:  x.rolling(window = skip).sum())
        else:
            sort_var = self.strategy.get_sort_var()
            
                
    def portfolio_formation(self):
        unique_bonds = self.unique_bonds
        nport = self.nport
        adj = self.adj
        ret_var  = 'ret'                             # this is the column used to compute the returns of portfolios
        sort_var = self.strategy.get_sort_var(adj)   # this is the column used to sort bonds into portfolios
        hor = self.strategy.K                        # holding period
        
        TM = len(self.datelist)
        tab = self.data.copy().sort_values(['ID', 'date']) # make a copy of the data
        
        if adj == 'wins':
            tab_ex_post_wins = self.data_winsorized_ex_post # this is a df w/ wisorized returns
        
        # Unpack for double sorting
        use_double_sort = getattr(self.strategy, 'DoubleSort', False)
        how = getattr(self.strategy, 'how', 'unconditional') # unconditional by default
        if use_double_sort:
            # use_double_sort = 1
            nport2 = self.strategy.nport2
            sort_var2 = self.strategy.sort_var2
            tot_nport = nport * nport2
                      
        else:
            nport2 = None
            sort_var2 = None
            tot_nport = nport
        self.tot_nport = tot_nport 
        # initialize storing   
        ewport_hor_ea = np.full((TM, hor, tot_nport), np.nan)
        vwport_hor_ea = np.full((TM, hor, tot_nport), np.nan)
        # for turnover compute scaled returns
        if self.turnover:
            self.ewport_weight_hor_ea_scaled = np.zeros((TM, hor, tot_nport,unique_bonds))
            self.vwport_weight_hor_ea_scaled = np.zeros((TM, hor, tot_nport,unique_bonds))     
            # weights
            self.ewport_weight_hor_ea = np.zeros((TM, hor, tot_nport,unique_bonds))
            self.vwport_weight_hor_ea = np.zeros((TM, hor, tot_nport,unique_bonds))
        
        # storing for chars of bins
        if self.chars:
            ew_ea_chars_dict = {}
            vw_ea_chars_dict = {}
            for char in self.chars:
                ew_ea_chars_dict[char] = np.full((TM, hor, tot_nport), np.nan)
                vw_ea_chars_dict[char] = np.full((TM, hor, tot_nport), np.nan)
        
        # banding: we need to save the ranks
        if self.banding_threshold is not None:
            # create a dictionary to store the ranks. This is used to compute the banding
            # the key of the dictionary is the cohort
            self.lag_rank = {i: pd.DataFrame for i in range(hor)}
            
        if adj:
            ewport_hor_ep = np.full((TM, hor, tot_nport), np.nan)
            vwport_hor_ep = np.full((TM, hor, tot_nport), np.nan) 
            # for turnover compute scaled returns
            if self.turnover:
                self.ewport_weight_hor_ep_scaled = np.zeros((TM, hor, tot_nport,unique_bonds))
                self.vwport_weight_hor_ep_scaled = np.zeros((TM, hor, tot_nport,unique_bonds))       
                # weights
                self.ewport_weight_hor_ep = np.zeros((TM, hor, tot_nport,unique_bonds))
                self.vwport_weight_hor_ep = np.zeros((TM, hor, tot_nport,unique_bonds))
            
            if self.chars:
                ew_ep_chars_dict = {}
                vw_ep_chars_dict = {}
                for char in self.chars:
                    ew_ep_chars_dict[char] = np.full((TM, hor, tot_nport), np.nan)
                    vw_ep_chars_dict[char] = np.full((TM, hor, tot_nport), np.nan)
        
        # for t in range((hor+1), TM - hor): to discuss this!
        for t in range( TM):
            # define cohort for turnover computation
            self.cohort = t % hor
            
            # print(self.cohort,h)
            date_t = self.datelist[t]

            # Filter based on ratings and signal != nan
            if use_double_sort:
                It0 = self.filter_by_rating(tab, date_t, sort_var, sort_var2)
            else:
                It0 = self.filter_by_rating(tab, date_t, sort_var)              
            
            # check if at time t we have bonds
            if It0.shape[0] == 0:
                if t > hor:
                    print(f"no bonds at time {t}:{date_t}. Going to next period.")      
                continue
            
            # Investment Universe matching
            if adj in ['trim', 'bounce', 'price']:
                It0 = self.filter_by_universe_matching(It0, adj, ret_var)
                # check if after the filter we have bonds
                if It0.shape[0] == 0:
                    if t > hor:
                        print(f"no bonds at time {t}: {date_t} after adjustment ({adj}). Going to next period.")      
                    continue

            
            # start sorting procedure
            for h in range(1, hor + 1):
                if t + h > TM -1:
                    continue
                # print(t,self.cohort+1,h)
                It0_h = It0.copy()
                # Investment universe for ret computation
                It1 = self.data_raw[(self.data_raw['date'] == self.datelist[t + h])& (~self.data_raw[ret_var].isna())]
                # Dynamically get the mv for different horizons
                It1m = tab[(tab['date'] == self.datelist[t + h - 1]) & (~tab['VW'].isna())]   
                
                if adj == 'wins' and 'signal' in sort_var:
                    # TODO can be removed
                    # use winsorized returns to assign to ptfs
                    port_ret_ea = self.port_sorted_ret(It0_h, It1,It1m,ret_var, sort_var,DoubleSort=use_double_sort,sig2 = sort_var2,nport2 = nport2, how = how )
                else:
                    # if signal is not in sort_var, we do not use winsorized returns to sort portfolios
                    port_ret_ea = self.port_sorted_ret(It0_h, It1,It1m,ret_var, sort_var,DoubleSort=use_double_sort,sig2 = sort_var2,nport2 = nport2, how = how )
                
                # unpack returns
                ret_strategy_ea =  port_ret_ea[0]

                # storing returns
                ewport_hor_ea[t + h, self.cohort, :] = ret_strategy_ea[0]
                vwport_hor_ea[t + h, self.cohort, :] = ret_strategy_ea[1]
                # storing weights
                if self.turnover:
                    weights_ea, weights_scaled_ea = port_ret_ea[1]
                    if not weights_ea.empty:
                        self.fill_weights(weights_ea, self.ewport_weight_hor_ea,self.vwport_weight_hor_ea, t, h,self.cohort)
                        self.fill_weights(weights_scaled_ea, self.ewport_weight_hor_ea_scaled,self.vwport_weight_hor_ea_scaled, t, h,self.cohort)
                # storing chars
                if self.chars:
                    # unpack
                    chars_ea = port_ret_ea[2]
                    for idx, c in enumerate(self.chars):
                        # store 
                        ew_ea_chars_dict[c][t + h,h-1,:] = chars_ea[0][c]
                        vw_ea_chars_dict[c][t + h,h-1,:] = chars_ea[1][c]
            
                if adj:
                    if adj == 'wins':
                        It2 = tab_ex_post_wins[(tab_ex_post_wins['date'] == self.datelist[t + h]) & (~tab_ex_post_wins[ret_var].isna())]
                    else:
                        It2 = tab[(tab['date'] == self.datelist[t + h]) & (~tab[ret_var + "_" + adj].isna())]
                    
                    port_ret_ep = self.port_sorted_ret(It0_h, It2,It1m,ret_var + "_" + adj, sort_var,DoubleSort=use_double_sort,sig2 = sort_var2,nport2 = nport2,how = how)
                    # unpack returns
                    ret_strategy_ep =  port_ret_ep[0]
                    # storing returns
                    ewport_hor_ep[t + h, h - 1, :] = ret_strategy_ep[0]
                    vwport_hor_ep[t + h, h - 1, :] = ret_strategy_ep[1]
                    # storing weights: 
                    if self.turnover:
                        weights_ep, weights_scaled_ep = port_ret_ep[1]
                        if not weights_ep.empty:
                            self.fill_weights(weights_ep, self.ewport_weight_hor_ep,self.vwport_weight_hor_ep, t, h,self.cohort)
                            self.fill_weights(weights_scaled_ep, self.ewport_weight_hor_ep_scaled,self.vwport_weight_hor_ep_scaled, t, h,self.cohort)
                        
                    # storing chars
                    if self.chars:
                        # unpack
                        chars_ep = port_ret_ep[2]
                        for idx, c in enumerate(self.chars):
                            # store 
                            ew_ep_chars_dict[c][t + h,h-1,:] = chars_ep[0][c]
                            vw_ep_chars_dict[c][t + h,h-1,:] = chars_ep[1][c]                 
                        
        self.ewport_ea = np.mean(ewport_hor_ea, axis=1)
        self.vwport_ea = np.mean(vwport_hor_ea, axis=1)
        if adj:
            self.ewport_ep = np.mean(ewport_hor_ep, axis=1)
            self.vwport_ep = np.mean(vwport_hor_ep, axis=1)    

        # Compute portfolio returns
        if use_double_sort:  
            # create column names for dfs
            colnames_ptf_df = [f'{self.signal_names[0]}{i}_{self.signal_names[1]}{j}' for i in range(1, nport + 1) for j in range(1, nport2 + 1)]
            # storing rets
            avg_res_ew = []
            avg_res_vw = []
            
            # storing legs
            long_leg_ew_ea   = []
            short_leg_ew_ea  = []
            long_leg_vw_ea   = []
            short_leg_vw_ea  = []            
            
            idx = self.compute_idx(nport, nport2)
            for i in range(1, nport + 1):
                col_idx_n2 = idx[(i, nport2)]
                col_idx_1 = idx[(i, 1)]

                avg_res_ew.append(self.ewport_ea[:,col_idx_n2-1]- (1 * self.ewport_ea[:,col_idx_1-1]))
                avg_res_vw.append(self.vwport_ea[:,col_idx_n2-1]- (1 * self.vwport_ea[:,col_idx_1-1]))
                
                long_leg_ew_ea.append(self.ewport_ea[:,col_idx_n2-1])
                short_leg_ew_ea.append(self.ewport_ea[:,col_idx_1-1])
                
                long_leg_vw_ea.append(self.vwport_ea[:,col_idx_n2-1])
                short_leg_vw_ea.append(self.vwport_ea[:,col_idx_1-1])                
                
            avg_res_ew = np.vstack(avg_res_ew).T
            avg_res_vw = np.vstack(avg_res_vw).T
            
            self.ewls_ea = np.mean(avg_res_ew,axis=1)
            self.vwls_ea = np.mean(avg_res_vw,axis=1)  
            
            self.ewls_ea_long_df = pd.DataFrame(np.vstack(long_leg_ew_ea).T,index = self.datelist, columns = [[str(x)+'_LONG_EWEA_' + self.name for x in range(1,nport+1)]]) 
            self.vwls_ea_long_df = pd.DataFrame(np.vstack(long_leg_vw_ea).T,index = self.datelist, columns = [[str(x)+'_LONG_VWEA_' + self.name for x in range(1,nport+1)]])
            # computing short leg df 
            self.ewls_ea_short_df = pd.DataFrame(np.vstack(short_leg_ew_ea).T,index = self.datelist, columns = [[str(x)+'_SHORT_EWEA_' + self.name for x in range(1,nport+1)]])
            self.vwls_ea_short_df = pd.DataFrame(np.vstack(short_leg_vw_ea).T,index = self.datelist, columns = [[str(x)+'_SHORT_VWEA_' + self.name for x in range(1,nport+1)]])
        else:           
            # create column names for dfs
            colnames_ptf_df = [f'{self.signal_names[0]}{i}' for i in range(1, nport + 1)]
            # computing long leg
            EWlong_leg_ea = self.ewport_ea[:, tot_nport - 1]
            VWlong_leg_ea = self.vwport_ea[:, tot_nport - 1]
            # computing short leg            
            EWshort_leg_ea = self.ewport_ea[:, 0]
            VWshort_leg_ea = self.vwport_ea[:, 0]          
            # Long short portfolio
            self.vwls_ea = self.vwport_ea[:, tot_nport - 1] - self.vwport_ea[:, 0]
            self.ewls_ea = self.ewport_ea[:, tot_nport - 1] - self.ewport_ea[:, 0]
            # computing long leg df
            self.ewls_ea_long_df = pd.DataFrame(EWlong_leg_ea,index = self.datelist, columns = ['LONG_EWEA_' + self.name]) 
            self.vwls_ea_long_df = pd.DataFrame(VWlong_leg_ea,index = self.datelist, columns = ['LONG_VWEA_' + self.name]) 
            # computing short leg df 
            self.ewls_ea_short_df = pd.DataFrame(EWshort_leg_ea,index = self.datelist, columns = ['SHORT_EWEA_' + self.name]) 
            self.vwls_ea_short_df = pd.DataFrame(VWshort_leg_ea,index = self.datelist, columns = ['SHORT_VWEA_' + self.name])  
      
        self.ewls_ea_df = pd.DataFrame(self.ewls_ea,index = self.datelist, columns = ['EWEA_' + self.name]) 
        self.vwls_ea_df = pd.DataFrame(self.vwls_ea,index = self.datelist, columns = ['VWEA_' + self.name]) 
        # create dfs for all portfolios
        self.ewport_ea_df = pd.DataFrame(self.ewport_ea, index = self.datelist, columns = colnames_ptf_df)
        self.vwport_ea_df = pd.DataFrame(self.vwport_ea, index = self.datelist, columns = colnames_ptf_df)

        if self.turnover:
            ew_port_turn_ea = self.compute_turnover(self.ewport_weight_hor_ea,self.ewport_weight_hor_ea_scaled)
            vw_port_turn_ea = self.compute_turnover(self.vwport_weight_hor_ea,self.vwport_weight_hor_ea_scaled)
        
            self.ewturnover_ea_df = pd.DataFrame(ew_port_turn_ea,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,tot_nport+1)])
            self.vwturnover_ea_df = pd.DataFrame(vw_port_turn_ea,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,tot_nport+1)])
        
        # computing chars stats
        if self.chars:
            self.ew_chars_ea = {}
            self.vw_chars_ea = {}
            for c in self.chars:
                self.ew_chars_ea[c] = pd.DataFrame(np.mean(ew_ea_chars_dict[c],axis=1),index = self.datelist,columns = [f"Q{x}" for x in range(1,tot_nport+1)]) # mean across horizon
                self.vw_chars_ea[c] = pd.DataFrame(np.mean(vw_ea_chars_dict[c],axis=1),index = self.datelist,columns = [f"Q{x}" for x in range(1,tot_nport+1)]) # mean across horizon
        
        if adj:
            if use_double_sort:
                avg_res_ew = []
                avg_res_vw = []
                # storing legs
                long_leg_ew_ep   = []
                short_leg_ew_ep  = []
                long_leg_vw_ep   = []
                short_leg_vw_ep  = []            
                
                idx = self.compute_idx(nport, nport2)
                for i in range(1, nport + 1):
                    col_idx_n2 = idx[(i, nport2)]
                    col_idx_1 = idx[(i, 1)]

                    avg_res_ew.append(self.ewport_ep[:,col_idx_n2-1]- (1 * self.ewport_ep[:,col_idx_1-1]))
                    avg_res_vw.append(self.vwport_ep[:,col_idx_n2-1]- (1 * self.vwport_ep[:,col_idx_1-1]))    
                    
                    long_leg_ew_ep.append(self.ewport_ep[:,col_idx_n2-1])
                    short_leg_ew_ep.append(self.ewport_ep[:,col_idx_1-1])
                    
                    long_leg_vw_ep.append(self.vwport_ep[:,col_idx_n2-1])
                    short_leg_vw_ep.append(self.vwport_ep[:,col_idx_1-1]) 
                    
                avg_res_ew = np.vstack(avg_res_ew).T
                avg_res_vw = np.vstack(avg_res_vw).T
            
                self.ewls_ep = np.mean(avg_res_ew,axis=1)
                self.vwls_ep = np.mean(avg_res_vw,axis=1)  
                
                self.ewls_ep_long_df = pd.DataFrame(np.vstack(long_leg_ew_ep).T,index = self.datelist, columns = [[str(x)+'_LONG_EWEP_' + self.name for x in range(1,nport+1)]]) 
                self.vwls_ep_long_df = pd.DataFrame(np.vstack(long_leg_vw_ep).T,index = self.datelist, columns = [[str(x)+'_LONG_VWEP_' + self.name for x in range(1,nport+1)]])
                # computing short leg df 
                self.ewls_ep_short_df = pd.DataFrame(np.vstack(short_leg_ew_ep).T,index = self.datelist, columns = [[str(x)+'_SHORT_EWEP_' + self.name for x in range(1,nport+1)]])
                self.vwls_ep_short_df = pd.DataFrame(np.vstack(short_leg_vw_ep).T,index = self.datelist, columns = [[str(x)+'_SHORT_VWEP_' + self.name for x in range(1,nport+1)]])
           
            else:      
                # Long short portfolio
                self.ewls_ep = self.ewport_ep[:, tot_nport - 1] - self.ewport_ep[:, 0]
                self.vwls_ep = self.vwport_ep[:, tot_nport - 1] - self.vwport_ep[:, 0]
                
                # computing long leg
                EWlong_leg_ep = self.ewport_ep[:, tot_nport - 1]
                VWlong_leg_ep = self.vwport_ep[:, tot_nport - 1]
                
                # computing short leg            
                EWshort_leg_ep = self.ewport_ep[:, 0]
                VWshort_leg_ep = self.vwport_ep[:, 0]    
                # computing long leg df      
                self.ewls_ep_long_df = pd.DataFrame(EWlong_leg_ep,index = self.datelist, columns = ['LONG_EWEP_' + self.name]) 
                self.vwls_ep_long_df = pd.DataFrame(VWlong_leg_ep,index = self.datelist, columns = ['LONG_VWEP_' + self.name]) 
                # computing short leg df
                self.ewls_ep_short_df = pd.DataFrame(EWshort_leg_ep,index = self.datelist, columns = ['SHORT_EWEP_' + self.name]) 
                self.vwls_ep_short_df = pd.DataFrame(VWshort_leg_ep,index = self.datelist, columns = ['SHORT_VWEP_' + self.name])  
                                
            self.ewls_ep_df = pd.DataFrame(self.ewls_ep,index = self.datelist,columns = ['EWEP_' + self.name])   
            self.vwls_ep_df = pd.DataFrame(self.vwls_ep,index = self.datelist,columns = ['VWEP_' + self.name])  
            # create dfs for all portfolios
            colnames_ptf_df_ep = [f"{col}_ep" for col in colnames_ptf_df]
            self.ewport_ep_df = pd.DataFrame(self.ewport_ep, index = self.datelist, columns = colnames_ptf_df_ep)
            self.vwport_ep_df = pd.DataFrame(self.vwport_ep, index = self.datelist, columns = colnames_ptf_df_ep)
            
            if self.turnover:
                ew_port_turn_ep = self.compute_turnover(self.ewport_weight_hor_ep,self.ewport_weight_hor_ep_scaled)
                vw_port_turn_ep = self.compute_turnover(self.vwport_weight_hor_ep,self.vwport_weight_hor_ep_scaled)
            
                self.ewturnover_ep_df = pd.DataFrame(ew_port_turn_ep,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,tot_nport+1)])
                self.vwturnover_ep_df = pd.DataFrame(vw_port_turn_ep,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,tot_nport+1)])
            
            #characteristics
            if self.chars:
                self.ew_chars_ep = {}
                self.vw_chars_ep = {}
                for c in self.chars:
                    self.ew_chars_ep[c] = pd.DataFrame(np.mean(ew_ep_chars_dict[c],axis=1),index = self.datelist,columns = [f"Q{x}" for x in range(1,tot_nport+1)]) # mean across horizon
                    self.vw_chars_ep[c] = pd.DataFrame(np.mean(vw_ep_chars_dict[c],axis=1),index = self.datelist,columns = [f"Q{x}" for x in range(1,tot_nport+1)]) # mean across horizon

    def port_sorted_ret(self, It0, It1, It1m, ret_col,sig,**kwargs):
        """
        It0: investment universe that is going to be sorted in portfolios
        It1: investment universe at t+h. Used to compute returns on ptfs
        ret_col: col of returns used to compute the ptfs returns
        sig: variable used to sort assets
        -------------
        Optional:
            perform (unconditional) double sorting 
        """
        # =====================================================================
        # Unpacking if double sorting
        # =====================================================================
        double_sort = kwargs.get('DoubleSort', None)
        sig2        = kwargs.get('sig2', None)
        nport2      = kwargs.get('nport2', None)
        how         = kwargs.get('how', 'unconditional')
        
        # =====================================================================
        # compute edges for first and second signals
        # =====================================================================
        time_t = It0["date"].iloc[0] if not It0.empty else "no bonds"
        time_t1 = It1["date"].iloc[0] if not It1.empty else "no bonds"
        nport = self.nport    # number of portfolios
        thres = np.percentile(It0[sig], np.linspace(0, 100, nport + 1))        # compute edges for signal
        thres[0] = -np.inf
        
        id0 = It0['ID']
        id1 = It1['ID']
        id2 = It1m['ID']
        
        intersect_ids  = id0[id0.isin(id1)]                     # i1
        intersect_idsm = intersect_ids[intersect_ids.isin(id2)] # i1m
        
        if self.dynamic_weights:
        # This to go in the if dynamic weights #
            It0  = It0[id0.isin(intersect_idsm)].copy()
            It1  = It1[id1.isin(intersect_idsm)].copy() 
            It1m = It1m[id2.isin(intersect_idsm)].copy()
            It1['VW'] = It1m['VW'].values    
        else:
            It0 = It0[id0.isin(intersect_ids)].copy()
            It1 = It1[id1.isin(intersect_ids)].copy() 
            It1['VW'] = It0['VW'].values # Adjust this!
    
        sortvar = It0[sig]
        # =====================================================================
        # Rank bonds based on signals
        # =====================================================================
        if double_sort:
            nportmax = nport * nport2 
            sortvar2 = It0[sig2]
            
            idx1 = self.assign_bond_bins(sortvar,thres,nport)     # sort first signal
            
            if how == "unconditional":
                # unconditional double sorting: compute the rank independently for the two signals
                thres2 = np.percentile(It0[sig2], np.linspace(0, 100, nport2 + 1))# compute edges for signal2 independently from signal1
                thres2[0] = -np.inf
                idx2 = self.assign_bond_bins(sortvar2,thres2,nport2)  # sort second signal independently    
                # create a column with final rank: ptfs going from 1 to nport1 x nport2
                It1['ptf_rank'] = self.double_sort_uncond(idx1, idx2, nport, nport2)

            if how == "conditional":
                # conditional double sorting: within each bin, sort the bonds based on the second signal
                It1['ptf_rank'] = self.double_sort_cond(sortvar2, idx1, nport, nport2)
                    
        else:
            nportmax = nport
            It1['ptf_rank'] = self.assign_bond_bins(sortvar,thres,nport)
        
        # store It1 if banding 
        if self.banding_threshold is not None:
            # we rebalance only at t so when h = 1!
            if self.lag_rank[self.cohort].empty:
                # if first period, just save the ranks
                self.lag_rank[self.cohort] = It1[['ID','ptf_rank']].copy()
            else:
                # if not first period, merge the ranks                
                # rank banding
                
                    # get the ranks from the previous period
                prev_rank = self.lag_rank[self.cohort]

                It1 = It1.merge( prev_rank, how = "left", left_on  = ['ID'],
                                right_on = ['ID'], suffixes = ('_current','_lag'))

                It1["ptf_rank"] = self.calculate_qnew_vectorized(It1['ptf_rank_lag'],It1['ptf_rank_current'],nportmax,self.banding_threshold)
                self.lag_rank[self.cohort] = It1[['ID','ptf_rank']].copy()

        # check if dfs are empty
        if It0.shape[0] == 0:
            print(f"no bonds matched between time {time_t} and {time_t1}. Setting return to nan and going to next period.")   
            nan_list = [np.nan] * nportmax
            if self.chars:
                nan_df = pd.DataFrame(np.full((nportmax,len(self.chars)),np.nan),columns=self.chars)
                return (nan_list, nan_list),(pd.DataFrame(),pd.DataFrame()), (nan_df,nan_df)
            else:
                return (nan_list, nan_list),(pd.DataFrame(),pd.DataFrame())
            
                               
        It1['weights'] = It1.groupby('ptf_rank')['VW'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)
        
        # It1[ret_col] = It1[ret_col].fillna(0)
        ptf_ret_ew = It1.groupby('ptf_rank')[ret_col].mean()
        ptf_ret_vw = It1.groupby('ptf_rank').apply(lambda x: (x[ret_col] * x['weights']).sum())

        nport_idx = range(1,int(nportmax+1))  # index for the number of portfolios

        # store the weights:  return the column with ID 
        if self.turnover:
            rank_ = It1[['ID','ptf_rank','ret']]
            rank = rank_.copy()
            rank['count'] = rank.groupby('ptf_rank')['ID'].transform('count')
            rank['eweights'] = 1 / rank['count']
            rank = rank.merge(It1[['ID', 'weights']], on='ID')
            rank = rank.rename(columns={"weights":"vweights"})
            _weights = rank[['ID','ptf_rank','eweights','vweights']]
            
            # scale returns
            retscaled = rank.copy()

            retscaled = retscaled.merge(ptf_ret_ew.to_frame(name='ewret').reset_index(), on ="ptf_rank",how="left")
            retscaled = retscaled.merge(ptf_ret_ew.to_frame(name='vwret').reset_index(), on ="ptf_rank",how="left")    
            
            retscaled["ewret_scaled"] = ((1 + retscaled['ret'])/ (1+retscaled['ewret']))/retscaled["count"]
            retscaled["vwret_scaled"] = ((1 + retscaled['ret'])/ (1+retscaled['vwret'])) * retscaled["vweights"]
            _weights_scaled = retscaled[['ID','ptf_rank','ewret_scaled','vwret_scaled']]
            _weights_scaled = _weights_scaled.rename(columns={'ewret_scaled':'eweights','vwret_scaled':'vweights'})
         
        # reindex         
        
        ptf_ret_ew = ptf_ret_ew.reindex(nport_idx)
        ptf_ret_vw = ptf_ret_vw.reindex(nport_idx)
       
        ewl = ptf_ret_ew.to_list()
        vwl = ptf_ret_vw.to_list()
        
        # =====================================================================
        # store the chars:  return the column with ID 
        # =====================================================================  
        if self.chars:
            nm = ['ID','ptf_rank','weights']
            # merge weights and ptf rank
            sub = It1[nm]
            It1m = It1m.merge(sub,on="ID")
            
            # augment with chars
            nm += self.chars
            
            chars = It1m[nm]
            ew_chars = pd.DataFrame()
            vw_chars = pd.DataFrame()
            
            for e, c in enumerate(self.chars):
                c_ew = chars.groupby('ptf_rank')[c].mean()
                c_vw = chars.groupby('ptf_rank').apply(lambda x: (x[c] * x['weights']).sum())
                # reindex to avoid problems when one of the bins is empty
                c_ew = c_ew.reindex(nport_idx)
                c_vw = c_vw.reindex(nport_idx)

                ew_chars = pd.concat([ew_chars,c_ew],axis=1)
                vw_chars = pd.concat([vw_chars,c_vw],axis=1)
            
            vw_chars.columns = ew_chars.columns
            if self.turnover:
                return (ewl, vwl),(_weights,_weights_scaled), (ew_chars,vw_chars)
            else:
                return (ewl, vwl),(None,None), (ew_chars,vw_chars)
        else:
            if self.turnover:
                return (ewl, vwl),(_weights,_weights_scaled)
            else:
                return (ewl, vwl),(None,None)
    
    # Over-write Q #
    # def calculate_qnew(self,row, nport, banding_thres):                         
    #     # Check if the drop is greater than the threshold
    #     if row['ptf_rank_lag'] == nport and row['ptf_rank_current'] >= nport - banding_thres:
    #         return nport
    #     # Check if the rise is greater than the threshold
    #     elif row['ptf_rank_lag'] == 1 and row['ptf_rank_current']   <= 1     + banding_thres:
    #         return 1
    #     else:
    #         return row['ptf_rank_current']  
        
    @staticmethod    
    def calculate_qnew_vectorized(lag,current, nport, banding_thres):
        # Create masks for the conditions
        mask_drop = (lag == nport) & (current >= nport - banding_thres)
        mask_rise = (lag == 1) & (current <= 1 + banding_thres)

        new_rank = np.where(mask_drop, nport, np.where(mask_rise, 1,current))

        return new_rank

    @staticmethod
    def assign_bond_bins(sortvar,thres,nport):
        idx = np.full(sortvar.shape, np.nan)
        for p in range(nport):
            f = (sortvar > thres[p]) & (sortvar <= thres[p + 1])
            idx[f] = p + 1
        return idx
    
    @staticmethod
    def intersect_col(i, j, n2):
        return (i - 1) * n2 + j
    
    @staticmethod
    def compute_idx(n1, n2):
        result = {}
        for i in range(1, n1 + 1):
            result[(i, n2)] = StrategyFormation.intersect_col(i, n2, n2)
            result[(i, 1)] = StrategyFormation.intersect_col(i, 1, n2)
        return result
            
    @staticmethod
    def double_sort_uncond(idx1, idx2, n1, n2):
        """
        Python adaptation of AssayingAnomalies func 
        https://github.com/velikov-mihail/AssayingAnomalies/tree/main
        """
        idx = np.full(idx1.shape,np.nan)
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                idx[(idx1 == i) & (idx2 == j)] = (i - 1) * n2 + j
        return idx
     
    @staticmethod
    def double_sort_cond(sortvar2, idx1, n1, n2):
        n2 = int(n2)
        n1 = int(n1)
        idx = np.zeros_like(sortvar2) # initialize the array 
        # loop over the number of portfolios and assign the rank within sorted bins
        for i in range(1, n1 + 1):
            temp = sortvar2[idx1 == i] # get sortvar2 values for stocks in bin i
            # if temp is empty then skip. This means that there are no assets in bin i
            if len(temp) == 0:
                continue
            # sort values within the bin i
            thres2 = np.percentile(temp, np.linspace(0, 100, n2 + 1))
            thres2[0] = -np.inf
            # assign a rank based on the break points in thres2
            id2 = StrategyFormation.assign_bond_bins(temp, thres2, n2)
            nmax = np.max(id2)  # get the maximum rank just for checking
            # assign bonds to idx from 1 to n1*n2
            idx[idx1 == i] = id2 + nmax * (i - 1)
        return idx
    
    @staticmethod
    def fill_weights(weights, array_ew,array_vw, t, h,cohort):
        p = weights['ptf_rank'].values.astype(int)
        ID = weights['ID'].values.astype(int)
        eweights = weights['eweights'].values
        vweights = weights['vweights'].values
            
        array_ew[t + h - 1, cohort, p - 1, ID - 1] = eweights
        array_vw[t + h - 1, cohort, p - 1, ID - 1] = vweights
            
    @staticmethod  
    def compute_turnover(w,w_scaled):
        #  Portfolio weight changes: Subtract return-adjusted weights (lagged) from weights;
        abs_dewport_weight = np.abs(w[1:,:,:,:] - w_scaled[:-1,:,:,:])
        port_turn_hor    = np.sum(abs_dewport_weight, axis=3)
        # Set any 0 values to np.NaN
        # port_turn_hor[port_turn_hor == 0] = np.nan           # Alex added 23-07-2024 #
        port_turn_hor = np.where(port_turn_hor == 0, np.nan, port_turn_hor) # this is slightly faster than the above line

        mean_port_turn_hor = np.mean(port_turn_hor, axis=1)  # Alex changed to np.mean 23-07-2024 #
        port_turn = np.squeeze(mean_port_turn_hor)
        return port_turn
    
    def filter_by_rating(self, tab, date, sort_var, sort_var2=None):
        # Basic filtering conditions
        conditions = (tab['date'] == date) & (~tab[sort_var].isna())

        # Add additional conditions based on the rating
        if self.rating == "NIG":
            conditions &= (tab['RATING_NUM'] > 10) & (tab['RATING_NUM'] <= 22)
        elif self.rating == "IG":
            conditions &= (tab['RATING_NUM'] >= 1) & (tab['RATING_NUM'] <= 10)

        # Add conditions for the second sort variable if use_double_sort is True
        if sort_var2 is not None:
            conditions &= ~tab[sort_var2].isna()

        return tab[conditions]
    
    def filter_by_universe_matching(self, It0, adj, ret_var):
        if adj == 'trim' and self.w:
            if isinstance(self.w, list) and len(self.w) == 2:
                lower_bound, upper_bound = self.w
                It0 = It0[(It0[ret_var] <= upper_bound) & (It0[ret_var] >= lower_bound)]
            else:
                It0 = It0[It0[ret_var] <= self.w] if self.w > 0 else It0[It0[ret_var] >= self.w]

        elif adj == 'bounce' and self.w:
            if isinstance(self.w, list) and len(self.w) == 2:
                lower_bound, upper_bound = self.w
                It0 = It0[(It0['bounce'] <= upper_bound) & (It0['bounce'] >= lower_bound)]
            else:
                It0 = It0[It0['bounce'] >= self.w] if self.w < 0 else It0[It0['bounce'] <= self.w]

        elif adj == 'price' and self.w:
            if isinstance(self.w, list) and len(self.w) == 2:
                lower_bound, upper_bound = self.w
                It0 = It0[(It0['PRICE'] <= upper_bound) & (It0['PRICE'] >= lower_bound)]
            else:
                It0 = It0[It0['PRICE'] <= self.w] if self.w > self.price_threshold else It0[It0['PRICE'] >= self.w]
                
        return It0

    
    # helper function for getters
    def _get_dataframes(self, *keys, **kwargs):
        checks = {
            'require_filters': lambda: self.filters,
            'require_turnover': lambda: self.turnover,
            'require_chars': lambda: self.chars,
        }
        
        for check, condition in checks.items():
            if kwargs.get(check, False) and not condition():
                warnings.warn(f"{check.replace('require_', '').capitalize()} are not specified.", UserWarning)
                return None

        dfs = [getattr(self, key) for key in keys]
        if any(df is None for df in dfs):
            warnings.warn("One or more DataFrames have not been initialized.", UserWarning)
            return None
        return dfs
    
    def _get_signal_names(self):
        sig1= self.strategy.get_sort_var().upper()

        if self.strategy.DoubleSort:
            sig2 = self.strategy.get_sort_var2().upper()
            return [sig1, sig2]
        return [sig1]

    # getters for the results
    def get_long_leg(self):
        return self._get_dataframes("ewls_ea_long_df", "vwls_ea_long_df")

    def get_long_leg_ex_post(self):
        return self._get_dataframes("ewls_ep_long_df", "vwls_ep_long_df", require_filters=True)

    def get_short_leg(self):
        return self._get_dataframes("ewls_ea_short_df", "vwls_ea_short_df")

    def get_short_leg_ex_post(self):
        return self._get_dataframes("ewls_ep_short_df", "vwls_ep_short_df", require_filters=True)

    def get_long_short(self):
        return self._get_dataframes("ewls_ea_df", "vwls_ea_df")

    def get_long_short_ex_post(self):
        return self._get_dataframes("ewls_ep_df", "vwls_ep_df", require_filters=True)

    def get_ptf(self):
        return self._get_dataframes("ewport_ea_df", "vwport_ea_df")

    def get_ptf_ex_post(self):
        return self._get_dataframes("ewport_ep", "vwport_ep", require_filters=True)

    def get_ptf_weights(self):
        return self._get_dataframes("ewport_weight_hor_ea", "vwport_weight_hor_ea")

    def get_ptf_weights_ex_post(self):
        return self._get_dataframes("ewport_weight_hor_ep", "vwport_weight_hor_ep", require_filters=True)

    def get_ptf_turnover(self):
        return self._get_dataframes("ewturnover_ea_df", "vwturnover_ea_df", require_turnover=True)

    def get_ptf_turnover_ex_post(self):
        return self._get_dataframes("ewturnover_ep_df", "vwturnover_ep_df", require_filters=True, require_turnover=True)

    def get_chars(self):
        return self._get_dataframes("ew_chars_ea", "vw_chars_ea", require_chars=True)

    def get_chars_ex_post(self):
        return self._get_dataframes("ew_chars_ep", "vw_chars_ep", require_filters=True, require_chars=True)
         
    def set_factors(self,factors):
        """
        set K factors
        
        """
        # check consistency for date column
        if 'date' in factors.columns:
            if not pd.api.types.is_datetime64_any_dtype(factors['date']):
                factors['date'] = pd.to_datetime(factors['date'])
            factors.set_index('date',inplace = True)
        else:
            raise ValueError("Missing required date. If you set date as index, please reset_index()")
        self.factors = factors
            
    def get_alphas(self,nw_lag = 0):           
        # check consistency of dates between the long-short ptf and factors
        anom    = pd.concat(self.get_long_short(),axis = 1) 
        if self.ewls_ep_df is not None:
            anom_ep = pd.concat(self.get_long_short_ex_post(),axis = 1) 
            anom = pd.concat([anom, anom_ep],axis =1 )
              
        idx = anom.index.intersection(self.factors.index)
        
        factors = self.factors.loc[idx]
        anom    = anom.loc[idx]
        
        # get alpha
        res = pd.DataFrame(np.zeros((6,anom.shape[1])),columns = anom.columns,index=['mean','tval_m','p-val_m','alpha','t-val_a','p-val_a'])
        for i in range(anom.shape[1]):
            mod1 = sm.OLS(np.array(anom)[:,i],np.ones_like(np.array(anom)[:,i]),missing='drop').fit(cov_type='HAC',
                             cov_kwds={'maxlags': nw_lag})
            mod2 = sm.OLS(np.array(anom)[:,i],sm.add_constant(np.array(factors)),missing='drop').fit(cov_type='HAC',
                             cov_kwds={'maxlags': nw_lag})
            
            res.iloc[0,i] = mod1.params[0]
            res.iloc[1,i] = mod1.tvalues[0]
            res.iloc[2,i] = mod1.pvalues[0]            
            
            res.iloc[3,i] = mod2.params[0]
            res.iloc[4,i] = mod2.tvalues[0]
            res.iloc[5,i] = mod2.pvalues[0]
            
        return res
    
    def plot(self,vw = False, ax=None):
        # plot cumulative returns

        if ax is None:
            fig, ax = plt.subplots()
        # get the long short ptf 
        ew_,vw_ = self.get_long_short()
        lab = 'ex ante'
        if self.ewls_ep_df is not None:
            ewp_, vwp_ = self.get_long_short_ex_post()
            ew_ = pd.concat([ew_,ewp_],axis = 1)
            vw_ = pd.concat([vw_,vwp_],axis = 1)
            lab = [lab, "ex post"]

        # plotting ew or vw
        v_ = vw_ if vw else ew_

        var_plot = (v_ + 1).cumprod()
        
        # fig = figure()
        # ax = fig.add_subplot(111)
        ax.plot(var_plot.index,var_plot ,label = lab)

        ax.set_ylabel('Value ($)')
        ax.set_xlabel('Date')

        title = "Value-weighted cumulative performance" if vw else "Equally-weighted cumulative performance"
        ax.set_title(title)
        ax.legend()

        return ax 

    def stats_bonds_adj(self):
        """
        Get the bonds filtered out and compute statistics for specified characteristics.

        Returns:
        pd.DataFrame: DataFrame containing the computed statistics.
        """
        if self.chars is None:
            raise ValueError("Please include 'chars' in the parameters.")

        adj = self.adj
        if f"ret_{adj}" in self.data.columns:
            if adj == 'wins':
                lb = np.nanpercentile(self.data_raw['ret'], 100 - self.w)
                ub = np.nanpercentile(self.data_raw['ret'], self.w)
                if self.loc == 'right':
                    df = self.data_winsorized_ex_post[(self.data_winsorized_ex_post[f"ret_{adj}"]== ub)]
                elif self.loc == 'left':
                    df = self.data_winsorized_ex_post[(self.data_winsorized_ex_post[f"ret_{adj}"]== lb)]
                elif self.loc == 'both':
                    df = self.data_winsorized_ex_post[(self.data_winsorized_ex_post[f"ret_{adj}"]== lb) | (self.data_winsorized_ex_post[f"ret_{adj}"]== ub)] 

            else:
                df = self.data[(self.data[f"ret_{adj}"].isna()) & (self.data["ret"].notna())]
            
            # stats whole sample
            col_perc = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

            stats_all_ret = df['ret'].describe().to_frame("ALL")

            stats_list = []
            for col in self.chars:
                stats_col = df[col].describe().loc[['mean']].to_frame(f"Avg. {col.replace('_', ' ').title()}").rename(index={'mean': 'ALL'})
                stats_list.append(stats_col)

            if self.rating is None:
                df['rating_cat'] = np.where(df['RATING_NUM']>10, 'NIG','IG')

                # group by rating category
                stats_cat_ret = df.groupby(['rating_cat'])['ret'].describe()
                stats_cat_list = []
                for col in self.chars:
                    stats_cat_col = df.groupby(['rating_cat'])[col].describe().loc[:, 'mean'].to_frame(f"Avg. {col.replace('_', ' ').title()}")
                    stats_cat_col = pd.concat([stats_cat_col, stats_list[self.chars.index(col)]])
                    stats_cat_list.append(stats_cat_col)

                # concatenate dfs
                stats_cat_ret = pd.concat([stats_cat_ret, stats_all_ret.T])
                stats = pd.DataFrame()
                for x in range(len(self.chars)):
                    stats = pd.concat([stats,stats_cat_list[x]], axis=1)
                stats = pd.concat([stats,stats_cat_ret],axis =1)
                total_count = stats.loc['ALL', 'count'].sum()

                stats['count'] = stats['count'].apply(lambda x: f"{int(x)} ({x / total_count * 100:.2f}%)")
            else:
                stats = pd.concat(stats_list + [stats_all_ret.T], axis=1)
            # rearranging and renaming columns

            stats[col_perc] = (stats[col_perc]).map(lambda x: f"{x * 100:.2f}%")
            for col in self.chars:
                stats[f"Avg. {col.replace('_', ' ').title()}"] = stats[f"Avg. {col.replace('_', ' ').title()}"].apply(lambda x: f"{x:,.2f}")
            # stats['Avg. TMT'] = stats['Avg. TMT'].round(3)

            stats = stats[['count'] + [f"Avg. {col.replace('_', ' ').title()}" for col in self.chars] + col_perc]            
            stats.rename(columns={'count':'# Bonds', 'mean':'Avg. Ret', 'std':'Std. Dev.', 'min':'Min', '25%':'25$^{th}$', '50%':'Median', '75%':'75$^{th}$', 'max':'Max'}, inplace=True)

            return stats


        






