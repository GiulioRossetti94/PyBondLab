# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:28:52 2024

@author: phd19gr
"""

import numpy as np
import pandas as pd
from .FilterClass import Filter
from .StrategyClass import *
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
    def __init__(self, data: pd.DataFrame,strategy: Strategy, rating: str = None, filters: dict = None):
        
        self.data_raw = data.copy()
        self.data = data.copy()
        # self.datelist = pd.Series(self.data['date'].unique()).sort_values().tolist()
        self.datelist = pd.Series(self.data['date'].unique()).sort_values().tolist() if 'date' in self.data.columns else None
            
        self.strategy = strategy
        self.nport = strategy.nport
        self.rating = rating
        self.filters = filters if filters else {}
        
        self.adj = self.filters.get('adj')
        self.w = self.filters.get('level')
        self.loc = self.filters.get('location')
        self.percentile_breakpoints = self.filters.get('df_breakpoints') if self.adj == 'wins' else None
        self.price_threshold = self.filters.get('price_threshold', 25) if self.adj == 'price' else None
        # stratey params
                
        if self.rating is None:
            self.name = "ALL_" + self.strategy.str_name
        else:
            self.name = f"{self.rating}_" + self.strategy.str_name
            
        # initialize df 
        
        self.ewls_ep_df = None 
        self.vwls_ep_df = None
    
    def fit(self, *, IDvar=None, DATEvar=None, RETvar=None, PRICEvar=None, Wvar = None):
        if IDvar or DATEvar or RETvar or PRICEvar:
            # here check if a "ret" column is present 
            if RETvar and "ret" in self.data.columns:
                self.data.drop(columns="ret", inplace=True)
                self.data_raw.drop(columns="ret", inplace=True)
                warnings.warn("Column 'ret' already exists. It will be overwritten.", UserWarning)
            if PRICEvar and "PRICE" in self.data.columns:
                self.data.drop(columns="PRICE", inplace=True)
                self.data_raw.drop(columns="PRICE", inplace=True)
                warnings.warn("Column 'PRICE' already exists. It will be overwritten.", UserWarning)
            if IDvar and "ID" in self.data.columns:
                self.data.drop(columns="ID", inplace=True)
                self.data_raw.drop(columns="ID", inplace=True)
                warnings.warn("Column 'ID' already exists. It will be overwritten.", UserWarning)
            if DATEvar and "date" in self.data.columns: 
                self.data.drop(columns="date", inplace=True)
                self.data_raw.drop(columns="date", inplace=True)
            if Wvar and "VW" in self.data.columns:
                self.data.drop(columns="VW", inplace=True)
                self.data_raw.drop(columns="VW", inplace=True)
                warnings.warn("Column 'VW' already exists. It will be overwritten.", UserWarning)
            if Wvar is None and "VW" not in self.data.columns:
                self.data['VW'] = 1
                self.data_raw['VW'] = 1
                warnings.warn("Column 'VW' does not exist. Setting VW = 1 (i.e. equal weights)", UserWarning)

            self.rename_id(IDvar=IDvar, DATEvar=DATEvar, RETvar=RETvar, PRICEvar=PRICEvar, Wvar = Wvar)
        
        required_columns = ['ID', 'date', 'ret']               
        if self.adj == 'price':
            required_columns.append('PRICE')
            
        missing_columns = [col for col in required_columns if col not in self.data.columns]    
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")         
        
        # force the IDs to be numbers. Needed to facilitate storing results
        N = len(np.unique(self.data["ID"]))
        ID = dict(zip(np.unique(self.data["ID"]).tolist(),np.arange(1,N+1)))
        self.unique_bonds = N
        self.data["ID"] = self.data["ID"].apply(lambda x: ID[x])
        
        self.compute_signal()
        self.portfolio_formation()
        return self
    
    def rename_id(self, *, IDvar=None, DATEvar=None,RETvar=None, PRICEvar=None, Wvar = None):
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
        # get params from strategy(
        print(self.strategy.__strategy_name__)
        if self.filters and self.adj in ["trim", "wins", "price", "bounce"]:
            filter_obj = Filter(self.data, self.adj, self.w, self.loc, self.percentile_breakpoints,self.price_threshold)
            self.name += filter_obj.name_filt
            
            self.data = filter_obj.apply_filters()
            sort_var = self.strategy.get_sort_var(self.adj)
            
            if self.adj == 'wins':
                # get the winsorized returns for ex post winsorization
                self.data_winsorized_ex_post = filter_obj.data_winsorized_ex_post# this is a df w/ wisorized returns
            
            if 'signal' in sort_var:
                if self.strategy.__strategy_name__ == "MOMENTUM":
                    J = self.strategy.J
                    skip = self.strategy.skip
                    varname = f'ret_{self.adj}'
                    self.data['logret'] = np.log(self.data[varname] + 1)
                    self.data[f'signal_{self.adj}'] = self.data.groupby(['ID'], group_keys=False)['logret']\
                        .rolling(J, min_periods=J).sum().values
                    self.data[f'signal_{self.adj}'] = np.exp(self.data[f'signal_{self.adj}']) - 1
                    self.data[f'signal_{self.adj}'] = self.data.groupby("ID")[f'signal_{self.adj}'].shift(skip)
                elif self.strategy.__strategy_name__ == "LT-REVERSAL":
                    J = self.strategy.J
                    skip = self.strategy.skip
                    varname = f'ret_{self.adj}'
                    self.data[f'signal_{self.adj}'] = self.data.groupby(['ID'], group_keys=False)[varname]\
                        .apply(lambda x:  x.rolling(window = J).sum()) - \
                               self.data.groupby(['ID'], group_keys=False)[varname]\
                        .apply(lambda x:  x.rolling(window = skip).sum())
                    
         
        else:
            sort_var = self.strategy.get_sort_var()
            # if sort_var == 'signal':
            #     J = self.strategy.J
            #     skip = self.strategy.skip
            #     self.data['logret'] = np.log(self.data['ret'] + 1)
            #     self.data['signal'] = self.data.groupby(['ID'], group_keys=False)['logret']\
            #         .rolling(J, min_periods=J).sum().values            
            #     self.data['signal'] = np.exp(self.data['signal']) - 1
            #     self.data['signal'] = self.data.groupby("ID")['signal'].shift(skip)
            
            
                
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
        
        # =====================================================================
        # Unpack for double sorting
        # =====================================================================
        DoubleSort = getattr(self.strategy, 'DoubleSort', False)
        if DoubleSort:
            # DoubleSort = 1
            nport2 = self.strategy.nport2
            sort_var2 = self.strategy.sort_var2
            tot_nport = nport * nport2
                      
        else:
            nport2 = None
            sort_var2 = None
            tot_nport = nport
        
        # initialize storing   
        ewport_hor_ea = np.full((TM, hor, tot_nport), np.nan)
        vwport_hor_ea = np.full((TM, hor, tot_nport), np.nan)
        
        # weights
        self.ewport_weight_hor_ea = np.zeros((TM, hor, tot_nport,unique_bonds))
        self.vwport_weight_hor_ea = np.zeros((TM, hor, tot_nport,unique_bonds))

        if adj:
            ewport_hor_ep = np.full((TM, hor, tot_nport), np.nan)
            vwport_hor_ep = np.full((TM, hor, tot_nport), np.nan) 
            
            # weights
            self.ewport_weight_hor_ep = np.zeros((TM, hor, tot_nport,unique_bonds))
            self.vwport_weight_hor_ep = np.zeros((TM, hor, tot_nport,unique_bonds))
        
        # for t in range((hor+1), TM - hor): to discuss this!
        for t in range( TM - hor):
            # =====================================================================
            # Filter based on ratings and signal != nan
            # =====================================================================
            if self.rating == "NIG":
                It0 = tab[(tab['date'] == self.datelist[t]) & (~tab[sort_var].isna()) & 
                          (tab['RATING_NUM'] > 10) & (tab['RATING_NUM'] <= 22)]
            elif self.rating == "IG":
                It0 = tab[(tab['date'] == self.datelist[t]) & (~tab[sort_var].isna()) & 
                          (tab['RATING_NUM'] >= 1) & (tab['RATING_NUM'] <= 10)]
            else:
                It0 = tab[(tab['date'] == self.datelist[t]) & (~tab[sort_var].isna())]                 
        
            if DoubleSort:
                # here also signal2 != nan
                if self.rating == "NIG":
                    It0 = tab[(tab['date'] == self.datelist[t]) & (~tab[sort_var].isna()) & (~tab[sort_var2].isna()) &
                              (tab['RATING_NUM'] > 10) & (tab['RATING_NUM'] <= 22)]
                elif self.rating == "IG":
                    It0 = tab[(tab['date'] == self.datelist[t]) & (~tab[sort_var].isna()) & (~tab[sort_var2].isna()) &
                              (tab['RATING_NUM'] >= 1) & (tab['RATING_NUM'] <= 10)]
                else:
                    It0 = tab[(tab['date'] == self.datelist[t]) & (~tab[sort_var].isna()) & (~tab[sort_var2].isna()) ]                 
            
            if It0.shape[0] == 0:
                if t > hor:
                    print(f"no bonds at time {t}:{self.datelist[t]}. Going to next period.")      
                continue
            
            # =====================================================================
            # Investment Universe matching
            # =====================================================================
            if adj == 'trim' and self.w:
                if isinstance(self.w, list) and len(self.w) == 2:
                    lower_bound, upper_bound = self.w
                    It0 = It0[(It0[ret_var] <= upper_bound) & (It0[ret_var] >= lower_bound)]
                else:
                    It0 = It0[It0[ret_var] <= self.w] if self.w > 0 else It0[It0[ret_var] >= self.w]
                    
            elif adj == ' bounce' and self.w:
                if isinstance(self.w, list) and len(self.w) == 2:
                    lower_bound, upper_bound = self.w
                    It0 = It0[(It0['bounce'] <= upper_bound) & (It0['bounce'] >= lower_bound)]
                else:
                    It0 = It0[It0['bounce'] >= self.w] if self.w < 0 else It0[It0['bounce']<= self.w]
                    
            elif adj == 'price' and self.w:
                if isinstance(self.w, list) and len(self.w) == 2:
                    lower_bound, upper_bound = self.w
                    It0 = It0[(It0['PRICE'] <= upper_bound) & (It0['PRICE'] >= lower_bound)]
                else:
                    It0 = It0[It0['PRICE'] <= self.w] if self.w > self.price_threshold else It0[It0['PRICE'] >= self.w]

            # =====================================================================
            # start sorting procedure
            # =====================================================================
            
            for h in range(1, hor + 1):
                # Investment universe for ret computation
                It1 = self.data_raw[(self.data_raw['date'] == self.datelist[t + h])& (~self.data_raw[ret_var].isna())]
                
                if adj == 'wins' and 'signal' in sort_var:
                    # TODO can be removed
                    # use winsorized returns to assign to ptfs
                    port_ret_ea = self.port_sorted_ret(It0, It1,ret_var, sort_var,DoubleSort=DoubleSort,sig2 = sort_var2,nport2 = nport2 )
                else:
                    # if signal is not in sort_var, we do not use winsorized returns to sort portfolios
                    port_ret_ea = self.port_sorted_ret(It0, It1,ret_var, sort_var,DoubleSort=DoubleSort,sig2 = sort_var2,nport2 = nport2 )
                
                # storing returns
                ewport_hor_ea[t + h, h - 1, :] = port_ret_ea[0]
                vwport_hor_ea[t + h, h - 1, :] = port_ret_ea[1]
                # storing weights: todo give option for this
                weights = port_ret_ea[2]
                self.fill_weights(weights, self.ewport_weight_hor_ea,self.vwport_weight_hor_ea, t, h)

                
                
                if adj:
                    if adj == 'wins':
                        It2 = tab_ex_post_wins[(tab_ex_post_wins['date'] == self.datelist[t + h]) & (~tab_ex_post_wins[ret_var].isna())]
                    else:
                        It2 = tab[(tab['date'] == self.datelist[t + h]) & (~tab[ret_var + "_" + adj].isna())]

                    
                    port_ret_ep = self.port_sorted_ret(It0, It2,ret_var + "_" + adj, sort_var,DoubleSort=DoubleSort,sig2 = sort_var2,nport2 = nport2 )
                    
                    # storing returns
                    ewport_hor_ep[t + h, h - 1, :] = port_ret_ep[0]
                    vwport_hor_ep[t + h, h - 1, :] = port_ret_ep[1]
                    # storing weights: 
                    weights_ep = port_ret_ep[2]
                    self.fill_weights(weights_ep, self.ewport_weight_hor_ep,self.vwport_weight_hor_ep, t, h)
                        
                
        self.ewport_ea = np.mean(ewport_hor_ea, axis=1)
        self.vwport_ea = np.mean(vwport_hor_ea, axis=1)
        if adj:
            self.ewport_ep = np.mean(ewport_hor_ep, axis=1)
            self.vwport_ep = np.mean(vwport_hor_ep, axis=1)    
            
        nport_tot = len(port_ret_ea[0])   # this is nport1 x nport2
        
        # =====================================================================
        # Compute portfolio returns
        # =====================================================================
        if DoubleSort:  
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
            # computing long leg
            EWlong_leg_ea = self.ewport_ea[:, nport_tot - 1]
            VWlong_leg_ea = self.vwport_ea[:, nport_tot - 1]
            
            # computing short leg            
            EWshort_leg_ea = self.ewport_ea[:, 0]
            VWshort_leg_ea = self.vwport_ea[:, 0]          
            
            # Long short portfolio
            self.vwls_ea = self.vwport_ea[:, nport_tot - 1] - self.vwport_ea[:, 0]
            self.ewls_ea = self.ewport_ea[:, nport_tot - 1] - self.ewport_ea[:, 0]
            # computing long leg df
            self.ewls_ea_long_df = pd.DataFrame(EWlong_leg_ea,index = self.datelist, columns = ['LONG_EWEA_' + self.name]) 
            self.vwls_ea_long_df = pd.DataFrame(VWlong_leg_ea,index = self.datelist, columns = ['LONG_VWEA_' + self.name]) 
            # computing short leg df 
            self.ewls_ea_short_df = pd.DataFrame(EWshort_leg_ea,index = self.datelist, columns = ['SHORT_EWEA_' + self.name]) 
            self.vwls_ea_short_df = pd.DataFrame(VWshort_leg_ea,index = self.datelist, columns = ['SHORT_VWEA_' + self.name])  
      
        self.ewls_ea_df = pd.DataFrame(self.ewls_ea,index = self.datelist, columns = ['EWEA_' + self.name]) 
        self.vwls_ea_df = pd.DataFrame(self.vwls_ea,index = self.datelist, columns = ['VWEA_' + self.name]) 
        
        # computing turnover
        ew_port_turn_ea = self.compute_turnover(self.ewport_weight_hor_ea)
        vw_port_turn_ea = self.compute_turnover(self.vwport_weight_hor_ea)
        
        self.ewturnover_ea_df = pd.DataFrame(ew_port_turn_ea,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,nport_tot+1)])
        self.vwturnover_ea_df = pd.DataFrame(vw_port_turn_ea,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,nport_tot+1)])
        
        if adj:
            if DoubleSort:
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
                self.ewls_ep = self.ewport_ep[:, nport_tot - 1] - self.ewport_ep[:, 0]
                self.vwls_ep = self.vwport_ep[:, nport_tot - 1] - self.vwport_ep[:, 0]
                
                # computing long leg
                EWlong_leg_ep = self.ewport_ep[:, nport_tot - 1]
                VWlong_leg_ep = self.vwport_ep[:, nport_tot - 1]
                
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
            
            # turnovocer
            ew_port_turn_ep = self.compute_turnover(self.ewport_weight_hor_ep)
            vw_port_turn_ep = self.compute_turnover(self.vwport_weight_hor_ep)
            
            self.ewturnover_ep_df = pd.DataFrame(ew_port_turn_ep,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,nport_tot+1)])
            self.vwturnover_ep_df = pd.DataFrame(vw_port_turn_ep,index = self.datelist[1:], columns = [f"Q{x}" for x in range(1,nport_tot+1)])
            
            
                
    def port_sorted_ret(self, It0, It1, ret_col, sig,**kwargs):
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
        doubleSort = kwargs.get('DoubleSort', None)
        sig2 = kwargs.get('sig2', None)
        nport2 = kwargs.get('nport2', None)
        
        # =====================================================================
        # compute edges for first and second signals
        # =====================================================================
        
        nport = self.nport    # number of portfolios
        thres = np.percentile(It0[sig], np.linspace(0, 100, nport + 1))        # compute edges for signal
        thres[0] = -np.inf
        
        if doubleSort:
            thres2 = np.percentile(It0[sig2], np.linspace(0, 100, nport2 + 1))# compute edges for signal2
            thres2[0] = -np.inf
        
        id0 = It0['ID']
        id1 = It1['ID']
        
        intersect_ids = id0[id0.isin(id1)]
        It0 = It0[id0.isin(intersect_ids)].copy()
        It1 = It1[id1.isin(intersect_ids)].copy() 
        
        # missing_ids = id0[~id0.isin(id1)]
        It1['VW'] = It0['VW'].values
        
        sortvar = It0[sig]
        # =====================================================================
        # Rank bonds based on signals
        # =====================================================================
        if doubleSort:
            nportmax = nport * nport2 
            # Double sorting: compute the rank independently
            sortvar2 = It0[sig2]
            idx1 = self.compute_rank_idx(sortvar,thres,nport)
            idx2 = self.compute_rank_idx(sortvar2,thres2,nport2)      
            # if condSort:
            # It1['ptf_rank'] = self.cond_sort(sortvar2, idx1, nport, nport2)#(sortvar2, idx1, n1, n2)
            
            # create a column with final rank: ptfs going from 1 to nport1 x nport2
            It1['ptf_rank'] = self.double_idx_uncond(idx1, idx2, nport, nport2)    
            # rank_bonds = self.double_idx_uncond(idx1, idx2, nport, nport2)  
                    
        else:
            nportmax = nport
            It1['ptf_rank'] = self.compute_rank_idx(sortvar,thres,nport)
                               
        It1['weights'] = It1.groupby('ptf_rank')['VW'].apply(lambda x: x / x.sum()).reset_index(level=0, drop=True)
        
        # It1[ret_col] = It1[ret_col].fillna(0)
        ptf_ret_ew = It1.groupby('ptf_rank')[ret_col].mean()
        ptf_ret_vw = It1.groupby('ptf_rank').apply(lambda x: (x[ret_col] * x['weights']).sum())
        
        # =====================================================================
        # Create csv with number of assets in each bin
        # =====================================================================
        # ptf_count = It1.groupby('ptf_rank')['ID'].count()
        # ptf_count_df = ptf_count.reset_index().T
        # ptf_count_df['date'] = It1.date.iloc[0]
        # ptf_count_df = ptf_count_df.drop('ptf_rank')
        # csv_file_path = 'debug_ptf_count.csv'
        # if not os.path.isfile(csv_file_path):
        #     ptf_count_df.to_csv(csv_file_path, mode='w', index=False, header=True)
        # else:
        #     ptf_count_df.to_csv(csv_file_path, mode='a', index=False, header=False)
        # =====================================================================
        # store the weights:  return the column with ID 
        # =====================================================================  
        rank_ = It1[['ID','ptf_rank']]
        rank = rank_.copy()
        rank['count'] = rank.groupby('ptf_rank')['ID'].transform('count')
        rank['eweights'] = 1 / rank['count']
        rank = rank.merge(It1[['ID', 'weights']], on='ID')
        rank = rank.rename(columns={"weights":"vweights"})
        
        _weights = rank[['ID','ptf_rank','eweights','vweights']]
                
        nport_idx = range(1,int(nportmax+1))
        ptf_ret_ew = ptf_ret_ew.reindex(nport_idx)
        ptf_ret_vw = ptf_ret_vw.reindex(nport_idx)
       
        ewl = ptf_ret_ew.to_list()
        vwl = ptf_ret_vw.to_list()
        
        return ewl, vwl,_weights
    
    @staticmethod
    def compute_rank_idx(sortvar,thres,nport):
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
    def double_idx_uncond(idx1, idx2, n1, n2):
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
    def cond_sort(sortvar2, idx1, n1, n2):
        """
        Python adaptation of AssayingAnomalies func 
        https://github.com/velikov-mihail/AssayingAnomalies/tree/main
        """
        n2 = int(n2)
        n1 = int(n1)
        idx = np.full(idx1.shape,np.nan)
        thres = np.percentile(sortvar2[~np.isnan(sortvar2)], np.linspace(0, 100, n2 + 1))
        thres[0] = -np.inf  # To handle values smaller than the first percentile
        for i in range(1, n1 + 1):
            # Sort within each of the portfolios sorted based on the first variable
            temp = np.copy(sortvar2)
            temp[idx1 != i] = np.nan
            temp_ind = StrategyFormation.compute_rank_idx(temp,thres, n2)
            n2 = int(np.nanmax(temp_ind))
            idx[(idx1 == i) & (temp_ind > 0)] = temp_ind[(idx1 == i) & (temp_ind > 0)] + n2 * (i - 1)
        return idx
    
    @staticmethod
    def fill_weights(weights, array_ew,array_vw, t, h):
        for _, row in weights.iterrows():
            p = int(row['ptf_rank'])
            ID = int(row['ID'])
            eweights = row['eweights']
            vweights = row['vweights']
            
            array_ew[t + h - 1, h - 1, p - 1, ID - 1] = eweights
            array_vw[t + h - 1, h - 1, p - 1, ID - 1] = vweights
            
    @staticmethod  
    def compute_turnover(weights):
        #  Portfolio weight changes: Subtract return-adjusted weights (lagged) from weights;
        abs_dewport_weight = abs(weights[1:,:,:,:] - weights[:-1,:,:,:])
        port_turn_hor    = np.sum(abs_dewport_weight, axis=3)
        mean_port_turn_hor = np.mean(port_turn_hor, axis=1)
        port_turn = np.squeeze(mean_port_turn_hor)
        return port_turn
    
    # getters 
    def get_long_leg(self):
        if self.ewls_ea_long_df is None or self.vwls_ea_long_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None  
        else:
            return self.ewls_ea_long_df, self.vwls_ea_long_df

    def get_long_leg_ex_post(self):
        if self.ewls_ea_long_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None     
        else:
            return self.ewls_ep_long_df, self.vwls_ep_long_df

    def get_short_leg(self):
        if self.ewls_ea_short_df is None or self.vwls_ea_short_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None     
        else:
            return self.ewls_ea_short_df, self.vwls_ea_short_df

    def get_short_leg_ex_post(self):
        if self.ewls_ep_short_df is None or self.vwls_ep_short_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None   
        else:
            return self.ewls_ep_short_df, self.vwls_ep_short_df
    
    # Long-Short ptf
    def get_long_short(self):
        if self.ewls_ea_df is None or self.vwls_ea_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None   
        else:
            return self.ewls_ea_df, self.vwls_ea_df

    def get_long_short_ex_post(self):
        if self.ewls_ep_df is None or self.vwls_ep_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None   
        else:
            return self.ewls_ep_df, self.vwls_ep_df
        
    def get_ptf(self):
        if self.ewport_ea is None or self.vwport_ea is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None           
        return self.ewport_ea,self.vwport_ea

    def get_ptf_ex_post(self):
        if self.ewport_ep is None or self.vwport_ep is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None           
        return self.ewport_ep,self.vwport_ep
    
    def get_ptf_weights(self):
        if self.ewport_weight_hor_ea is None or self.vwport_weight_hor_ea is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None           
        return self.ewport_weight_hor_ea,self.vwport_weight_hor_ea   
    
    def get_ptf_weights_ex_post(self):
        if self.ewport_weight_hor_ep is None or self.vwport_weight_hor_ep is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None           
        return self.ewport_weight_hor_ep,self.vwport_weight_hor_ep  
    
    def get_ptf_turnover(self):
        if self.ewturnover_ea_df is None or self.vwturnover_ea_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None           
        return self.ewturnover_ea_df,self.vwturnover_ea_df

    def get_ptf_turnover_ex_post(self):
        if self.ewturnover_ep_df is None or self.vwturnover_ep_df is None:
            warnings.warn("The DataFrame has not been initialized.", UserWarning)
            return None           
        return self.ewturnover_ep_df,self.vwturnover_ep_df
    
    
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
        get the bonds filtered out
        
        """
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
            stats_all_ret = df['ret'].describe().to_frame("ALL")
            stats_all_tmt = df['TMT'].describe().loc[['mean']].to_frame("Avg. TMT").rename(index={'mean':'ALL'})
            stats_all_amt = df['AMOUNT_OUTSTANDING'].describe().loc[['mean']].to_frame("Avg. AMT. OUT").rename(index={'mean':'ALL'})
            if self.rating is None:
                df['rating_cat'] = np.where(df['RATING_NUM']>10, 'NIG','IG')

                # df['tmt_cat'] = pd.cut(df['TMT'], bins=[-float('inf'), 5, 12, float('inf')], labels=['short', 'med', 'long'])

                # group by rating category
                stats_cat_ret = df.groupby(['rating_cat'])['ret'].describe()
                stats_cat_tmt = df.groupby(['rating_cat'])['TMT'].describe().loc[:,'mean'].to_frame("Avg. TMT")
                stats_cat_amt = df.groupby(['rating_cat'])['AMOUNT_OUTSTANDING'].describe().loc[:,'mean'].to_frame("Avg. AMT. OUT")
                # concatenate dfs
                
                stats_cat_tmt = pd.concat([stats_cat_tmt,stats_all_tmt])
                stats_cat_ret = pd.concat([stats_cat_ret,stats_all_ret.T])
                stats_cat_amt = pd.concat([stats_cat_amt,stats_all_amt])

            stats = pd.concat([stats_cat_amt,stats_cat_tmt,stats_cat_ret],axis=1)
            col_perc = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

            total_count = stats.loc['ALL','count'].sum()

            stats['count'] = stats['count'].apply(lambda x: f"{int(x)} ({x / total_count * 100:.2f}%)")
            stats[col_perc] = (stats[col_perc]).map(lambda x: f"{x * 100:.2f}%")
            stats['Avg. AMT. OUT'] = stats['Avg. AMT. OUT'].apply(lambda x: f"{x:,.0f}")
            stats['Avg. TMT'] = stats['Avg. TMT'].round(3)

            # rearranging and renaming columns

            stats = stats[['count','Avg. AMT. OUT', 'Avg. TMT' ,'mean', 'std', 'min', '25%', '50%', '75%', 'max', ]]
            stats.rename(columns={'count':'# Bonds', 'mean':'Avg. Ret', 'std':'Std. Dev.', 'min':'Min', '25%':'25$^{th}$', '50%':'Median', '75%':'75$^{th}$', 'max':'Max'}, inplace=True)
            return stats


        






