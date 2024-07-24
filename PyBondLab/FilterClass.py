# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:12:30 2024

@authors: Giulio Rossetti & Alex Dickerson
"""
import numpy as np
import pandas as pd

class Filter:
    def __init__(self, data, adj, w, loc, percentile_breakpoints=None, price_threshold=None):
        self.data = data
        self.adj = adj
        self.w = w
        self.loc = loc
        self.percentile_breakpoints = percentile_breakpoints
        self.price_threshold = price_threshold   
        
        if isinstance(self.w, list) and len(self.w) == 2:
            lower_bound, upper_bound = w
            self.name_filt = f"_{self.adj}_{str(round(lower_bound,3))}_{str(round(upper_bound,3))}"
        else:    
            self.name_filt = f"_{self.adj}_{str(round(self.w,3))}"
        
        
        if self.loc:
            self.name_filt += f"_{self.loc}"

    def apply_filters(self):
        if self.adj == 'trim':
            self._trimming(self.w)
        elif self.adj == 'price':
            self._pricefilter(self.w)
        elif self.adj == 'wins':
            self.data_winsorized_ex_post = self._winsorizing_ep(self.data.copy(),self.w)
            
            self._winsorizing(self.w)
        elif self.adj == 'bounce':
            self._bounce(self.w)
        return self.data
    
    def _ex_ante_wins_threshold(self, w):
        if self.percentile_breakpoints is not None:
            df_d = self.percentile_breakpoints.loc[:, [str(round(w, 4)), str(round(100 - w, 4))]]
        else:
            data_w = self.data.copy()
            data_w.sort_values(by='date', inplace=True)
            df_d = pd.DataFrame(index=data_w['date'].unique(), columns=[w, 100 - w])
            for current_date in data_w['date'].unique():
                pooled_data = data_w[data_w['date'] < current_date]['ret']
                lb = np.nanpercentile(pooled_data, 100 - w)
                ub = np.nanpercentile(pooled_data, w)
                df_d.loc[current_date, w] = ub
                df_d.loc[current_date, 100 - w] = lb
        df_d.reset_index(inplace=True)
        df_d.rename({'index': 'date'}, axis=1, inplace=True)
        self.winz_threshold_ex_ante = df_d
        return df_d

    def _trimming(self, w):
        adj = 'trim'
        if isinstance(w, list) and len(w) == 2:
            lower_bound, upper_bound = w
            self.data[f"ret_{adj}"] = np.where((self.data['ret'] > upper_bound) | (self.data['ret'] < lower_bound), np.nan, self.data['ret'])          
        elif w >= 0:
            self.data[f"ret_{adj}"] = np.where(self.data['ret'] > w, np.nan, self.data['ret'])
        else:
            self.data[f"ret_{adj}"] = np.where(self.data['ret'] < w, np.nan, self.data['ret'])
        self.data[f"ret_{adj}"] = pd.to_numeric(self.data[f"ret_{adj}"])

    def _pricefilter(self, w):
        adj = 'price'
        if isinstance(w, list) and len(w) == 2:
            lower_bound, upper_bound = w
            self.data[f"ret_{adj}"] = np.where((self.data['PRICE'] > upper_bound) | (self.data['PRICE'] < lower_bound), np.nan, self.data['ret'])              
        elif w >= self.price_threshold:
            self.data[f"ret_{adj}"] = np.where(self.data['PRICE'] > w, np.nan, self.data['ret'])
        else:
            self.data[f"ret_{adj}"] = np.where(self.data['PRICE'] < w, np.nan, self.data['ret'])

    def _bounce(self, w):
        adj = 'bounce'
        self.data['ret_LAG'] = self.data.groupby("ID")['ret'].shift(1)   
        self.data['bounce'] = self.data['ret_LAG'] * self.data['ret']
        if isinstance(w, list) and len(w) == 2:
            lower_bound, upper_bound = w
            self.data[f"ret_{adj}"] = np.where((self.data['bounce'] > upper_bound) | (self.data['bounce'] < lower_bound), np.nan, self.data['ret'])              
        elif w >= 0:
            self.data[f"ret_{adj}"] = np.where(self.data['bounce'] > w, np.nan, self.data['ret'])
        else:
            self.data[f"ret_{adj}"] = np.where(self.data['bounce'] < w, np.nan, self.data['ret'])

    def _winsorizing(self, w):
        adj = 'wins'

        thr_ts = self._ex_ante_wins_threshold(w)
        self.data = pd.merge(self.data, thr_ts, on='date', how='left')
        if self.percentile_breakpoints is not None:
            bound_u = str(w)
            bound_d = str(round(100 - w, 4))
        else:
            bound_u = w
            bound_d = 100 - w
        if self.loc == "both":
            self.data[f"ret_{adj}"] = np.where(
                pd.isna(self.data['ret']), self.data['ret'],
                np.where(
                    self.data['ret'] > self.data[bound_u], self.data[bound_u],
                    np.where(self.data['ret'] < self.data[bound_d], self.data[bound_d], self.data['ret'])
                )
            )
        elif self.loc == "right":
            self.data[f"ret_{adj}"] = np.where(
                pd.isna(self.data['ret']), self.data['ret'],
                np.where(self.data['ret'] > self.data[bound_u], self.data[bound_u], self.data['ret'])
            )
        elif self.loc == "left":
            self.data[f"ret_{adj}"] = np.where(
                pd.isna(self.data['ret']), self.data['ret'],
                np.where(self.data['ret'] < self.data[bound_d], self.data[bound_d], self.data['ret'])
            )
        self.data[f"ret_{adj}"] = pd.to_numeric(self.data[f"ret_{adj}"])
        
    def _winsorizing_ep(self,df,w):
        adj = 'wins'
        df1 = df.copy() # redundant
        self.lb_ep = np.nanpercentile(df1['ret'], 100 - w)
        self.ub_ep = np.nanpercentile(df1['ret'], w)
        if self.loc == 'both':
            df1[f"ret_{adj}"] = np.where(df1['ret'] > self.ub_ep, self.ub_ep,
                                 np.where(df1['ret'] < self.lb_ep, self.lb_ep, 
                                        df1['ret']))
        if self.loc == 'right':
            df1[f"ret_{adj}"] = np.where(df1['ret'] > self.ub_ep, self.ub_ep, df1['ret'])
        
        if self.loc == 'left':
            df1[f"ret_{adj}"] = np.where(df1['ret'] < self.lb_ep, self.lb_ep, df1['ret'])
        return df1
    
