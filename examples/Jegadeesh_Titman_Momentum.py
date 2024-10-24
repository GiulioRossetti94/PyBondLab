################################################
# Jegadeesh & Titman (1993) Momentum Portfolio #
# October 2024                                 #  
# PyBondLab Example                            #
# October 2024                                 #
# Alex Dickerson                               #
# GitHub: 
################################################

##########################################################
# Credit to Qingyi (Freda) Song Drechsler and Alex Malek #
# for the code that downloads the equity data from CRSP  # 
# and formats it.                                        #
##########################################################
import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats
from tqdm import tqdm
import PyBondLab as pbl
import urllib.request
import zipfile
import statsmodels.api as sm
tqdm.pandas()

###################
# Connect to WRDS #
###################
wrds_user = ''
conn=wrds.Connection(wrds_username = wrds_user)

###################
# CRSP Block      #
###################
# sql similar to crspmerge macro
# added exchcd=-2,-1,0 to address the issue that stocks temp stopped trading
# without exchcd=-2,-1, 0 the non-trading months will be tossed out in the output
# leading to wrong cumret calculation in momentum step
# Code	Definition
# -2	Halted by the NYSE or AMEX
# -1	Suspended by the NYSE, AMEX, or NASDAQ
# 0	Not Trading on NYSE, AMEX, or NASDAQ
# 1	New York Stock Exchange
# 2	American Stock Exchange

crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, b.ncusip, a.date, 
                      b.shrcd, b.exchcd, b.siccd,
                      a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1961' and '12/31/1991'
                      and b.exchcd between -2 and 2
                      and b.shrcd between 10 and 11
                      """) 

# Change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=\
    crsp_m[['permco','permno','shrcd','exchcd']].astype(int)
    
# Line up date to be end of month
crsp_m['date']=pd.to_datetime(crsp_m['date'])    
    
#### Re-format variables for PyBondLab ####
crsp_m['ret']= crsp_m['ret'].fillna(0)

crsp_m['logret'] = np.log( 1+  crsp_m['ret'] )
crsp_m = crsp_m.set_index(['permno','date'])

crsp_m['mom6'] = crsp_m.groupby("permno",group_keys=False)['logret'].progress_apply(
    lambda x: x.rolling(window=6, min_periods=6).sum()
)

crsp_m['mom6']=np.exp(crsp_m['mom6'])-1

# price adjusted
crsp_m['p']=crsp_m['prc'].abs()/crsp_m['cfacpr'] 

# total shares out adjusted
crsp_m['tso']=crsp_m['shrout']*crsp_m['cfacshr']*1e3 

# market cap in $mil
crsp_m['me'] = crsp_m['p']*crsp_m['tso']/1e6 

crsp_m['VW'] = crsp_m['me']


crsp_m = crsp_m.reset_index()
N = len(np.unique(crsp_m['permno']))
ID = dict(zip(np.unique(crsp_m['permno']).tolist(),np.arange(1,N+1)))
crsp_m["ID"] = crsp_m['permno'].apply(lambda x: ID[x]) 
crsp_m = crsp_m.sort_values(['ID','date'])

# Dummy for rating number #
crsp_m['RATING_NUM'] = 1

# initialize the strategy
holding_period = 6             # holding period returns
n_portf        = 10            # number of portfolios, 5 is quintiles, 10 deciles
skip           = 0             # no skip period, i.e., observe at signal at t
                               # and trade in signal at t (no delay in trade)

sort_var1 = 'mom6'

single_sort = pbl.SingleSort(holding_period, 
                                 sort_var1, 
                                 n_portf,
                                 skip)
  
# parameters
# we do not conduct any filtering, so we leave most of this blank
params = {'strategy': single_sort,
          'rating':None, 
          'dynamic_weights':True,      
}

# Fit the strategy to the data. 
RESULTS = pbl.StrategyFormation(crsp_m, **params).fit()

# extract the long-short strategies #
factor = pd.concat(RESULTS.get_long_short(), axis = 1) 
factor.columns = ['EW_'+ str('MOM') , 'VW_'+ str('MOM')]  
print( factor.mean()*100*12 )

# long (winners)
factorl = pd.concat(RESULTS.get_long_leg(), axis = 1) 
factorl.columns = ['EW_'+ str('winners') , 'VW_'+ str('winners')]  
print( factorl.mean()*100*12 )

# long (losers)
factors = pd.concat(RESULTS.get_short_leg(), axis = 1) 
factors.columns = ['EW_'+ str('losers') , 'VW_'+ str('losers')]  
print( factors.mean()*100*12 )

factor = factor.merge(factorl , how = "inner", left_index = True, right_index = True)
factor = factor.merge(factors , how = "inner", left_index = True, right_index = True)
factor = factor.dropna()

#* ************************************** */
#* Import additional Factors              */
#* ************************************** */
         
ff_url = str("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/")+\
str("F-F_Research_Data_5_Factors_2x3_CSV.zip")

urllib.request.urlretrieve(ff_url,'fama_french.zip')
zip_file = zipfile.ZipFile('fama_french.zip', 'r')

# We will call it ff_factors.csv
zip_file.extractall()
zip_file.close()

ff_factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', skiprows = 3)
ff_factors = ff_factors.iloc[:714,:]
ff_factors.rename(columns={'Unnamed: 0':'date',
                            'Mkt-RF':'MKTS'}, inplace=True)
ff_factors['date'] = pd.to_datetime(ff_factors['date'], format = "%Y%m")
ff_factors['date'] = ff_factors['date'] + MonthEnd(0)
ff_factors = ff_factors.set_index('date')
ff_factors[ff_factors.columns] =\
    ff_factors[ff_factors.columns].apply(pd.to_numeric,
                                          errors='coerce')
ff_factors = ff_factors / 100
ff5        =   ff_factors
ff5 .drop(['RF'], axis = 1, inplace = True)

# Risk premia/alpha and t-statistics #
tstats                =  list()
tstats_alpha          =  list()
alpha                 =  list()

for i,s in enumerate(factor.columns):
    print(s)
    regB = sm.OLS(factor.iloc[:,i].values, 
                      pd.DataFrame(np.tile(1,(len(factor.iloc[:,i]),1)) ).values,
                      missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':4})
       
    tstats.append(np.round(regB.tvalues[0],2))  
    
    _HML = factor.iloc[:,i].to_frame()
    _HML = _HML.merge(ff5,how="inner",left_index=True,right_index=True).dropna()
    regB = sm.OLS(_HML.iloc[:,0], 
                      sm.add_constant(_HML[['MKTS', 'SMB', 'HML', 'RMW', 'CMA']]),                      
                      ).fit(cov_type='HAC',cov_kwds={'maxlags':4})
    regB.summary()
        
    tstats_alpha.append(np.round(regB.tvalues.iloc[0] ,2)) 
    alpha.append( np.round(regB.params.iloc [0]*100,2))

# Premia #
Mean       = (pd.DataFrame(np.array(factor.mean())) * 100).round(2).T  
Mean.index = ['Avg Ret(%)' ]
Mean.columns= factor.columns
Tstats     = pd.DataFrame(np.array(tstats)).T
Tstats.index = [str('t-stat') ]
Tstats  = "(" + Tstats.astype(str)  + ")" 
Tstats .columns= factor.columns
         
RiskPremia = pd.concat([Mean, Tstats], axis = 0)

print(RiskPremia)

# Alpha #
Alpha       = (pd.DataFrame(np.array(alpha)) ).round(2).T  
Alpha.index = ['Alpha (%)' ]
Alpha.columns= factor.columns
TstatsA     = pd.DataFrame(np.array(tstats_alpha)).T
TstatsA.index = [str('t-stat') ]
TstatsA  = "(" + TstatsA.astype(str)  + ")" 
TstatsA .columns= factor.columns
         
Alpha = pd.concat([Alpha, TstatsA], axis = 0)

print(Alpha)
################################ END ##########################################
