# https://openbondassetpricing.com/machine-learning-data/
# https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples
# Load required packages
# I place the versions that this code was run after the package #
# Python: 3.12.4
import numpy as np   # 1.26.4
import pandas as pd  # 2.2.2
import PyBondLab as pbl # Make sure you are updated to PyBondLab 0.1.3
import statsmodels.api as sm # 0.14.2
import matplotlib.pyplot as plt # NA
from pandas.tseries.offsets import * # NA
import gzip # NA
import pickle # NA

# Load the OSBAP ML Database #
# Please specify your full file path where you have saved and extracted the 
# OSBAP_ML_Panel_Oct_2024.pkl.gz (compressed pickle file)
# i.e., r'C:\Users\JoeSoap\OSBAP\OSBAP_ML_Panel_Oct_2024.pkl.gz'
input_file = r''

# Read from file with gzip compression
with gzip.open(input_file, 'rb') as file:
    tbl1 = pickle.load(file)

# Allows is to inspect the columns etc. 
Columns = pd.Series( tbl1.columns  ) # For reference #

# Drop ID, which will be reset later after merging to returns #   
tbl1.drop(['ID'], axis= 1,inplace = True)

# Load the OSBAP Database for returns / other variables #
# This loads the data dirctly from the website #
url = "https://openbondassetpricing.com/wp-content/uploads/2024/07/WRDS_MMN_Corrected_Data_2024_July.csv"

# Read the CSV file directly from the URL into a pandas DataFrame
dfop = pd.read_csv(url)
dfop = dfop[['date', 'cusip','exretnc_bns','exretn','bond_ret']]
dfop['date'] = pd.to_datetime(dfop['date'])

# Merge the 2 datasts (inner join)
tbl1 = tbl1.merge(dfop, how  = "inner", left_on = ['date','cusip'], 
                  right_on   =['date','cusip'] )

# Sort the panel on the bond cusip and date
tbl1 = tbl1.sort_values(['cusip','date'])

# Create overall panel index
tbl1 = tbl1.reset_index()
tbl1['index'] = range(1,(len(tbl1)+1))

# Rename cusip to ID #
# This is for PyBondLab, feel free to keep CUSIP and create a new
# column for ID, i.e., tb1['ID'] = tb1['cusip']
tbl1.rename(columns = {'cusip':'ID'}, inplace = True)

# Set the staggered holding period and the breaks
# We use monthly rebalance, where 100% of the portfolio
# is rebalanced each month
# We use quintiles, (5)

holding_period = 1             # holding period returns
n_portf        = 5            # number of portfolios, 5 is quintiles, 10 deciles

# We will set the main return spec to extretnc_bns,
# which are the duation adjusted returns from:
# "Duration-Based Valuation of Corporate Bonds":
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3914422
# Feel free to set this to "bond_ret", which is total bond return,
# if you want this return definition. 
# i.e., use tbl1['ret'] = tbl1['exretn'] or 
#           tbl1['ret'] = tbl1['bond_ret']

tbl1['ret'] = tbl1['exretnc_bns']

# Create some "storage" variables which we will use to store the long-short
# strategy returns, and the portfolio turnover #

# Indpendent of the package, create dataframe to store the sort output #
Strategies_LS = pd.DataFrame() # Long-Short (LS) strategies #
Strategies_L  = pd.DataFrame() # Long (L) strategies #
Strategies_S  = pd.DataFrame() # Short (S) strategies #

LongShortTurn = pd.DataFrame() # Long-Short Turnover (TO)  #
LongTurn      = pd.DataFrame() # Long (L) Turnover (TO)    #
ShortTurn     = pd.DataFrame() # Short (S) Turnover (TO)   #

# I select some variables to sort on here,
# feel free to choose your own, or sort on ALL variables #

Sort_Vars = [
            '18_spread',
            '22_rating',
            '6_duration',
            'seas_1_1na',
            'AnnouncementReturn',
            'AssetGrowth'
             ]

for sort_var1 in Sort_Vars:
    print('Sorting on ' + str(sort_var1))
    
    # initialize the strategy
    single_sort = pbl.SingleSort(holding_period, 
                                 sort_var1, 
                                 n_portf,
                                 )
              
    # parameters
    # we do not conduct any filtering, so we leave most of this blank
    params = {'strategy': single_sort,
              'rating':None, 
              'turnover': True,
    }
    
    # Copy the dataframe for this iteration / sort
    data = tbl1[['date','ID','VW','RATING_NUM',
                 sort_var1,'ret']]  .copy()
    
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
    _out.columns = ['EW_'+ str(sort_var1) + str('_Shrt') , 'VW_'+ str(sort_var1) + str('_Short')]
    Strategies_S  = pd.concat([Strategies_S , _out], axis = 1)
    
    # Extract turnover #
    # Long-Short #
    ew_turnover,vw_turnover = RESULTS .get_ptf_turnover()
    _ls_vw_turn = (vw_turnover.iloc[:,(n_portf-1)]+(vw_turnover.iloc[:,0]))/2
    _ls_ew_turn = (ew_turnover.iloc[:,(n_portf-1)]+(ew_turnover.iloc[:,0]))/2
    _ls_out     = pd.concat([_ls_ew_turn, _ls_vw_turn], axis = 1)
    _ls_out.columns = ['EW_'+ str(sort_var1)+'_TO' , 'VW_'+ str(sort_var1)+'_TO']
    LongShortTurn  = pd.concat([LongShortTurn , _ls_out], axis = 1)
    
    # Long #    
    _l_vw_turn = vw_turnover.iloc[:,(n_portf-1)]
    _l_ew_turn = ew_turnover.iloc[:,(n_portf-1)]
    _l_out     = pd.concat([_l_ew_turn, _l_vw_turn], axis = 1)
    _l_out.columns = ['EW_'+ str(sort_var1)+ str('_Long')+'_TO' ,
                      'VW_'+ str(sort_var1)+ str('_Long')+'_TO']
    LongTurn  = pd.concat([LongTurn , _l_out], axis = 1)
    
    # Short #    
    _s_vw_turn = vw_turnover.iloc[:,0]
    _s_ew_turn = ew_turnover.iloc[:,0]
    _s_out     = pd.concat([_s_ew_turn, _s_vw_turn], axis = 1)
    _s_out.columns = ['EW_'+ str(sort_var1)+ str('_Shrt')+'_TO' ,
                      'VW_'+ str(sort_var1)+ str('_Shrt')+'_TO']
    ShortTurn  = pd.concat([ShortTurn , _s_out], axis = 1)


# Import factors for risk-adjustment #
_url = 'https://openbondassetpricing.com/wp-content/uploads/2023/10/bbw_wrds_oct_2023_lastest.csv'
MKTb = pd.read_csv(_url)
MKTb['date'] = pd.to_datetime(MKTb['date'])
MKTb = MKTb.set_index(['date'])

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
         
RiskPremia = pd.concat([Mean, Tstats], axis = 0).T

print(RiskPremia)

# Alpha #
Alpha       = (pd.DataFrame(np.array(alpha)) ).round(2).T  
Alpha.index = ['Alpha (%)' ]
Alpha.columns= Strategies_LS.columns
TstatsA     = pd.DataFrame(np.array(tstats_alpha)).T
TstatsA.index = [str('t-stat') ]
TstatsA  = "(" + TstatsA.astype(str)  + ")" 
TstatsA .columns= Strategies_LS.columns
         
Alpha = pd.concat([Alpha, TstatsA], axis = 0).T

print(Alpha)

# Inspect Turnover #
# Exclude the final row, because this TO value will be ~100%
# by construction!
print( LongShortTurn.iloc[:-1,:].mean()*100 )

# Examine some cumulative returns #
# Some Long-Short strategies #
s = ['VW_18_spread', 'VW_AssetGrowth']
(1+Strategies_LS[s]).cumprod().plot() 
plt.title('Cumulative Returns of Portfolios', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative ($) Value ', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()
################################ END ##########################################    
