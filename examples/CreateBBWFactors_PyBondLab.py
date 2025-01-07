# This script constructs double sorts #
# It creates the BBW 4-factors with the OSBAP data #
# It computes the turnover #
# It then computes the net of cost performance assuming 19 bps half-spread #
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
# Choose either 'bond_amount_out' or BOND_VALUE
tbl1['VW'] = tbl1['bond_amount_out']

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
                     "bond_ret":"ret"},inplace=True) # Rename bond_ret to ret

# i.e., if you want to use duration-adjusted returns, rename 
# 'exretnc_bns' to 'ret'

# i.e., if your price, bond id and bond return variables were called:
#               PRC, BOND_ID and BOND_RET, we would have:
# tbl1.rename(columns={"PRC":"PRICE","BOND_ID":"ID","BOND_RET":"ret"},inplace=True)

# Creat new variable called 'strev', which is MMN-adjusted bond return
tbl1['strev'] = tbl1['BOND_RET']

#==============================================================================
#   USAGE: Double Sorts
#==============================================================================
# Play around with the numbers below
# set holding period to: 1 (monthly rebalance)
#                        3 (quarterly)
#                        6 (semi-annual)
#                       12 (annual)
# The portfolio returns are computed with the overlapping methodology
# of Jegadeesh and Titman (1993) when holding_period > 1
holding_period = 1            # holding period returns
n_portf        = 5            # number of portfolios, 5 is quintiles, 10 deciles
n_portf2       = 5            # number of portfolios for the second sort
skip           = 0            # no skip period, i.e., observe at signal at t
                              # and trade in signal at t (no delay in trade)
b              = 0            # No banding by default (see Novy-Marx and Velikov's RFS paper)

# Initialize the strategies (BBW-related)
cols      = ['var95','ILLIQ','strev']
sort_var  = 'RATING_NUM'

# Returns #
factors     = pd.DataFrame()
factors_crf = pd.DataFrame()

# Turnover #
factors_to     = pd.DataFrame()
factors_crf_to = pd.DataFrame()

for i,s in enumerate(cols):
    print(s)  
    sort_var2 = s
        
    if s == 'illiq':        
        data = tbl1[['date','ID','VW','RATING_NUM',sort_var2,'ret','n_trades_month']].copy()
        data = data[data['n_trades_month'] >=5]
    else:
        data = tbl1[['date','ID','VW','RATING_NUM',sort_var2,'ret']].copy()#.dropna()
            
    double_sort = pbl.DoubleSort(holding_period,
                                 sort_var,
                                 n_portf,
                                 sort_var2,
                                 n_portf2,
                                 how = 'unconditional')
    
    # parameters
    # we do not conduct any filtering, so we leave most of this blank
    params = {'strategy': double_sort,
                'rating':None, 
                'banding_threshold': b,
                'turnover': True
    }


    # Fit the strategy to the data. 
    RESULTS = pbl.StrategyFormation(data, **params).fit()
    
    # Get pfolios #
    sorts = RESULTS.get_ptf()[1]
    ew_turn,vw_turn = RESULTS .get_ptf_turnover()
    
    # rename vw_turnover columns to be the same as sorts #
    vw_turn.columns = sorts.columns
        
    # Across sort variable k #    
    subs_dict    = {}
    subs_dict_to = {}
    for i in range(1, (n_portf+1)):  # Adjust range as needed for arbitrary iterations
        column_label = f'RATING_NUM{i}_DIFF'
        subs_dict[column_label] = sorts.loc[:, f'RATING_NUM{i}_' + str(s.upper() + str(n_portf))] - (
                                  sorts.loc[:, f'RATING_NUM{i}_' + str(s.upper() + '1')] ) 
        subs_dict_to[column_label] = (vw_turn.loc[:, f'RATING_NUM{i}_' + str(s.upper() + str(n_portf))] +\
                                      vw_turn.loc[:, f'RATING_NUM{i}_' + str(s.upper() + '1')])/2 
    subs_df    = pd.DataFrame(subs_dict)
    subs_df_to = pd.DataFrame(subs_dict_to)
    
    _factor = subs_df.mean(axis = 1).to_frame() 
    _factor.columns = [s]
    
    _factor_to = subs_df_to.mean(axis = 1).to_frame() 
    _factor_to.columns = [s]
       
    # Across rating for CRF #
    subs_dict = {}
    subs_dict_to = {}
    for i in range(1, (n_portf+1)):  # Adjust range as needed for arbitrary iterations
        column_label = f'RATING_NUM{i}_DIFF'
        subs_dict[column_label] = sorts.loc[:, 'RATING_NUM'   +str(n_portf)+'_' + str(s.upper() + f'{i}')] -\
                                  sorts.loc[:, 'RATING_NUM1_' + str(s.upper() + f'{i}')] 
        subs_dict_to[column_label] = (vw_turn.loc[:, 'RATING_NUM'   + str(n_portf)+'_' + str(s.upper() + f'{i}')] +\
                                     vw_turn.loc[:, 'RATING_NUM1_' + str(s.upper() + f'{i}')])/2                          
    subs_df = pd.DataFrame(subs_dict)
    subs_df_to = pd.DataFrame(subs_dict_to)
    
    _factor_crf = subs_df.mean(axis = 1).to_frame() 
    _factor_crf.columns = [s+'_crf']
    
    _factor_to_crf = subs_df_to.mean(axis = 1).to_frame() 
    _factor_to_crf.columns = [s+'_crf']
             
    factors     = pd.concat([factors    , _factor], axis = 1)
    factors_crf = pd.concat([factors_crf, _factor_crf], axis = 1)
    
    factors_to     = pd.concat([factors_to    , _factor_to], axis = 1)
    factors_crf_to = pd.concat([factors_crf_to, _factor_to_crf], axis = 1)
    
    
    
# Sign-correct the reversal factor #     
factors['strev']   =   factors['strev']*-1
print(factors.mean()*100)

# Create the "CRF" BBW factor #
CRFx = factors_crf.mean(axis = 1).to_frame()
CRFx.columns = ['CRF']

factors.drop(['strev'], axis = 1, inplace = True)
factors.columns = ['DRF','LRF']

# Create the MKTB factor #
data = tbl1[['date','ID','VW','ret']].copy()#.dropna()
data['VW'] = data.groupby(["ID"])['VW'].shift(1)
data = data.dropna()
data['value-weights'] = data.groupby(by = 'date',group_keys=False )['VW']\
    .progress_apply( lambda x: x/np.nansum(x) )
    
MKTx = data.groupby('date')[['ret',
                            'value-weights']].\
    apply( lambda x: np.nansum( x['ret'] * x['value-weights']) ).to_frame()
    
MKTx.columns = ['MKTB']
rfr = pd.read_csv('rfr.csv')
rfr['date'] = pd.to_datetime(rfr['date'])
rfr = rfr.set_index('date')
MKTx.index = pd.to_datetime(MKTx.index )
rfr.index  = pd.to_datetime(rfr.index )
MKTx = MKTx.merge(rfr, how = "inner", left_index = True, right_index = True)
MKTx = MKTx['MKTB'] - MKTx['RF']
MKTx = MKTx.to_frame()
MKTx.columns = ['MKTB']

bbw_factors = factors.merge(CRFx, how = "inner", left_index = True, right_index = True)
bbw_factors = (
    bbw_factors
    .reset_index()
    .rename(columns={'index': 'date'})
    .set_index('date')
)

MKTx.index = pd.to_datetime(MKTx.index )
bbw_factors.index = pd.to_datetime(bbw_factors.index )

bbw_factors = bbw_factors.merge(MKTx, how = "inner", left_index = True, right_index = True)
bbw_factors = bbw_factors[['MKTB','DRF','CRF','LRF']]

bbw_factors.index = pd.to_datetime(bbw_factors.index)

# Create the set of turnovers #
CRFto = factors_crf_to.mean(axis = 1).to_frame()
CRFto.columns = ['CRF']

factors_to.drop(['strev'], axis = 1, inplace = True)
factors_to.columns = ['DRF','LRF']

bbw_turnover = factors_to.merge(CRFto, how = "inner", left_index=True, right_index=True)
bbw_turnover['MKTB'] = 0
bbw_turnover = bbw_turnover[['MKTB','DRF','CRF','LRF']]
bbw_turnover.index.rename('date',inplace = True)

# Compute net of cost BBW factors, assuming 19 bps half-spread as in
# Kelly, Palhares and Pruitt JF paper #
bbw_factors_net = bbw_factors - bbw_turnover.shift(1)*19/10000 * 2 # Scale by 2 for L and S leg.
print( bbw_factors_net.mean()*100 ) # Only DRF has a +ve mean.

#### Compare with the JFE Factors ####
_url = 'https://openbondassetpricing.com/wp-content/uploads/2023/10/bbw_wrds_oct_2023_lastest.csv'
dmr = pd.read_csv(_url)
dmr['date'] = pd.to_datetime(dmr['date'])
dmr = dmr.set_index('date')
dmr = dmr[['MKTB','DRF','CRF','LRF']]
dmr.columns = [f"{col}jfe" for col in dmr.columns]

bbw_factors.index = pd.to_datetime(bbw_factors.index)
dmr.index = pd.to_datetime(dmr.index)

merged_factors  = bbw_factors .merge(dmr, how = "right", left_index = True, right_index = True).dropna()

#### Compare correlations ####
base_columns = ['MKTB', 'DRF', 'CRF', 'LRF']
jfe_columns = ['MKTBjfe', 'DRFjfe', 'CRFjfe', 'LRFjfe']

# Compute pairwise correlations
correlations = {
    f"{base} & {jfe}": merged_factors[base].corr(merged_factors[jfe])
    for base, jfe in zip(base_columns, jfe_columns)
}

# Convert the results into a DataFrame for better readability
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', 
                                        columns=['Correlation'])

print(correlation_df)

# Cum. returns #
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# Plot each pair in the respective subplot
for i, (base, jfe) in enumerate(zip(base_columns, jfe_columns)):
    row, col = divmod(i, 2)
    axs[row, col].plot(merged_factors[base].cumsum(), label=base, linewidth=1.5)
    axs[row, col].plot(merged_factors[jfe].cumsum(), label=jfe, linewidth=1.5)
    axs[row, col].set_title(f'Cum. ret: {base} & {jfe}')
    axs[row, col].legend()
    axs[row, col].grid()

# Adjust layout
plt.tight_layout()
plt.show()

# Descriptive statistics #
stats = {
    "Mean": (merged_factors[base_columns + jfe_columns].mean() * 100).round(3),
    "StdDev": (merged_factors[base_columns + jfe_columns].std() * 100).round(3),
    "Sharpe Ratio": (merged_factors[base_columns + jfe_columns].mean() / merged_factors[base_columns + jfe_columns].std()).round(3)
}

# Create a DataFrame for comparison
stats_table = pd.DataFrame(stats)

print(stats_table)
########## END ##########
