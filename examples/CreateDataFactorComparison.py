# =============================================================================
# Comparing DFPS vs. OSBAP Factors
# =============================================================================

# 1.  Download data from https://openbondassetpricing.com/machine-learning-data/

import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# Load the uploaded files
file_dfps  = r'ExcessLongShortVW_DFPS.csv'
file_osbap = r'ExcessLongShortVW_OSBAP.csv'
file_ice   = r'ExcessLongShortVW_ICE.csv'

df_dfps = pd.read_csv(file_dfps)
df_osbap = pd.read_csv(file_osbap)
df_ice  = pd.read_csv(file_ice)

# Set the index to the 'date' column for both DataFrames
df_dfps.set_index('date', inplace=True)
df_osbap.set_index('date', inplace=True)
df_ice.set_index('date', inplace=True)

# Ensure the indices are datetime for proper alignment and operations
df_dfps.index = pd.to_datetime(df_dfps.index)
df_osbap.index = pd.to_datetime(df_osbap.index)
df_ice.index  = pd.to_datetime(df_ice.index)

# Data is limited by the DFPS data which ends in November 2021
df_osbap = df_osbap[df_osbap.index <= df_dfps.index.max()]
df_ice  = df_ice [df_ice .index <= df_dfps.index.max()]
df_ice  = df_ice [df_ice .index >= df_dfps.index.min()]

# Strip any asterisks (*) from column names in both DataFrames
df_dfps.columns = df_dfps.columns.str.replace('*', '', regex=False)
df_osbap.columns = df_osbap.columns.str.replace('*', '', regex=False)
df_ice.columns  = df_ice. columns.str.replace('*', '', regex=False)

# Compute pairwise correlations across all three datasets
correlations = pd.DataFrame({
    'DFPS_OSBAP': df_dfps.corrwith(df_osbap),
    'DFPS_ICE' : df_dfps.corrwith(df_ice),
    'OSBAP_ICE' : df_osbap.corrwith(df_ice)
})

# Compute means for all columns in each dataset
means = pd.DataFrame({
    'Mean_DFPS': df_dfps.mean()*100,
    'Mean_OSBAP': df_osbap.mean()*100,
    'Mean_ICE': df_ice.mean()  *100
})

# Sharpes
sharpes = pd.DataFrame({
    'SR_DFPS':  df_dfps.mean()/df_dfps.std()   * np.sqrt(12),
    'SR_OSBAP': df_osbap.mean()/df_osbap.std() * np.sqrt(12),
    'SR_ICE':   df_ice.mean()  /df_ice.std()   * np.sqrt(12),
})

# Compute t-tests for the differences in means between datasets
t_test_results_multi = {
    't_stat_DFPS_OSBAP': [],
    'p_value_DFPS_OSBAP': [],
    't_stat_DFPS_ICE': [],
    'p_value_DFPS_ICE': [],
    't_stat_OSBAP_ICE': [],
    'p_value_OSBAP_ICE': []
}

for col in df_dfps.columns:
    if col in df_osbap.columns and col in df_ice.columns:
        # Perform t-tests for each pair of datasets
        t_stat_dw, p_value_dw = ttest_ind(df_dfps[col].dropna(), df_osbap[col].dropna(), nan_policy='omit')
        t_stat_di, p_value_di = ttest_ind(df_dfps[col].dropna(), df_ice[col].dropna(), nan_policy='omit')
        t_stat_wi, p_value_wi = ttest_ind(df_osbap[col].dropna(), df_ice[col].dropna(), nan_policy='omit')
        
        t_test_results_multi['t_stat_DFPS_OSBAP'].append(t_stat_dw)
        t_test_results_multi['p_value_DFPS_OSBAP'].append(p_value_dw)
        t_test_results_multi['t_stat_DFPS_ICE'].append(t_stat_di)
        t_test_results_multi['p_value_DFPS_ICE'].append(p_value_di)
        t_test_results_multi['t_stat_OSBAP_ICE'].append(t_stat_wi)
        t_test_results_multi['p_value_OSBAP_ICE'].append(p_value_wi)
    else:
        # Append NaNs if a column is missing in any dataset
        t_test_results_multi['t_stat_DFPS_OSBAP'].append(None)
        t_test_results_multi['p_value_DFPS_OSBAP'].append(None)
        t_test_results_multi['t_stat_DFPS_ICE'].append(None)
        t_test_results_multi['p_value_DFPS_ICE'].append(None)
        t_test_results_multi['t_stat_OSBAP_ICE'].append(None)
        t_test_results_multi['p_value_OSBAP_ICE'].append(None)

# Create a DataFrame for the t-test results
t_test_df_multi = pd.DataFrame(t_test_results_multi, index=df_dfps.columns)
p_vals = t_test_df_multi .filter(like='p_value')

# Merge to Means and Sharpe Ratios #
data_compare = pd.concat([correlations, means,sharpes, p_vals], axis=1)
data_compare.to_csv(r'DatabaseComparison_DNR_2022.csv')
##################################### END #################################### 
