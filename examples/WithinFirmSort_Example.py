"""
Usage Example: WithinFirmSort Strategy in PyBondLab

This example demonstrates how to use the WithinFirmSort strategy
to construct within-firm factors.

The key innovation: portfolio formation happens within each firm,
isolating within-firm bond dispersion from cross-firm differences.

Author:  Giulio Rossetti
Date: 2025-01-06
"""

import numpy as np
import pandas as pd
import PyBondLab as pbl
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================================================================
# Load Bond Data
# =============================================================================
# This example uses OSBAP (Open Source Bond Asset Pricing) data
# Download: https://openbondassetpricing.com/

# TODO: Update this path to point to your local OSBAP data file
# Download from: https://openbondassetpricing.com/
data_path = "path/to/your/WRDS_MMN_Corrected_Data_2024_July.csv"

print("Loading bond data...")
try:
    tbl1 = pd.read_csv(data_path)
    print(f"Loaded {len(tbl1)} observations")
except FileNotFoundError:
    print(f"Data file not found at: {data_path}")
    print("Please update the path to your bond data file.")
    print("\nFor this example, we'll generate synthetic data instead...")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2020-01-31', '2023-12-31', freq='ME')

    records = []
    cusip_counter = 0

    for firm_id in range(100):  # 100 firms
        n_bonds = np.random.randint(2, 6)

        for bond_id in range(n_bonds):
            cusip = f'CUSIP{cusip_counter:06d}'
            cusip_counter += 1

            for date in dates:
                records.append({
                    'cusip': cusip,
                    'PERMNO': firm_id,  # Firm identifier
                    'date': date,
                    'rating': np.random.randint(1, 21),
                    'CS': np.random.uniform(0.01, 0.08),  # Credit spread
                    'bond_ret': np.random.normal(0.005, 0.02),
                    'BOND_VALUE': np.random.lognormal(10, 1),
                    'BONDPRC': 100 * np.random.uniform(0.8, 1.2),
                })

    tbl1 = pd.DataFrame(records)
    print(f"Generated {len(tbl1)} synthetic observations")

# =============================================================================
# Format the Data
# =============================================================================
print("\nFormatting data...")

# Ensure date is datetime
tbl1['date'] = pd.to_datetime(tbl1['date'])

# Sort by bond identifier and date (REQUIRED by PyBondLab)
tbl1 = tbl1.sort_values(['cusip', 'date'])

# Filter to official start date (for real OSBAP data)
if 'BOND_VALUE' in tbl1.columns:
    tbl1 = tbl1[tbl1['date'] >= "2002-08-31"]

# Create value weight variable
tbl1['VW'] = tbl1.get('BOND_VALUE', tbl1.get('bond_size', 1.0))

# Rename columns to PyBondLab conventions
tbl1.rename(columns={
    "BONDPRC": "PRICE",      # Price
    "cusip": "ID",           # Bond identifier
    "rating": "RATING_NUM",  # Rating
    "bond_ret": "ret"        # Return
}, inplace=True)

# Ensure PERMNO exists (firm identifier)
if 'PERMNO' not in tbl1.columns:
    print("WARNING: PERMNO column not found. WithinFirmSort requires a firm identifier.")
    print("Please add a firm identifier column to your data.")

print(f"Data shape after formatting: {tbl1.shape}")
print(f"Unique bonds: {tbl1['ID'].nunique()}")
print(f"Unique firms: {tbl1.get('PERMNO', pd.Series()).nunique()}")
print(f"Date range: {tbl1['date'].min()} to {tbl1['date'].max()}")

# =============================================================================
# Example 1: Basic Within-Firm Sort on Credit Spread
# =============================================================================
print("\n" + "="*80)
print("Example 1: Within-Firm Sort on Credit Spread")
print("="*80)

# Initialize WithinFirmSort strategy
strategy = pbl.WithinFirmSort(
    holding_period=1,              # Monthly rebalancing
    sort_var='CS',                 # Sort on credit spread
    firm_id_col='PERMNO',          # Firm identifier column
    min_bonds_per_firm=2,          # Require at least 2 bonds per firm
    rating_bins=[-np.inf, 7, 10, np.inf],  # IG+, IG-, SG
    num_portfolios=2,              # High/low portfolios
    verbose=True
)

# Set up parameters
params = {
    'strategy': strategy,
    'rating': None,  # No additional rating filter (using rating_bins in strategy)
}

# Copy data
data = tbl1.copy()

# Fit the strategy
print("\nForming portfolios...")
RESULTS = pbl.StrategyFormation(data, **params).fit()

# Extract long-short returns
ls_returns = pd.concat(RESULTS.get_long_short(), axis=1)
ls_returns.columns = ['EW_CS_WFS', 'VW_CS_WFS']

print("\nLong-Short Returns (first 10 months):")
print(ls_returns.head(10))

# Summary statistics
print("\nSummary Statistics:")
print(f"  Mean return (EW): {ls_returns['EW_CS_WFS'].mean()*100:.3f}%")
print(f"  Mean return (VW): {ls_returns['VW_CS_WFS'].mean()*100:.3f}%")
print(f"  Std dev (EW): {ls_returns['EW_CS_WFS'].std()*100:.3f}%")
print(f"  Std dev (VW): {ls_returns['VW_CS_WFS'].std()*100:.3f}%")
print(f"  Sharpe (EW, annualized): {ls_returns['EW_CS_WFS'].mean() / ls_returns['EW_CS_WFS'].std() * np.sqrt(12):.2f}")
print(f"  Sharpe (VW, annualized): {ls_returns['VW_CS_WFS'].mean() / ls_returns['VW_CS_WFS'].std() * np.sqrt(12):.2f}")

# =============================================================================
# Example 2: Comparing Within-Firm vs Standard Sorting
# =============================================================================
print("\n" + "="*80)
print("Example 2: Within-Firm vs Standard Cross-Sectional Sort")
print("="*80)

# Standard single sort for comparison
standard_sort = pbl.SingleSort(
    holding_period=1,
    sort_var='CS',
    num_portfolios=5,  # Quintiles
    skip=0,
    verbose=False
)

params_standard = {'strategy': standard_sort, 'rating': None}
data_standard = tbl1.copy()

print("Forming standard cross-sectional portfolios...")
RESULTS_STANDARD = pbl.StrategyFormation(data_standard, **params_standard).fit()

# Extract standard long-short
ls_standard = pd.concat(RESULTS_STANDARD.get_long_short(), axis=1)
ls_standard.columns = ['EW_CS_STD', 'VW_CS_STD']

# Compare
comparison = pd.concat([ls_returns, ls_standard], axis=1).dropna()

print("\nComparison: Within-Firm vs Standard Sort")
print(f"{'Metric':<30} {'WFS (VW)':<15} {'Standard (VW)':<15}")
print("-"*60)
print(f"{'Mean return (%/month)':<30} {comparison['VW_CS_WFS'].mean()*100:>14.3f} {comparison['VW_CS_STD'].mean()*100:>14.3f}")
print(f"{'Std dev (%/month)':<30} {comparison['VW_CS_WFS'].std()*100:>14.3f} {comparison['VW_CS_STD'].std()*100:>14.3f}")
print(f"{'Sharpe (annualized)':<30} {comparison['VW_CS_WFS'].mean() / comparison['VW_CS_WFS'].std() * np.sqrt(12):>14.2f} {comparison['VW_CS_STD'].mean() / comparison['VW_CS_STD'].std() * np.sqrt(12):>14.2f}")
print(f"{'Correlation':<30} {comparison[['VW_CS_WFS', 'VW_CS_STD']].corr().iloc[0,1]:>14.3f}")

# =============================================================================
# Example 3: Risk-Adjusted Performance
# =============================================================================
print("\n" + "="*80)
print("Example 3: Risk-Adjusted Performance (CAPM Alpha)")
print("="*80)

# Create market factor (equal-weighted bond market return)
MKTb = tbl1.groupby("date")[['ret']].mean()
MKTb.columns = ['MKTB']

# Merge with strategy returns
perf_data = comparison.merge(MKTb, left_index=True, right_index=True, how='inner')

# Run CAPM regression for within-firm sort
X = sm.add_constant(perf_data['MKTB'])
y = perf_data['VW_CS_WFS']

model_wfs = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

print("\nWithin-Firm Sort (VW) - CAPM Regression:")
print(f"  Alpha: {model_wfs.params[0]*100:.3f}% per month (t={model_wfs.tvalues[0]:.2f})")
print(f"  Beta:  {model_wfs.params[1]:.3f} (t={model_wfs.tvalues[1]:.2f})")
print(f"  R²:    {model_wfs.rsquared:.3f}")

# Run CAPM regression for standard sort
y_std = perf_data['VW_CS_STD']
model_std = sm.OLS(y_std, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

print("\nStandard Sort (VW) - CAPM Regression:")
print(f"  Alpha: {model_std.params[0]*100:.3f}% per month (t={model_std.tvalues[0]:.2f})")
print(f"  Beta:  {model_std.params[1]:.3f} (t={model_std.tvalues[1]:.2f})")
print(f"  R²:    {model_std.rsquared:.3f}")

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "="*80)
print("Creating Plots...")
print("="*80)

# Plot 1: Cumulative returns
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Cumulative returns
ax1 = axes[0]
(1 + comparison[['VW_CS_WFS', 'VW_CS_STD']]).cumprod().plot(
    ax=ax1, linewidth=2
)
ax1.set_title('Cumulative Returns: Within-Firm vs Standard Sort', fontsize=14)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative Value ($)', fontsize=12)
ax1.legend(['Within-Firm Sort', 'Standard Sort'], fontsize=11)
ax1.grid(True, alpha=0.3)

# Rolling Sharpe ratios (12-month window)
ax2 = axes[1]
rolling_window = 12
rolling_sharpe_wfs = (
    comparison['VW_CS_WFS'].rolling(rolling_window).mean() /
    comparison['VW_CS_WFS'].rolling(rolling_window).std() *
    np.sqrt(12)
)
rolling_sharpe_std = (
    comparison['VW_CS_STD'].rolling(rolling_window).mean() /
    comparison['VW_CS_STD'].rolling(rolling_window).std() *
    np.sqrt(12)
)

pd.DataFrame({
    'Within-Firm': rolling_sharpe_wfs,
    'Standard': rolling_sharpe_std
}).plot(ax=ax2, linewidth=2)

ax2.set_title(f'Rolling {rolling_window}-Month Sharpe Ratio', fontsize=14)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Sharpe Ratio', fontsize=12)
ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('withinfirmsort_example.png', dpi=150, bbox_inches='tight')
print("Saved plot to: withinfirmsort_example.png")

plt.show()

print("\n" + "="*80)
print("Example Complete!")
print("="*80)
