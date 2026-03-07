"""
07 - Data Uncertainty Analysis
================================
Demonstrates DataUncertaintyAnalysis for testing sensitivity of
factor premia to data filtering choices: price filters, return
trimming, winsorization, and bounce-back exclusion.

This replicates the DUA analysis from Dickerson, Robotti & Rossetti
using the exact filter grids from the paper.

PyBondLab features used:
  - DataUncertaintyAnalysis     (main DUA engine)
  - DataUncertaintyResults      (result container)
  - .fit(IDvar, RETvar, ...)    (column mapping at fit time)
  - summary()                   (NW t-stats, means in %)
  - aggregate_by                (grouping in summary)
  - filters dict                (trim, price, bounce, wins)
  - ratings                     (IG/NIG/all subsamples)
  - holding_periods             (multiple HPs)
  - include_baseline            (no-filter baseline)
  - use_fast_path               (numba acceleration)
  - ew_ex_ante / vw_ex_ante     (EA returns)
  - ew_ex_post / vw_ex_post     (EP returns with filters)
  - ew_long_ex_ante             (long leg EA)
  - ew_short_ex_ante            (short leg EA)
  - configs                     (filter configuration log)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from _config import load_panel

from PyBondLab import DataUncertaintyAnalysis

data = load_panel()

# Price column needed for price filters (derived from book-to-market)
if 'PRICE' not in data.columns and 'bbtm' in data.columns:
    data['PRICE'] = 100 / data['bbtm']

print(f"Panel: {len(data):,} obs")

# =============================================================================
# Filter grids from DRR paper
# =============================================================================

# Trim: symmetric return trimming at various percentile thresholds
trim_levels = [round(w, 4) for w in np.arange(0.20, 1.00, 0.05)]

# Price: exclude bonds with extreme prices (left = too low, right = too high)
price_levels_down = list(np.arange(20, 0, -2))
price_levels_up   = list(np.arange(150, 300, 15))

# Bounce-back: exclude return reversals exceeding threshold
bounce_levels = [round(w, 4) for w in np.arange(0.01, 0.11, 0.01)]

# Winsorization: clip returns at percentile thresholds
wins_levels = [(w, loc)
               for w in np.arange(98.0, 99.8, 0.5)
               for loc in ['right', 'left', 'both']]

# Full filter dictionary (all types)
filters_full = {
    'trim': trim_levels,
    'price': [price_levels_down, price_levels_up],
    'bounce': bounce_levels,
    'wins': wins_levels,
}

# Smaller grid for quick demonstration
filters_small = {
    'trim': [0.5, 1.0],
    'price': [[1, 5], [150, 200]],
    'bounce': [0.05, 0.10],
    'wins': [(99, 'both'), (95, 'both')],
}

print(f"Full grid: trim={len(trim_levels)}, price={len(price_levels_down)+len(price_levels_up)}, "
      f"bounce={len(bounce_levels)}, wins={len(wins_levels)}")

# =============================================================================
# 1. DUA with small filter grid (quick demo)
# =============================================================================
dua = DataUncertaintyAnalysis(
    data=data, signals=['mom6_1'],
    holding_periods=[1, 3],
    filters=filters_small,
    ratings=[None, 'IG', 'NIG'],
    num_portfolios=5,
    dynamic_weights=True, include_baseline=True, use_fast_path=True, verbose=True,
)
results = dua.fit(IDvar='ID', RETvar='ret', VWvar='VW', RATINGvar='RATING_NUM', PRICEvar='PRICE')

print("\n-- DUA: momentum (6,1) - small grid --")
print(results.summary())

# =============================================================================
# 2. Access EA and EP returns directly
# =============================================================================
print(f"\n-- Result attributes --")
print(f"EW EA shape: {results.ew_ex_ante.shape}")
print(f"VW EA shape: {results.vw_ex_ante.shape}")
print(f"EW EP shape: {results.ew_ex_post.shape}")
print(f"VW EP shape: {results.vw_ex_post.shape}")

print(f"EW Long  EA shape: {results.ew_long_ex_ante.shape}")
print(f"EW Short EA shape: {results.ew_short_ex_ante.shape}")
print(f"VW Long  EP shape: {results.vw_long_ex_post.shape}")
print(f"VW Short EP shape: {results.vw_short_ex_post.shape}")

print(f"\nConfigs shape: {results.configs.shape}")
print(results.configs.head())

# =============================================================================
# 3. Summary aggregated by holding period
# =============================================================================
print("\n-- Summary by holding period --")
print(results.summary(aggregate_by='hp'))

# =============================================================================
# 4. Summary aggregated by rating
# =============================================================================
print("\n-- Summary by rating --")
print(results.summary(aggregate_by='rating'))

# =============================================================================
# 5. Multi-signal DUA (5 signals, HP=1, deciles)
# =============================================================================
results_multi = DataUncertaintyAnalysis(
    data=data, signals=['mom6_1', 'cs', 'ami', 'dvol', 'val_ipr'],
    holding_periods=[1], filters=filters_small,
    ratings=[None], num_portfolios=10,
    dynamic_weights=True, include_baseline=True, use_fast_path=True, verbose=True,
).fit(IDvar='ID', RETvar='ret', VWvar='VW', RATINGvar='RATING_NUM', PRICEvar='PRICE')
print("\n-- Multi-signal DUA (5 signals, deciles) --")
print(results_multi.summary())

# =============================================================================
# 6. DUA with subset_filter (short maturity only)
# =============================================================================
results_short = DataUncertaintyAnalysis(
    data=data, signals=['cs'],
    holding_periods=[1],
    filters={'trim': [0.5, 1.0], 'price': [[1, 5], [150, 200]]},
    ratings=[None, 'IG'], num_portfolios=5,
    subset_filter={'tmat': (0, 5)},
    dynamic_weights=True, include_baseline=True, verbose=True,
).fit(IDvar='ID', RETvar='ret', VWvar='VW', RATINGvar='RATING_NUM', PRICEvar='PRICE')
print("\n-- DUA: credit spread (short maturity 0-5y) --")
print(results_short.summary())

# =============================================================================
# 7. Comparing EA vs EP effects
# =============================================================================
ea_means = results.ew_ex_ante.mean()
ep_means = results.ew_ex_post.mean()
print("\n-- EA vs EP comparison (EW, mom 6,1) --")
comparison = pd.DataFrame({
    'EA_mean_%': ea_means * 100,
    'EP_mean_%': ep_means * 100,
    'diff_%': (ep_means - ea_means) * 100,
})
print(comparison.head(10))

# =============================================================================
# 8. Full DRR filter grid (all bonds, HP=1, single signal)
#    WARNING: This runs ~120 filter configurations - takes time
# =============================================================================
RUN_FULL_GRID = False  # Set to True to replicate exact DRR filters

if RUN_FULL_GRID:
    results_full = DataUncertaintyAnalysis(
        data=data, signals=['mom6_1'],
        holding_periods=[1], filters=filters_full,
        ratings=[None], num_portfolios=10,
        dynamic_weights=True, include_baseline=True, use_fast_path=True, verbose=True,
    ).fit(IDvar='ID', RETvar='ret', VWvar='VW', RATINGvar='RATING_NUM', PRICEvar='PRICE')
    print("\n-- Full DRR filter grid: momentum (6,1) --")
    print(results_full.summary())

print("\n[Done] Script 07 complete.")
