"""
00 - Pre-Analysis Descriptive Statistics
=========================================
Demonstrates PreAnalysisStats for computing cross-sectional and
time-series summary statistics of bond characteristics prior to
portfolio formation.

PyBondLab features used:
  - PreAnalysisStats            (descriptive statistics engine)
  - PreAnalysisResult           (result container)
  - summary()                   (aggregated statistics)
  - get_cs_stats()              (cross-sectional stats per period)
  - get_ts_stats()              (time-series aggregation)
  - rating filter               (IG/NIG subsample)
  - subset_filter               (maturity/size subsets)
  - percentiles                 (custom quantile selection)
  - validate_panel              (panel validation)
  - check_duplicates            (duplicate detection)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from _config import load_panel

from PyBondLab import PreAnalysisStats, validate_panel, check_duplicates

data = load_panel()
print(f"Panel: {len(data):,} obs")

# =============================================================================
# 0. Panel validation checks
# =============================================================================
validate_panel(data, date_col='date', id_col='ID')
dups = check_duplicates(data, date_col='date', id_col='ID')
print(f"Duplicates found: {dups}")

# =============================================================================
# 1. Basic pre-analysis: summary statistics for key variables
# =============================================================================
variables = ['ret', 'cs', 'tmat', 'sze', 'ami', 'dvol']

pre = PreAnalysisStats(data=data, variables=variables, date_col='date', id_col='ID')
result = pre.compute()

print("\n-- Summary statistics (full sample) --")
for var in variables:
    print(f"\n  {var}:")
    print(f"  {result.summary(var).to_string()}")

# =============================================================================
# 2. Custom percentiles
# =============================================================================
result_pct = PreAnalysisStats(
    data=data, variables=['cs', 'tmat', 'sze'],
    date_col='date', id_col='ID',
    percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99],
).compute()
print("\n-- Summary with extended percentiles --")
print(result_pct.summary('cs'))

# =============================================================================
# 3. Cross-sectional stats over time
# =============================================================================
cs_stats = result.get_cs_stats('cs')
print(f"\n-- Cross-sectional stats shape: {cs_stats.shape} (months x stats) --")
print(cs_stats.tail())

# =============================================================================
# 4. Time-series stats
# =============================================================================
ts_stats = result.get_ts_stats('cs')
print(f"\n-- Time-series aggregation --")
print(ts_stats)

# =============================================================================
# 5. IG-only subsample
# =============================================================================
result_ig = PreAnalysisStats(
    data=data, variables=['ret', 'cs', 'tmat'],
    date_col='date', id_col='ID',
    rating='IG', rating_col='RATING_NUM',
).compute()
print("\n-- IG bonds only --")
print(result_ig.summary('ret'))

# =============================================================================
# 6. NIG (high-yield) subsample
# =============================================================================
result_nig = PreAnalysisStats(
    data=data, variables=['ret', 'cs', 'tmat'],
    date_col='date', id_col='ID',
    rating='NIG', rating_col='RATING_NUM',
).compute()
print("\n-- NIG (high-yield) bonds only --")
print(result_nig.summary('ret'))

# =============================================================================
# 7. Subset filter: short maturity bonds only
# =============================================================================
result_short = PreAnalysisStats(
    data=data, variables=['ret', 'cs', 'sze'],
    date_col='date', id_col='ID',
    subset_filter={'tmat': (0, 5)},
).compute()
print("\n-- Short maturity (0-5y) --")
print(result_short.summary('ret'))

# =============================================================================
# 8. With issuer-level counts
# =============================================================================
if 'ticker' not in data.columns:
    data['ticker'] = data['ID'].str[:6]

result_issuer = PreAnalysisStats(
    data=data, variables=['ret', 'cs'],
    date_col='date', id_col='ID', issuer_col='ticker',
).compute()
print("\n-- With issuer-level counts --")
print(result_issuer.summary('ret'))

# =============================================================================
# 9. Multiple variables summary comparison
# =============================================================================
print("\n-- All variables summary --")
for var in variables:
    s = result.summary(var)
    print(f"\n  {var}:")
    print(f"  {s.to_string()}")

print("\n[Done] Script 00 complete.")
