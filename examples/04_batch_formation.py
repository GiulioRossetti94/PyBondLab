"""
04 - Batch Portfolio Formation & Panel Extraction
==================================================
Demonstrates BatchStrategyFormation for processing multiple signals
simultaneously, with rating/maturity filtering, and extract_panel()
for building standardized factor panels.

PyBondLab features used:
  - BatchStrategyFormation      (multi-signal batch processing)
  - BatchResults                (result container with dict access)
  - NamingConfig                (factor naming conventions)
  - extract_panel               (unified panel extraction)
  - rating filter               ('IG', 'NIG')
  - subset_filter               (maturity subsetting)
  - get_factor_returns()        (wide-format factor returns)
  - summary_df                  (batch summary statistics)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from _config import load_panel, SIGNALS

from PyBondLab import BatchStrategyFormation, NamingConfig, extract_panel

data = load_panel()
print(f"Panel: {len(data):,} obs, signals: {SIGNALS}")

# =============================================================================
# 1. Basic batch processing (10 signals, all bonds)
# =============================================================================
results = BatchStrategyFormation(
    data=data, signals=SIGNALS,
    holding_period=1, num_portfolios=5,
    turnover=False, dynamic_weights=True, verbose=True,
).fit()

print(f"\n-- Batch results: {len(results)} signals --")
print(f"Successful: {results.successful_signals}")
if results.failed_signals:
    print(f"Failed: {results.failed_signals}")

for sig in results.successful_signals[:5]:
    ew, vw = results[sig].get_long_short()
    t_vw = vw.mean() / vw.std() * np.sqrt(len(vw))
    print(f"  {sig:10s}  VW={vw.mean()*100:+.3f}%/mo (t={t_vw:+.2f})")

# =============================================================================
# 2. Wide-format factor returns
# =============================================================================
ew_factors = results.get_factor_returns(weight_type='ew')
vw_factors = results.get_factor_returns(weight_type='vw')
print(f"\nFactor returns shape: {vw_factors.shape}  (months x signals)")

# =============================================================================
# 3. Wide-format factor returns (available from fast path)
# =============================================================================

# =============================================================================
# 4. Batch with rating filter (IG only)
# =============================================================================
results_ig = BatchStrategyFormation(
    data=data, signals=SIGNALS[:5],
    holding_period=1, num_portfolios=5,
    turnover=False, rating='IG', dynamic_weights=True, verbose=False,
).fit()

print(f"\n-- IG-only batch --")
for sig in results_ig.successful_signals:
    ew, vw = results_ig[sig].get_long_short()
    t_vw = vw.mean() / vw.std() * np.sqrt(len(vw))
    print(f"  {sig:10s}  VW={vw.mean()*100:+.3f}%/mo (t={t_vw:+.2f})")

# =============================================================================
# 5. Batch with subset_filter (short maturity)
# =============================================================================
results_short = BatchStrategyFormation(
    data=data, signals=SIGNALS[:5],
    holding_period=1, num_portfolios=5,
    turnover=False, subset_filter={'tmat': (0, 5)},
    dynamic_weights=True, verbose=False,
).fit()

print(f"\n-- Short maturity (0-5y) batch --")
for sig in results_short.successful_signals:
    ew, vw = results_short[sig].get_long_short()
    t_vw = vw.mean() / vw.std() * np.sqrt(len(vw))
    print(f"  {sig:10s}  VW={vw.mean()*100:+.3f}%/mo (t={t_vw:+.2f})")

# =============================================================================
# 6. Batch with turnover and characteristics
# =============================================================================
results_full = BatchStrategyFormation(
    data=data, signals=SIGNALS[:3],
    holding_period=1, num_portfolios=5,
    turnover=True, chars=['tmat', 'cs'],
    dynamic_weights=True, verbose=False,
).fit()

print(f"\n-- Batch with turnover & chars --")
for sig in results_full.successful_signals:
    ew, vw = results_full[sig].get_long_short()
    ew_to, _ = results_full[sig].get_turnover()
    t_vw = vw.mean() / vw.std() * np.sqrt(len(vw))
    print(f"  {sig:10s}  VW={vw.mean()*100:+.3f}%/mo (t={t_vw:+.2f})  "
          f"avg TO={ew_to.mean().mean()*100:.1f}%")

naming = NamingConfig(lowercase=True, sign_correct=True, use_signal_name=True)
panel = extract_panel(results_full, naming=naming)
print(f"Extracted panel shape: {panel.shape}")
print(f"Columns: {list(panel.columns)}")

# =============================================================================
# 7. Batch with staggered holding period (HP=3)
# =============================================================================
results_hp3 = BatchStrategyFormation(
    data=data, signals=SIGNALS[:5],
    holding_period=3, num_portfolios=5,
    turnover=False, dynamic_weights=True, verbose=False,
).fit()

print(f"\n-- HP=3 (staggered cohorts) batch --")
for sig in results_hp3.successful_signals:
    ew, vw = results_hp3[sig].get_long_short()
    t_vw = vw.mean() / vw.std() * np.sqrt(len(vw))
    print(f"  {sig:10s}  VW={vw.mean()*100:+.3f}%/mo (t={t_vw:+.2f})")

# =============================================================================
# 8. Batch with banding
# =============================================================================
results_band = BatchStrategyFormation(
    data=data, signals=SIGNALS[:3],
    holding_period=1, num_portfolios=5,
    turnover=True, banding=1, dynamic_weights=True, verbose=False,
).fit()

print(f"\n-- Batch with banding=1 --")
for sig in results_band.successful_signals:
    ew, vw = results_band[sig].get_long_short()
    ew_to, _ = results_band[sig].get_turnover()
    print(f"  {sig:10s}  VW={vw.mean()*100:+.3f}%/mo  "
          f"avg TO={ew_to.mean().mean()*100:.1f}%")

# =============================================================================
# 9. extract_panel with naming customization
# =============================================================================
naming_custom = NamingConfig(lowercase=True, sign_correct=False,
                             weighting_prefix=True, include_rating_suffix=True)
panel_custom = extract_panel(results_full, naming=naming_custom)
print(f"\nCustom naming panel columns: {list(panel_custom.columns)}")

print("\n[Done] Script 04 complete.")
