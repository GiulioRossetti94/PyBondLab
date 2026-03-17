"""
06 - Anomaly Assaying & Methodological Uncertainty Analysis
=============================================================
Demonstrates AssayAnomaly (slow path with bond counts),
assay_anomaly_fast (numba-accelerated single-signal), and
BatchAssayAnomaly (multi-signal batch) for robustness testing
across specification grids.

This replicates the MUA analysis from Dickerson, Robotti & Rossetti:
for each signal, vary weighting, number of portfolios, breakpoint
scheme, breakpoint universe, rating filter, and maturity filter.

PyBondLab features used:
  - AssayAnomaly                (convenience facade, slow path)
  - AssayAnomalyRunner          (full runner with save_idx)
  - AnomalyResults              (.df, .summary_results())
  - assay_anomaly_fast          (numba fast path, single signal)
  - AnomalyAssayResult          (.summary(), .returns_df)
  - BatchAssayAnomaly           (multi-signal batch)
  - BatchAssayResults           (dict-like access per signal)
  - validate_specs              (specification validation)
  - SpecificationValidator      (validator class)
  - generate_spec_list          (spec grid generation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from _config import load_panel, load_mktb, SIGNALS, ASSAY_COLUMNS, BATCH_ASSAY_COLUMNS

from PyBondLab import AssayAnomaly, assay_anomaly_fast, BatchAssayAnomaly, SingleSort
from PyBondLab.AnomalyAssayer import AssayAnomalyRunner  # internal; not in public __all__
from PyBondLab.spec_validator import (
    validate_specs, SpecificationValidator, generate_spec_list, get_valid_spec_list,
)

data = load_panel()
mktb = load_mktb()
print(f"Panel: {len(data):,} obs")

# =============================================================================
# Specification grid (Tier 2 from DRR paper)
# =============================================================================
def bp_filter_ig_only(df):
    """IG-only breakpoints (RATING_NUM 1-10)."""
    return (df['RATING_NUM'] >= 1) & (df['RATING_NUM'] <= 10)
bp_filter_ig_only.required_columns = ['RATING_NUM']

def bp_filter_large_only(df):
    """Large bonds (VW > $100M) for breakpoints."""
    return df['VW'] > 100
bp_filter_large_only.required_columns = ['VW']

TIER2_SPECS = {
    'weighting': ['EW', 'VW'],
    'portfolio_structures': [
        (3, 'T', None),
        (3, 'E10_90', [10, 90]),
        (5, 'Q', None),
        (10, 'D', None),
    ],
    'rating_filters': {'all': None, 'ig': 'IG', 'nig': 'NIG'},
    'bp_universes': {
        'all': None,
        'ig_bp': bp_filter_ig_only,
        'lg_bp': bp_filter_large_only,
    },
    'maturity_filters': {'all': None, 'short': (0, 5), 'mid': (5, 10), 'long': (10, 100)},
}

n_specs = (len(TIER2_SPECS['weighting']) * len(TIER2_SPECS['portfolio_structures']) *
           len(TIER2_SPECS['rating_filters']) * len(TIER2_SPECS['bp_universes']) *
           len(TIER2_SPECS['maturity_filters']))
print(f"Tier 2 spec grid: {n_specs} configurations per signal")

# =============================================================================
# 1. Validate specification grid
# =============================================================================
validator = SpecificationValidator(verbose=True)
validation = validator.validate(TIER2_SPECS, data=data, rating_col='RATING_NUM')
print(f"\n-- Specification validation --")
print(f"Valid: {validation.is_valid}  |  Total: {validation.total_specs}  |  "
      f"Errors: {validation.error_specs}  |  Warnings: {validation.warning_specs}")

val_result = validate_specs(TIER2_SPECS, data=data, rating_col='RATING_NUM')
print(f"Function validation: valid={val_result.is_valid}")

spec_list = generate_spec_list(TIER2_SPECS)
print(f"Spec list length: {len(spec_list)}")

valid_specs = get_valid_spec_list(TIER2_SPECS, data=data, rating_col='RATING_NUM')
print(f"Valid specs: {len(valid_specs)}")

# =============================================================================
# 2. assay_anomaly_fast - single signal across full grid
# =============================================================================
result_fast = assay_anomaly_fast(
    data=data, signal='cs', specs=TIER2_SPECS,
    holding_period=1, dynamic_weights=True, skip_invalid=True, verbose=True,
    **BATCH_ASSAY_COLUMNS,
)

summary_fast = result_fast.summary()
print(f"\n-- assay_anomaly_fast: credit spread --")
print(f"Specs computed: {len(summary_fast)}")
print(summary_fast[['spec_id', 'mean_ret', 't_stat']].head(10).to_string(index=False))

returns_df = result_fast.returns_df
print(f"Returns shape: {returns_df.shape}  (months x specs)")

# =============================================================================
# 3. BatchAssayAnomaly - multiple signals in parallel
# =============================================================================
batch_signals = SIGNALS[:5]
batch_results = BatchAssayAnomaly(
    data=data, signals=batch_signals, specs=TIER2_SPECS,
    holding_period=1, dynamic_weights=True, skip_invalid=True,
    n_jobs=1, verbose=True,
    **BATCH_ASSAY_COLUMNS,
).fit()

print(f"\n-- BatchAssayAnomaly: {len(batch_signals)} signals --")
print(f"Successful: {batch_results.successful_signals}")
if batch_results.failed_signals:
    print(f"Failed: {batch_results.failed_signals}")

batch_summary = batch_results.summary_df
print(f"Total rows in summary: {len(batch_summary)}")

for sig in batch_results.successful_signals:
    sig_summary = batch_results[sig].summary()
    print(f"  {sig}: {len(sig_summary)} specs, "
          f"median t-stat = {sig_summary['t_stat'].median():.2f}")

# =============================================================================
# 4. AssayAnomaly - slow path with bond counts and turnover
# =============================================================================
assay_results = AssayAnomaly(
    data=data, sort_var='mom6_1',
    subset_filter={
        'tmat': [(0, float('inf')), (0, 5), (5, 10), (10, float('inf'))],
    },
    holding_periods=[1], nport=[5, 10],
    ratings=[None, 'IG', 'NIG'],
    dynamic_weights=True, turnover=True, save_idx=True, verbose=True,
    **ASSAY_COLUMNS,
)

ts_df = assay_results.df
print(f"\n-- AssayAnomaly (slow path): momentum --")
print(f"Time series shape: {ts_df.shape}")
print(f"Columns: {list(ts_df.columns[:10])}...")

full_results, recap = assay_results.summary_results(factor=mktb, nw_lag=3)
print(f"\nFull results shape: {full_results.shape}")
print(f"Recap by type:\n{recap}")

# =============================================================================
# 5. AssayAnomalyRunner with custom breakpoint universe
# =============================================================================
runner_results = AssayAnomalyRunner(
    strategy=SingleSort(sort_var='dvol', holding_period=1, num_portfolios=5,
                        breakpoint_universe_func=bp_filter_ig_only, verbose=False),
    data=data,
    holding_periods=[1], nport=[3, 5],
    ratings=[None, 'IG'],
    subset_filter={'tmat': [(0, float('inf')), (0, 5)]},
    breakpoint_universe_func=bp_filter_ig_only,
    dynamic_weights=True, turnover=False, save_idx=True, n_jobs=1, verbose=True,
    **ASSAY_COLUMNS,
).run()

ts_runner = runner_results.df
print(f"\n-- AssayAnomalyRunner: downside vol (IG breakpoints) --")
print(f"Time series shape: {ts_runner.shape}")

runner_full, runner_recap = runner_results.summary_results(factor=mktb, nw_lag=3)
print(f"Results shape: {runner_full.shape}")

# =============================================================================
# 6. Smaller spec grid for quick comparison
# =============================================================================
SMALL_SPECS = {
    'weighting': ['VW'],
    'portfolio_structures': [(5, 'Q', None), (10, 'D', None)],
    'rating_filters': {'all': None, 'ig': 'IG'},
    'bp_universes': {'all': None},
    'maturity_filters': {'all': None},
}

result_small = assay_anomaly_fast(
    data=data, signal='ami', specs=SMALL_SPECS,
    holding_period=1, dynamic_weights=True,
    **BATCH_ASSAY_COLUMNS,
)
print(f"\n-- Small spec grid (Amihud) --")
print(result_small.summary().to_string(index=False))

print("\n[Done] Script 06 complete.")
