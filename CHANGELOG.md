# Changelog

## 0.2.0

Major release: numba-optimized batch processing, within-firm sorts, non-staggered rebalancing,
data uncertainty analysis, anomaly assaying, and 11 new example scripts.

### New Classes and Functions

- **`BatchStrategyFormation`** — Multi-signal portfolio formation in a single call.
  Auto-selects a numba fast path (returns-only) or slow path (turnover/chars/banding).
  Supports `rating`, `subset_filter`, and parallel processing via `n_jobs`.
- **`WithinFirmSort`** — Within-issuer sorting strategy (2 portfolios: High/Low).
  Hierarchical aggregation: within-firm VW, cap-weighted across firms, averaged across rating terciles.
- **`BatchWithinFirmSortFormation`** — Batch within-firm sorts across multiple signals.
- **`DataUncertaintyAnalysis`** — Robustness analysis across filters, ratings, and holding periods.
  Produces ex-ante and ex-post factor returns under multiple data-cleaning configurations.
- **`AssayAnomaly`** — Factor significance testing across specification choices
  (weighting, portfolios, rating subsets, breakpoint universes).
- **`assay_anomaly_fast`** — Numba-optimized anomaly assaying with spec grids.
- **`BatchAssayAnomaly`** — Parallel anomaly assaying across multiple signals.
- **`RollingBeta`** — Rolling window beta estimation with numba (~30x) and numpy engines.
- **`PreAnalysisStats`** — Cross-sectional summary statistics with rating/issuer/filter support.
- **`NamingConfig`** — Consistent factor naming with sign correction, rating suffixes, and leg labels.
- **`extract_panel()`** — Unified long-format panel extraction from batch results.
- **`validate_panel()`**, **`check_duplicates()`** — Panel validation utilities.
- **`validate_specs()`**, **`SpecificationValidator`** — Strategy specification validation.

### Non-Staggered Rebalancing

- Quarterly, semi-annual, annual, and custom-frequency rebalancing via `rebalance_frequency`
  on `SingleSort` / `DoubleSort`.
- Returns computed every month (not just at rebalancing dates).
- Weights renormalized on bond dropout between rebalancing dates.
- For non-staggered, omit `holding_period` (defaults to 1); the actual holding period
  is determined by `rebalance_frequency`.

### Performance

- Numba-optimized fast paths for `StrategyFormation` (SingleSort HP=1, returns-only),
  `BatchStrategyFormation` (multi-signal), and non-staggered rebalancing (~21-103x speedup).
- Fast path auto-detection: silently falls back to slow (pandas) path when conditions
  aren't met (turnover, chars, banding, filters, DoubleSort, WithinFirmSort).
  Both paths produce identical results.
- `dynamic_weights` support for HP>1 staggered portfolios (VW from d-1 vs formation date).
  No effect at HP=1.

### Configuration

- **`StrategyFormationConfig`** — Config-object API path for `StrategyFormation`
  (alternative to kwargs). Contains `DataConfig`, `FormationConfig`, `FilterConfig`.
- **Default differences** between classes:
  - `StrategyFormation`: `turnover=False`, `dynamic_weights=True`
  - `BatchStrategyFormation`: `turnover=True`, `dynamic_weights=True`
- **Banding**: `StrategyFormation` uses `banding_threshold` (float, e.g., `1/5`);
  `BatchStrategyFormation` uses `banding` (integer, e.g., `1`).

### New Dependencies

- `scipy>=1.10` (required)
- `pyarrow>=10.0` (required, for parquet support)
- `numba>=0.57` (required)
- `wrds` (optional, `pip install PyBondLab[wrds]`)

### Recent RELEASE Fixes

- Added a package-level workflow guide and linked the main docs back to a single canonical workflow.
- Fixed `DataUncertaintyAnalysis` so `ratings=` is a true analysis dimension on both slow and fast paths.
- Unified anomaly/spec-validator terminology and bounds around `NIG` (11-22).
- Clarified fast-vs-full result tiers, including `save_idx`, `extract_panel()`, and bond-count availability.
- Reorganized anomaly documentation around the public APIs instead of the internal runner.
- Added regression tests for rating bounds, non-monthly semantics, results contracts, anomaly equivalence, and specification validation.

### Bug Fixes

- Fixed numpy engine crash with non-contiguous index in `RollingBeta`
- Fixed categorical data handling in panel preparation
- Fixed `fastmath` bug in multi-signal rank computation

### Documentation

- 12 documentation files in `docs/` covering all major classes
- 11 example scripts in `examples/` with progressive tutorial structure
- API semantic notes in `docs/0_API_semantic_README.md` documenting conditional parameters
- Comprehensive README with Quick Start, API reference, and filter examples

---

## 0.1.5

Initial public release.

- Core portfolio sorting: `StrategyFormation`, `SingleSort`, `DoubleSort`, `Momentum`, `LTreversal`
- Look-ahead bias free data filtering procedures (trim, price, bounce, winsorize)
- Staggered holding period portfolios (overlapping cohorts)
- EW and VW portfolio returns
- WRDS breakpoint loading: `load_breakpoints_WRDS`
