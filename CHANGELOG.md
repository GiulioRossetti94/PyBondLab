# Changelog

## 0.2.0

### New Features
- **Batch Processing**: `BatchStrategyFormation` for multi-signal portfolio formation in a single call
- **Within-Firm Sorting**: `WithinFirmSort` strategy + `BatchWithinFirmSortFormation` for within-issuer sorts
- **Rolling Beta Estimation**: `RollingBeta` with numba (~30x speedup) and numpy engines
- **Pre-Analysis Statistics**: `PreAnalysisStats` for cross-sectional summary statistics before sorting
- **Data Uncertainty Analysis**: `DataUncertaintyAnalysis` for robustness across filters/ratings/holding periods
- **Anomaly Assaying**: `AssayAnomaly` (standard) + `assay_anomaly_fast` (numba-optimized) + `BatchAssayAnomaly`
- **Specification Validation**: `validate_specs`, `SpecificationValidator` for validating strategy configurations
- **Factor Naming**: `NamingConfig` for consistent factor/portfolio naming conventions
- **Panel Extraction**: `extract_panel()` for unified panel construction from batch results
- **Non-Staggered Rebalancing**: Quarterly, semi-annual, annual rebalancing with monthly returns

### New Dependencies
- `scipy>=1.10` (required)
- `pyarrow>=10.0` (required, for parquet support)
- `numba>=0.57` (optional, `pip install PyBondLab[performance]`)
- `wrds` (optional, `pip install PyBondLab[wrds]`)

### API Changes
- `__all__` expanded from 6 to 30+ exports
- New strategy class: `WithinFirmSort`
- Panel validation utilities: `validate_panel`, `check_duplicates`

### Bug Fixes
- Fixed numpy engine crash with non-contiguous index in `RollingBeta`

### Documentation
- 12 new documentation files in `docs/` covering all major classes
- Comprehensive README with feature examples and API reference

## 0.1.5

- Initial public release
- Core portfolio sorting: `StrategyFormation`, `SingleSort`, `DoubleSort`, `Momentum`, `LTreversal`
- WRDS breakpoint loading: `load_breakpoints_WRDS`
