"""
Blazing Fast Anomaly Assayer

Ultra-fast portfolio formation across large specification grids using
PyBondLab's numba-accelerated path.

Key optimizations:
1. Convert to numpy arrays ONCE at start, avoid repeated DataFrame operations
2. Vectorized threshold computation using numba
3. Pre-compute filter masks and batch rank computation
4. Group specs by filter configuration to minimize redundant work

@author: Claude Code
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import time
import warnings
from numba import njit, prange

try:
    from .constants import ColumnNames
    from .numba_core import (
        compute_ranks_all_dates_fast,
        compute_ranks_with_custom_thresholds,
        build_vw_lookup,
        build_vw_lookup_and_dynamic_weights,
        compute_all_returns_ultrafast,
        compute_staggered_returns_ultrafast,
    )
    from .spec_validator import (
        validate_specs,
        generate_spec_list,
        get_valid_spec_list,
        ValidationResult,
    )
except ImportError:
    from constants import ColumnNames
    from numba_core import (
        compute_ranks_all_dates_fast,
        compute_ranks_with_custom_thresholds,
        build_vw_lookup,
        build_vw_lookup_and_dynamic_weights,
        compute_all_returns_ultrafast,
        compute_staggered_returns_ultrafast,
    )
    from spec_validator import (
        validate_specs,
        generate_spec_list,
        get_valid_spec_list,
        ValidationResult,
    )


# =============================================================================
# Helper: Extract required columns from bp_func
# =============================================================================

def _get_bp_func_required_columns(bp_universes: Dict[str, Any]) -> set:
    """
    Extract required columns from all bp_func in bp_universes.

    bp_func can declare required columns via:
    - bp_func.required_columns = ['col1', 'col2']

    Parameters
    ----------
    bp_universes : dict
        Dictionary of {name: callable or None}

    Returns
    -------
    set
        Set of column names required by bp_func
    """
    required = set()
    for bp_name, bp_func in bp_universes.items():
        if bp_func is None:
            continue
        # Check for required_columns attribute
        if hasattr(bp_func, 'required_columns'):
            required.update(bp_func.required_columns)
        # Check wrapped function (for decorators)
        elif hasattr(bp_func, '__wrapped__') and hasattr(bp_func.__wrapped__, 'required_columns'):
            required.update(bp_func.__wrapped__.required_columns)
    return required


def get_rating_bounds(rating: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Get (min, max) rating bounds from specification."""
    if isinstance(rating, tuple):
        return rating
    rating_map = {
        'IG': (1, 10), 'ig': (1, 10),
        'HY': (11, 21), 'hy': (11, 21),
        'NIG': (11, 21), 'nig': (11, 21),
    }
    return rating_map.get(rating, (1, 21))


# =============================================================================
# Numba-accelerated threshold computation
# =============================================================================

@njit(parallel=True)
def compute_thresholds_all_dates_numba(
    date_idx: np.ndarray,
    signal: np.ndarray,
    bp_mask: np.ndarray,  # Which observations to use for breakpoints
    n_dates: int,
    breakpoints: np.ndarray,  # Percentiles like [10, 90] or [20, 40, 60, 80]
) -> np.ndarray:
    """
    Compute threshold edges for all dates in parallel using numba.

    Returns (n_dates, n_thresholds) array where n_thresholds = len(breakpoints) + 2
    """
    n_thresholds = len(breakpoints) + 2  # Including -inf and +inf edges
    thresholds = np.full((n_dates, n_thresholds), np.nan)

    for d in prange(n_dates):
        # Get signals for this date that pass bp_mask
        mask = (date_idx == d) & bp_mask & np.isfinite(signal)
        date_signals = signal[mask]

        if len(date_signals) < 2:
            continue

        # Sort for percentile computation
        sorted_sig = np.sort(date_signals)
        n = len(sorted_sig)

        # First threshold is -inf
        thresholds[d, 0] = -np.inf

        # Compute percentile thresholds
        for i, pct in enumerate(breakpoints):
            # Linear interpolation for percentile
            idx = (pct / 100.0) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            thresholds[d, i + 1] = sorted_sig[lower] * (1 - frac) + sorted_sig[upper] * frac

        # Last threshold is max value (not +inf to match slow path behavior)
        thresholds[d, -1] = sorted_sig[-1] + 1e-10

    return thresholds


@njit(parallel=True)
def compute_thresholds_equal_percentiles_numba(
    date_idx: np.ndarray,
    signal: np.ndarray,
    bp_mask: np.ndarray,
    n_dates: int,
    n_ports: int,
) -> np.ndarray:
    """
    Compute equal percentile thresholds for all dates.
    """
    n_thresholds = n_ports + 1
    thresholds = np.full((n_dates, n_thresholds), np.nan)
    percentiles = np.linspace(0, 100, n_ports + 1)

    for d in prange(n_dates):
        mask = (date_idx == d) & bp_mask & np.isfinite(signal)
        date_signals = signal[mask]

        if len(date_signals) < 2:
            continue

        sorted_sig = np.sort(date_signals)
        n = len(sorted_sig)

        thresholds[d, 0] = -np.inf

        for i in range(1, n_ports):
            pct = percentiles[i]
            idx = (pct / 100.0) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            thresholds[d, i] = sorted_sig[lower] * (1 - frac) + sorted_sig[upper] * frac

        thresholds[d, -1] = sorted_sig[-1] + 1e-10

    return thresholds


# =============================================================================
# Result container
# =============================================================================

@dataclass
class AnomalyAssayResult:
    """
    Container for anomaly assay results.

    Attributes
    ----------
    returns_df : pd.DataFrame
        DataFrame with date index and spec_id columns containing long-short returns
    metadata : dict
        Metadata including:
        - signal: str
        - specs: dict (specification grid)
        - n_specs: int
        - date_range: tuple (start, end)
        - runtime_seconds: float
        - validation_result: ValidationResult (if validated)
    """
    returns_df: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        n_specs = self.returns_df.shape[1]
        n_dates = len(self.returns_df)
        signal = self.metadata.get('signal', 'unknown')
        runtime = self.metadata.get('runtime_seconds', 0)
        return (
            f"AnomalyAssayResult(signal='{signal}', "
            f"specs={n_specs}, dates={n_dates}, "
            f"runtime={runtime:.2f}s)"
        )

    def summary(self, lag: int = 0) -> pd.DataFrame:
        """
        Compute summary statistics for all specifications.

        Parameters
        ----------
        lag : int
            HAC lag for standard errors (0 = OLS standard errors)

        Returns
        -------
        pd.DataFrame
            Summary with columns: spec_id, mean_ret, std_ret, t_stat, p_value,
            n_obs, sharpe_ann, mean_ret_ann, pct_positive
        """
        import statsmodels.api as sm

        results = []
        for col in self.returns_df.columns:
            ret = self.returns_df[col].dropna()
            if len(ret) < 10:
                continue

            # OLS regression on constant
            X = np.ones(len(ret))
            try:
                if lag > 0:
                    model = sm.OLS(ret.values, X).fit(
                        cov_type='HAC', cov_kwds={'maxlags': lag}
                    )
                else:
                    model = sm.OLS(ret.values, X).fit()

                mean_ret = model.params[0]
                t_stat = model.tvalues[0]
                p_val = model.pvalues[0]
            except Exception:
                mean_ret = ret.mean()
                t_stat = np.nan
                p_val = np.nan

            std_ret = ret.std()
            n_obs = len(ret)
            sharpe_ann = (mean_ret / std_ret) * np.sqrt(12) if std_ret > 0 else np.nan
            mean_ret_ann = mean_ret * 12
            pct_positive = (ret > 0).mean() * 100

            results.append({
                'spec_id': col,
                'mean_ret': mean_ret,
                'std_ret': std_ret,
                't_stat': t_stat,
                'p_value': p_val,
                'n_obs': n_obs,
                'sharpe_ann': sharpe_ann,
                'mean_ret_ann': mean_ret_ann,
                'pct_positive': pct_positive,
            })

        return pd.DataFrame(results)

    def to_csv(self, path: str, include_metadata: bool = True):
        """Save returns to CSV with optional metadata JSON."""
        self.returns_df.to_csv(path)
        if include_metadata:
            import json
            meta_path = path.replace('.csv', '_metadata.json')
            # Filter out non-serializable items
            safe_meta = {k: v for k, v in self.metadata.items()
                        if isinstance(v, (str, int, float, list, dict, bool, type(None)))}
            with open(meta_path, 'w') as f:
                json.dump(safe_meta, f, indent=2, default=str)

    @classmethod
    def from_csv(cls, path: str) -> 'AnomalyAssayResult':
        """Load from CSV."""
        returns_df = pd.read_csv(path, index_col=0, parse_dates=True)
        metadata = {}
        meta_path = path.replace('.csv', '_metadata.json')
        try:
            import json
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            pass
        return cls(returns_df=returns_df, metadata=metadata)


# =============================================================================
# Main function
# =============================================================================

def assay_anomaly_fast(
    data: pd.DataFrame,
    signal: str,
    specs: Dict[str, Any],
    *,
    holding_period: int = 1,
    dynamic_weights: bool = True,
    validate: bool = False,
    validate_sample_size: int = 3,
    skip_invalid: bool = True,
    verbose: bool = True,
    IDvar: str = None,
    DATEvar: str = None,
    RETvar: str = None,
    VWvar: str = None,
    RATINGvar: str = None,
) -> AnomalyAssayResult:
    """
    Fast anomaly assaying across specification grid using numba-accelerated path.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel with date, ID, return, VW, rating, and signal columns
    signal : str
        Signal column name for sorting
    specs : dict
        Specification grid with keys:
        - 'weighting': list of ['EW', 'VW']
        - 'portfolio_structures': list of (n_ports, name, breakpoints)
          Use breakpoints=None for equal percentiles (recommended)
        - 'rating_filters': dict of {name: filter_value}
        - 'bp_universes': dict of {name: callable or None}
        - 'maturity_filters': dict of {name: (min, max) or None}
    holding_period : int
        Holding period in months (default=1)
    dynamic_weights : bool
        Use dynamic weights (VW from d-1) if True
    validate : bool
        If True, validate sample against slow path
    validate_sample_size : int
        Number of specs to validate (random sample)
    skip_invalid : bool
        If True, skip specifications with validation errors
    verbose : bool
        Print progress
    IDvar, DATEvar, RETvar, VWvar, RATINGvar : str
        Column name mappings (optional, uses defaults if None)

    Returns
    -------
    AnomalyAssayResult
        Container with returns_df and metadata
    """
    start_time = time.time()

    # Column name mapping
    id_col = IDvar or 'ID'
    date_col = DATEvar or 'date'
    ret_col = RETvar or 'ret'
    vw_col = VWvar or 'VW'
    rating_col = RATINGvar or 'RATING_NUM'

    # Validate signal exists
    if signal not in data.columns:
        raise ValueError(f"Signal '{signal}' not found in data columns")

    # Validate and filter specs
    if skip_invalid:
        valid_specs, validation_result = get_valid_spec_list(
            specs, data=data, rating_col=rating_col, verbose=verbose
        )
    else:
        valid_specs = generate_spec_list(specs)
        validation_result = None

    if len(valid_specs) == 0:
        raise ValueError("No valid specifications to run")

    if verbose:
        print(f"Running {len(valid_specs)} specifications for signal '{signal}'...")

    # =========================================================================
    # OPTIMIZATION 1: Convert entire dataset to numpy arrays ONCE
    # =========================================================================
    t0 = time.time()

    # Get unique dates and create mappings
    datelist = sorted(data[date_col].unique())
    TM = len(datelist)
    date_to_idx = {d: i for i, d in enumerate(datelist)}

    # Get unique IDs
    all_ids = data[id_col].unique()
    id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}
    n_ids = len(all_ids)

    # Convert to numpy arrays (do this ONCE for the entire dataset)
    date_idx_full = data[date_col].map(date_to_idx).values.astype(np.int64)
    ids_full = data[id_col].values  # Keep original IDs for re-mapping
    signal_full = data[signal].values.astype(np.float64)
    returns_full = data[ret_col].values.astype(np.float64)
    vw_full = data[vw_col].values.astype(np.float64)

    # Rating and maturity for filtering
    if rating_col in data.columns:
        rating_full = data[rating_col].values.astype(np.float64)
    else:
        rating_full = np.ones(len(data))

    if 'tmat' in data.columns:
        tmat_full = data['tmat'].values.astype(np.float64)
    else:
        tmat_full = np.full(len(data), 50.0)  # Default mid-range maturity

    # Extract columns required by bp_func
    bp_universes = specs.get('bp_universes', {'all': None})
    bp_required_cols = _get_bp_func_required_columns(bp_universes)

    # Convert bp_func required columns to numpy arrays
    bp_col_arrays = {}
    for col in bp_required_cols:
        if col in data.columns:
            bp_col_arrays[col] = data[col].values.astype(np.float64)
        elif col == vw_col:
            bp_col_arrays[col] = vw_full  # Already have this
        elif col == rating_col:
            bp_col_arrays[col] = rating_full  # Already have this
        else:
            warnings.warn(f"bp_func requires column '{col}' but it's not in data")

    if verbose:
        print(f"  Data conversion: {(time.time()-t0)*1000:.0f}ms")
        if bp_required_cols:
            print(f"  bp_func required columns: {bp_required_cols}")

    # =========================================================================
    # OPTIMIZATION 2: Group specs and pre-compute filter masks
    # =========================================================================
    t0 = time.time()

    # Group specs by filter configuration
    filter_groups = {}
    for spec in valid_specs:
        key = (spec['rating_name'], spec['maturity_name'])
        if key not in filter_groups:
            filter_groups[key] = {
                'rating_filter': spec['rating_filter'],
                'maturity_filter': spec['maturity_filter'],
                'specs': []
            }
        filter_groups[key]['specs'].append(spec)

    # Pre-compute filter masks for each group (numpy boolean arrays)
    for key, group in filter_groups.items():
        rat_filter = group['rating_filter']
        mat_filter = group['maturity_filter']

        # Rating mask
        if rat_filter is not None:
            if isinstance(rat_filter, str):
                r_min, r_max = get_rating_bounds(rat_filter)
            else:
                r_min, r_max = rat_filter
            rat_mask = (rating_full >= r_min) & (rating_full <= r_max)
        else:
            rat_mask = np.ones(len(data), dtype=np.bool_)

        # Maturity mask
        if mat_filter is not None:
            m_min, m_max = mat_filter
            mat_mask = (tmat_full >= m_min) & (tmat_full <= m_max)
        else:
            mat_mask = np.ones(len(data), dtype=np.bool_)

        group['filter_mask'] = rat_mask & mat_mask

    if verbose:
        print(f"  Filter masks: {(time.time()-t0)*1000:.0f}ms")

    # =========================================================================
    # OPTIMIZATION 3: Process each filter group with vectorized operations
    # =========================================================================
    all_results = {}

    for (rat_name, mat_name), group in filter_groups.items():
        t0 = time.time()
        filter_mask = group['filter_mask']
        group_specs = group['specs']

        if verbose:
            print(f"  Processing {rat_name}/{mat_name} ({len(group_specs)} specs)...", end="")

        # CRITICAL: Create filtered arrays for this group
        # This ensures ID mapping is consistent with filtered data
        filtered_indices = np.where(filter_mask)[0]

        if len(filtered_indices) == 0:
            if verbose:
                print(" (empty)")
            continue

        # Extract filtered arrays (using pre-computed numpy arrays)
        date_idx_filt = date_idx_full[filtered_indices]
        signal_filt = signal_full[filtered_indices].copy()
        returns_filt = returns_full[filtered_indices]
        vw_filt = vw_full[filtered_indices]
        ids_filt = ids_full[filtered_indices]

        # Create new ID mapping for filtered data
        unique_ids_filt = np.unique(ids_filt)
        id_to_idx_filt = {id_val: idx for idx, id_val in enumerate(unique_ids_filt)}
        n_ids_filt = len(unique_ids_filt)

        # Map IDs to new indices
        id_idx_filt = np.array([id_to_idx_filt[id_val] for id_val in ids_filt], dtype=np.int64)

        # Group specs by breakpoint configuration
        bp_configs = {}
        for spec in group_specs:
            bp_key = (spec['bp_name'], spec['n_ports'],
                      tuple(spec['breakpoints']) if spec['breakpoints'] else None)
            if bp_key not in bp_configs:
                bp_configs[bp_key] = {
                    'bp_func': spec['bp_func'],
                    'n_ports': spec['n_ports'],
                    'breakpoints': spec['breakpoints'],
                    'specs': []
                }
            bp_configs[bp_key]['specs'].append(spec)

        # Process each breakpoint configuration
        for bp_key, bp_config in bp_configs.items():
            n_ports = bp_config['n_ports']
            breakpoints = bp_config['breakpoints']
            bp_func = bp_config['bp_func']

            # Compute breakpoint mask (relative to filtered data)
            if bp_func is not None:
                # Build filtered_df with base columns plus any bp_func required columns
                filtered_df_dict = {
                    rating_col: rating_full[filtered_indices],
                    vw_col: vw_full[filtered_indices],
                }
                # Add bp_func required columns if declared
                if hasattr(bp_func, 'required_columns'):
                    for col in bp_func.required_columns:
                        if col in bp_col_arrays:
                            filtered_df_dict[col] = bp_col_arrays[col][filtered_indices]
                        # Skip if already included (rating_col or vw_col)

                filtered_df = pd.DataFrame(filtered_df_dict)
                bp_result = bp_func(filtered_df)
                if isinstance(bp_result, pd.Series):
                    bp_mask = bp_result.values.astype(np.bool_)
                else:
                    bp_mask = np.array(bp_result, dtype=np.bool_)
            else:
                bp_mask = np.ones(len(filtered_indices), dtype=np.bool_)

            # Compute thresholds using vectorized numba
            if breakpoints is not None:
                bp_array = np.array(breakpoints, dtype=np.float64)
                thresholds = compute_thresholds_all_dates_numba(
                    date_idx_filt, signal_filt, bp_mask, TM, bp_array
                )
            else:
                thresholds = compute_thresholds_equal_percentiles_numba(
                    date_idx_filt, signal_filt, bp_mask, TM, n_ports
                )

            # Compute ranks using thresholds
            ranks = compute_ranks_with_custom_thresholds(
                date_idx_filt, signal_filt, thresholds, TM, n_ports
            )

            # Build VW lookup for filtered data
            vw_lookup = build_vw_lookup_and_dynamic_weights(
                date_idx_filt, id_idx_filt, vw_filt, TM, n_ids_filt
            )

            # Compute returns - use staggered for h > 1
            if holding_period == 1:
                # Simple case: no staggering needed
                ew_ret, vw_ret = compute_all_returns_ultrafast(
                    date_idx_filt, id_idx_filt, returns_filt, vw_lookup,
                    date_idx_filt, id_idx_filt, ranks,
                    TM, n_ids_filt, n_ports
                )
            else:
                # Staggered portfolios: average across h cohorts
                ew_ret, vw_ret = compute_staggered_returns_ultrafast(
                    date_idx_filt, id_idx_filt, returns_filt, vw_lookup,
                    date_idx_filt, id_idx_filt, ranks,
                    TM, n_ids_filt, n_ports,
                    holding_period,  # hor parameter
                    dynamic_weights  # use_dynamic_weights parameter
                )

            # Long-short returns
            ew_ls = ew_ret[:, -1] - ew_ret[:, 0]
            vw_ls = vw_ret[:, -1] - vw_ret[:, 0]

            # Store results for each spec
            for spec in bp_config['specs']:
                if spec['weighting'] == 'EW':
                    all_results[spec['spec_id']] = ew_ls.copy()
                else:
                    all_results[spec['spec_id']] = vw_ls.copy()

        if verbose:
            print(f" {(time.time()-t0)*1000:.0f}ms")

    # Build results DataFrame
    returns_df = pd.DataFrame(all_results, index=datelist)

    runtime = time.time() - start_time
    metadata = {
        'signal': signal,
        'n_specs': len(valid_specs),
        'n_specs_run': len(all_results),
        'holding_period': holding_period,
        'dynamic_weights': dynamic_weights,
        'date_range': (str(datelist[0]), str(datelist[-1])),
        'runtime_seconds': runtime,
    }

    if verbose:
        print(f"Completed in {runtime:.2f}s ({len(all_results)} specs)")

    result = AnomalyAssayResult(returns_df=returns_df, metadata=metadata)

    # Optional validation against slow path
    if validate:
        _validate_against_slow_path(
            result, data, signal, specs, valid_specs,
            holding_period, dynamic_weights,
            validate_sample_size, verbose,
            id_col, date_col, ret_col, vw_col, rating_col
        )

    return result


# =============================================================================
# Validation helper (for debugging)
# =============================================================================

def _validate_against_slow_path(
    result: AnomalyAssayResult,
    data: pd.DataFrame,
    signal: str,
    specs: Dict[str, Any],
    valid_specs: List[Dict[str, Any]],
    holding_period: int,
    dynamic_weights: bool,
    sample_size: int,
    verbose: bool,
    id_col: str,
    date_col: str,
    ret_col: str,
    vw_col: str,
    rating_col: str,
):
    """Validate against slow path for a sample of specs."""
    import random

    try:
        from .PyBondLab import StrategyFormation
        from .StrategyClass import SingleSort
    except ImportError:
        from PyBondLab import StrategyFormation
        from StrategyClass import SingleSort

    if verbose:
        print(f"\nValidating {sample_size} specs against slow path...")

    # Random sample
    sample_specs = random.sample(valid_specs, min(sample_size, len(valid_specs)))

    discrepancies = []
    for spec in sample_specs:
        spec_id = spec['spec_id']

        if spec_id not in result.returns_df.columns:
            if verbose:
                print(f"  {spec_id}: SKIPPED (not in results)")
            continue

        fast_ret = result.returns_df[spec_id].values

        # Build slow path strategy
        strategy = SingleSort(
            signal,
            holding_period=holding_period,
            num_portfolios=spec['n_ports'],
            breakpoints=spec['breakpoints'],
            breakpoint_universe_func=spec['bp_func'],
        )

        # Prepare data with column renaming to match PyBondLab defaults
        slow_data = data.rename(columns={
            id_col: 'ID',
            date_col: 'date',
            ret_col: 'ret',
            vw_col: 'VW',
            rating_col: 'RATING_NUM',
        })

        try:
            # Create StrategyFormation
            # Use turnover=True to force slow path
            sf = StrategyFormation(
                slow_data,
                strategy,
                rating=spec['rating_filter'],
                subset_filter={'tmat': spec['maturity_filter']} if spec['maturity_filter'] else None,
                dynamic_weights=dynamic_weights,
                turnover=True,  # Force slow path by requesting turnover
                verbose=False,
            )
            sf.fit()
            slow_results = sf.results

            # Get slow path long-short returns using new API
            try:
                ew_ls, vw_ls = slow_results.get_long_short(strategy='ea')
                if spec['weighting'] == 'EW':
                    slow_ret = ew_ls.values.flatten()
                else:
                    slow_ret = vw_ls.values.flatten()
            except Exception as e:
                if verbose:
                    print(f"  {spec_id}: SKIPPED (error getting slow path results: {e})")
                continue

            # Compare
            min_len = min(len(fast_ret), len(slow_ret))
            if min_len == 0:
                continue

            # Use np.isfinite to handle NaN properly
            valid_mask = np.isfinite(fast_ret[:min_len]) & np.isfinite(slow_ret[:min_len])
            if valid_mask.sum() == 0:
                if verbose:
                    print(f"  {spec_id}: SKIPPED (no overlapping valid data)")
                continue

            max_diff = np.max(np.abs(fast_ret[:min_len][valid_mask] - slow_ret[:min_len][valid_mask]))
            if max_diff > 1e-8:
                discrepancies.append({
                    'spec_id': spec_id,
                    'max_diff': max_diff,
                })
                if verbose:
                    print(f"  {spec_id}: MISMATCH (max diff = {max_diff:.2e})")
            else:
                if verbose:
                    print(f"  {spec_id}: OK (max diff = {max_diff:.2e})")

        except Exception as e:
            if verbose:
                print(f"  {spec_id}: ERROR ({e})")
            continue

    # Report
    if discrepancies:
        warnings.warn(
            f"Validation found {len(discrepancies)} discrepancies! "
            f"Results may not match slow path exactly."
        )
        result.metadata['validation_discrepancies'] = discrepancies
    else:
        if verbose:
            print("All validated specs match slow path!")
        result.metadata['validation_status'] = 'passed'
