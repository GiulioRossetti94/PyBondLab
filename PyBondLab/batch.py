# -*- coding: utf-8 -*-
"""
Batch Strategy Formation for PyBondLab.

This module provides efficient batch processing for multiple signals,
using parallel processing to achieve significant speedups.

Memory Optimizations
--------------------
- Only required columns are sent to workers (reduces pickle size by 50-80%)
- shared_precomp is NOT sent to workers (each worker computes its own)
- On Linux/macOS, uses 'fork' for copy-on-write memory sharing

Example Usage
-------------
>>> from PyBondLab import BatchStrategyFormation
>>>
>>> batch_sf = BatchStrategyFormation(
...     data=data,
...     signals=['momentum', 'value', 'size', 'reversal'],
...     holding_period=3,
...     num_portfolios=5,
...     turnover=True,
...     n_jobs=-1,  # Use all CPU cores
... )
>>> results = batch_sf.fit()
>>>
>>> # Access individual signal results
>>> results['momentum'].get_long_short()
>>> results['momentum'].get_turnover()
>>>
>>> # Summary across all signals
>>> results.summary_df

Authors: PyBondLab Team
Created: 2024
"""

import os
import sys
import gc
import platform
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import OrderedDict

# Type alias for subset_filter
SubsetFilter = Dict[str, Tuple[float, float]]
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# =============================================================================
# Shared utilities from batch_base
# =============================================================================

from .batch_base import (
    _get_start_method,
    TQDM_AVAILABLE,
    tqdm,
    REQUIRED_COLUMNS,
    DEFAULT_COLUMNS,
)

# Import PyBondLab components
from .PyBondLab import StrategyFormation
from .StrategyClass import SingleSort
from .config import StrategyFormationConfig, DataConfig, FormationConfig
from .results import FormationResults


# =============================================================================
# Worker function for parallel processing (must be at module level for pickle)
# =============================================================================

def _process_single_signal(args: Tuple) -> Tuple[str, Any, float, Optional[str]]:
    """
    Process a single signal - worker function for parallel execution.

    Parameters
    ----------
    args : tuple
        (signal, data, holding_period, num_portfolios, turnover,
         chars, rating, subset_filter, banding_threshold,
         rebalance_frequency, rebalance_month)

        NOTE: shared_precomp is NOT passed to avoid pickle overhead.
        Each worker computes its own precompute data.

    Returns
    -------
    tuple
        (signal_name, result_or_none, elapsed_time, error_or_none)
    """
    (signal, data, holding_period, num_portfolios, turnover,
     chars, rating, subset_filter, banding_threshold,
     rebalance_frequency, rebalance_month) = args

    t_start = time.time()

    try:
        # Suppress warnings in worker
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create strategy
            strategy = SingleSort(
                holding_period=holding_period,
                sort_var=signal,
                num_portfolios=num_portfolios,
                rebalance_frequency=rebalance_frequency,
                rebalance_month=rebalance_month,
                verbose=False
            )

            # Create config
            sf_config = StrategyFormationConfig(
                data=DataConfig(
                    rating=rating,
                    subset_filter=subset_filter,
                    chars=chars,
                ),
                formation=FormationConfig(
                    dynamic_weights=True,
                    compute_turnover=turnover,
                    banding_threshold=banding_threshold,
                    verbose=False,
                )
            )

            # Create and run StrategyFormation
            # Each worker computes its own precompute (no shared_precomp passed)
            sf = StrategyFormation(
                data=data,
                strategy=strategy,
                config=sf_config
            )

            result = sf.fit()

            elapsed = time.time() - t_start
            return (signal, result, elapsed, None)

    except Exception as e:
        elapsed = time.time() - t_start
        return (signal, None, elapsed, str(e))


def _process_signal_batch(args: Tuple) -> List[Tuple[str, Any, float, Optional[str]]]:
    """
    Process multiple signals in a single worker - reduces overhead.

    Parameters
    ----------
    args : tuple
        (signals_list, data, holding_period, num_portfolios, turnover,
         chars, rating, subset_filter, banding_threshold,
         rebalance_frequency, rebalance_month)

    Returns
    -------
    list of tuple
        [(signal_name, result_or_none, elapsed_time, error_or_none), ...]
    """
    (signals_list, data, holding_period, num_portfolios, turnover,
     chars, rating, subset_filter, banding_threshold,
     rebalance_frequency, rebalance_month) = args

    results = []
    for signal in signals_list:
        t_start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                strategy = SingleSort(
                    holding_period=holding_period,
                    sort_var=signal,
                    num_portfolios=num_portfolios,
                    rebalance_frequency=rebalance_frequency,
                    rebalance_month=rebalance_month,
                    verbose=False
                )

                sf_config = StrategyFormationConfig(
                    data=DataConfig(rating=rating, subset_filter=subset_filter, chars=chars),
                    formation=FormationConfig(
                        dynamic_weights=True,
                        compute_turnover=turnover,
                        banding_threshold=banding_threshold,
                        verbose=False,
                    )
                )

                sf = StrategyFormation(data=data, strategy=strategy, config=sf_config)
                result = sf.fit()

                elapsed = time.time() - t_start
                results.append((signal, result, elapsed, None))

        except Exception as e:
            elapsed = time.time() - t_start
            results.append((signal, None, elapsed, str(e)))

    return results


# =============================================================================
# Batch Results Container
# =============================================================================

@dataclass
class BatchResults:
    """
    Container for batch strategy formation results.

    Provides dictionary-like access to individual signal results,
    plus aggregate statistics across all signals.
    """

    results: OrderedDict = field(default_factory=OrderedDict)
    signals: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, signal: str) -> FormationResults:
        """Get results for a specific signal."""
        if signal not in self.results:
            raise KeyError(f"Signal '{signal}' not found. Available: {list(self.results.keys())}")
        return self.results[signal]

    def __contains__(self, signal: str) -> bool:
        return signal in self.results

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    @property
    def successful_signals(self) -> List[str]:
        return list(self.results.keys())

    @property
    def failed_signals(self) -> List[str]:
        return list(self.errors.keys())

    @property
    def summary_df(self) -> pd.DataFrame:
        """Summary DataFrame with key statistics for all signals."""
        rows = []
        for signal, result in self.results.items():
            try:
                ea = result.ea
                ew_ls, vw_ls = result.get_long_short(strategy='ea')
                row = {
                    'signal': signal,
                    'ew_mean': ew_ls.mean() * 12,
                    'vw_mean': vw_ls.mean() * 12,
                    'ew_std': ew_ls.std() * np.sqrt(12),
                    'vw_std': vw_ls.std() * np.sqrt(12),
                    'ew_sharpe': (ew_ls.mean() / ew_ls.std()) * np.sqrt(12) if ew_ls.std() > 0 else np.nan,
                    'vw_sharpe': (vw_ls.mean() / vw_ls.std()) * np.sqrt(12) if vw_ls.std() > 0 else np.nan,
                    'n_periods': len(ew_ls),
                }
                if ea.turnover is not None and ea.turnover.ew_turnover_df is not None:
                    row['ew_turnover'] = ea.turnover.ew_turnover_df.mean().mean()
                    row['vw_turnover'] = ea.turnover.vw_turnover_df.mean().mean()
                rows.append(row)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index('signal')
        return df

    def get_factor_returns(self, weight_type: str = 'ew') -> pd.DataFrame:
        """Get long-short factor returns for all signals as a DataFrame."""
        factor_dict = {}
        for signal, result in self.results.items():
            try:
                ew_ls, vw_ls = result.get_long_short(strategy='ea')
                factor_dict[signal] = ew_ls if weight_type == 'ew' else vw_ls
            except Exception:
                continue
        if not factor_dict:
            return pd.DataFrame()
        return pd.DataFrame(factor_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signals': self.signals,
            'config': self.config,
            'timings': self.timings,
            'errors': self.errors,
            'summary': self.summary_df.to_dict() if len(self.results) > 0 else {},
        }


# =============================================================================
# Fast Batch Result (lightweight result object for fast path)
# =============================================================================

class _FastBatchResult:
    """
    Lightweight result object for fast batch path.

    Provides the same interface as FormationResults for accessing long-short returns,
    but without the full portfolio breakdown (only returns long-short factor).
    """

    def __init__(self, ew_ls: pd.Series, vw_ls: pd.Series, signal: str):
        """
        Parameters
        ----------
        ew_ls : pd.Series
            Equal-weighted long-short returns (index=dates)
        vw_ls : pd.Series
            Value-weighted long-short returns (index=dates)
        signal : str
            Signal name
        """
        self.ew_ls = ew_ls
        self.vw_ls = vw_ls
        self.signal = signal

        # Create a mock 'ea' attribute for compatibility with summary_df
        self._ea = _FastBatchEA(ew_ls, vw_ls)

    @property
    def ea(self):
        """Ex-ante results (returns-only for fast path)."""
        return self._ea

    @property
    def ep(self):
        """Ex-post results (same as EA for baseline/no-filter case)."""
        return self._ea

    def get_long_short(self, strategy: str = 'ea'):
        """
        Get long-short portfolio returns.

        Parameters
        ----------
        strategy : str, default='ea'
            'ea' for ex-ante, 'ep' for ex-post (same for fast path)

        Returns
        -------
        tuple
            (ew_ls, vw_ls) - Equal-weighted and value-weighted long-short returns
        """
        return self.ew_ls, self.vw_ls

    def get_turnover(self):
        """Turnover not available in fast batch path."""
        return None, None

    def get_characteristics(self):
        """Characteristics not available in fast batch path."""
        return None, None


class _FastBatchEA:
    """Mock EA object for fast batch results compatibility."""

    def __init__(self, ew_ls: pd.Series, vw_ls: pd.Series):
        self._ew_ls = ew_ls
        self._vw_ls = vw_ls
        self.turnover = None

    @property
    def returns(self):
        """Mock returns object."""
        return _FastBatchReturns(self._ew_ls, self._vw_ls)


class _FastBatchReturns:
    """Mock returns object for fast batch results compatibility."""

    def __init__(self, ew_ls: pd.Series, vw_ls: pd.Series):
        self._ew_ls = ew_ls
        self._vw_ls = vw_ls

    @property
    def ewls_df(self):
        """Equal-weighted long-short returns as DataFrame."""
        return self._ew_ls.to_frame('ewls')

    @property
    def vwls_df(self):
        """Value-weighted long-short returns as DataFrame."""
        return self._vw_ls.to_frame('vwls')


# =============================================================================
# Batch Strategy Formation
# =============================================================================

class BatchStrategyFormation:
    """
    Batch processing for multiple signals with parallel execution.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signals : list of str
        Column names to use as sorting signals
    holding_period : int, default=1
        Holding period in months
    num_portfolios : int, default=5
        Number of portfolios to form
    turnover : bool, default=True
        Whether to compute portfolio turnover
    chars : list of str, optional
        Characteristic columns to aggregate
    rating : str or tuple, optional
        Rating filter:
        - 'IG': Investment grade (RATING_NUM 1-10)
        - 'NIG': Non-investment grade (RATING_NUM 11-22)
        - (min, max): Custom rating range, e.g., (7, 10) for BBB only
    subset_filter : dict, optional
        Characteristic-based filters: {column: (min, max)}
        Example: {'MATURITY': (1, 5), 'DURATION': (2, 8)}
        Filters are applied at formation date only (no look-ahead bias).
    banding : int, optional
        Banding parameter
    rebalance_frequency : str or int, default='monthly'
        Rebalancing frequency:
        - 'monthly': Rebalance every month (staggered portfolios)
        - 'quarterly' or 3: Rebalance every 3 months
        - 'semi-annual' or 6: Rebalance every 6 months
        - 'annual' or 12: Rebalance every 12 months
    rebalance_month : int or list of int, default=6
        Month(s) when rebalancing occurs (1=Jan, 6=Jun, 12=Dec)
    columns : dict, optional
        Column name mapping from PyBondLab names to your data's column names.
        Default mapping: {'date': 'date', 'ID': 'ID', 'ret': 'ret', 'VW': 'VW', 'RATING_NUM': 'RATING_NUM'}
        Example: {'ID': 'cusip', 'ret': 'ret_vw', 'VW': 'mcap_e', 'RATING_NUM': 'spc_rat'}
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all cores, 1 for sequential.
    signals_per_worker : int, default=1
        Number of signals to process per worker. Higher values reduce overhead
        but increase per-worker memory. Recommended: 2-4 for large datasets.
    chunk_size : int, optional
        Process signals in chunks of this size to limit peak memory usage.
        If None, processes all signals at once. Recommended for 50+ signals.
    verbose : bool, default=True
        Whether to show progress

    Examples
    --------
    >>> # With custom column names
    >>> batch = BatchStrategyFormation(
    ...     data=data,
    ...     signals=['cs', 'ytm'],
    ...     columns={'ID': 'cusip', 'ret': 'ret_vw', 'VW': 'mcap_e', 'RATING_NUM': 'spc_rat'},
    ...     n_jobs=4
    ... )
    >>> results = batch.fit()
    >>>
    >>> # With rating and subset filters
    >>> batch = BatchStrategyFormation(
    ...     data=data,
    ...     signals=['signal1', 'signal2'],
    ...     rating=(1, 10),  # IG only
    ...     subset_filter={'MATURITY': (1, 5)},  # Maturity 1-5 years
    ...     turnover=False,
    ... )
    >>> results = batch.fit()
    >>>
    >>> # With non-staggered rebalancing (quarterly, June start)
    >>> batch = BatchStrategyFormation(
    ...     data=data,
    ...     signals=['signal1', 'signal2'],
    ...     holding_period=3,
    ...     rebalance_frequency='quarterly',  # or 3
    ...     rebalance_month=6,  # June, September, December, March
    ...     turnover=False,
    ... )
    >>> results = batch.fit()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: List[str],
        holding_period: int = 1,
        num_portfolios: int = 5,
        turnover: bool = True,
        chars: Optional[List[str]] = None,
        rating: Optional[Union[str, tuple]] = None,
        subset_filter: Optional[SubsetFilter] = None,
        banding: Optional[int] = None,
        rebalance_frequency: Union[str, int] = 'monthly',
        rebalance_month: Union[int, List[int]] = 6,
        columns: Optional[Dict[str, str]] = None,
        n_jobs: int = 1,
        signals_per_worker: int = 1,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
    ):
        # Store verbose first (needed by _prepare_data)
        self.verbose = verbose

        # Build column mapping (merge defaults with user-provided)
        self.columns = DEFAULT_COLUMNS.copy()
        if columns is not None:
            self.columns.update(columns)

        # Prepare data (rename columns to PyBondLab standard names)
        self.data_raw = data
        self.data = self._prepare_data(data, signals)

        self._validate_inputs(self.data, signals)

        self.signals = list(signals)
        self.holding_period = holding_period
        self.num_portfolios = num_portfolios
        self.turnover = turnover
        self.chars = chars
        self.rating = rating
        self.subset_filter = subset_filter
        self.banding = banding
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_month = rebalance_month
        self.n_jobs = n_jobs
        self.signals_per_worker = max(1, signals_per_worker)
        self.chunk_size = chunk_size

        self.banding_threshold = None
        if banding is not None:
            self.banding_threshold = banding / num_portfolios

        # Determine if this is non-staggered (non-monthly) rebalancing
        self._is_nonstaggered = self._check_nonstaggered()

        self.config = {
            'holding_period': holding_period,
            'num_portfolios': num_portfolios,
            'turnover': turnover,
            'chars': chars,
            'rating': rating,
            'subset_filter': subset_filter,
            'banding': banding,
            'rebalance_frequency': rebalance_frequency,
            'rebalance_month': rebalance_month,
            'n_jobs': n_jobs,
            'signals_per_worker': signals_per_worker,
            'chunk_size': chunk_size,
        }

    def _prepare_data(self, data: pd.DataFrame, signals: List[str]) -> pd.DataFrame:
        """
        Prepare data by renaming columns to PyBondLab standard names.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data with user's column names
        signals : List[str]
            Signal column names (these are NOT renamed)

        Returns
        -------
        pd.DataFrame
            Data with standardized column names
        """
        # Build rename mapping: user_col_name -> pbl_name
        rename_map = {}
        for pbl_name, user_name in self.columns.items():
            if user_name != pbl_name and user_name in data.columns:
                rename_map[user_name] = pbl_name

        if not rename_map:
            # No renaming needed
            return data

        # Check for conflicts: user columns that would overwrite existing columns
        for user_name, pbl_name in rename_map.items():
            if pbl_name in data.columns and pbl_name != user_name:
                # The target column already exists and is different from source
                # This could cause issues, warn or handle appropriately
                pass

        # Rename columns
        data_prepared = data.rename(columns=rename_map)

        if self.verbose:
            renamed_str = ', '.join(f'{k}->{v}' for k, v in rename_map.items())
            print(f"Columns renamed: {renamed_str}")

        return data_prepared

    def _validate_inputs(self, data: pd.DataFrame, signals: List[str]):
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        if not signals:
            raise ValueError("Must provide at least one signal")
        required = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']
        missing = [col for col in required if col not in data.columns]
        if missing:
            # Provide helpful error message with column mapping info
            user_cols = [self.columns.get(c, c) for c in missing]
            raise ValueError(
                f"Data missing required columns: {missing}. "
                f"Expected columns (based on your 'columns' mapping): {user_cols}. "
                f"Use the 'columns' parameter to map your column names."
            )
        missing_signals = [s for s in signals if s not in data.columns]
        if missing_signals:
            raise ValueError(f"Signal columns not found in data: {missing_signals}")

    def _check_nonstaggered(self) -> bool:
        """Check if this is non-staggered (non-monthly) rebalancing."""
        freq = self.rebalance_frequency
        if isinstance(freq, str):
            return freq != 'monthly'
        elif isinstance(freq, int):
            return freq > 1
        return False

    def _get_n_workers(self) -> int:
        """Determine number of worker processes."""
        if self.n_jobs == 1:
            return 1
        elif self.n_jobs == -1:
            return mp.cpu_count()
        elif self.n_jobs < -1:
            return max(1, mp.cpu_count() + 1 + self.n_jobs)
        else:
            return min(self.n_jobs, mp.cpu_count())

    def _can_use_fast_batch_path(self) -> bool:
        """
        Check if fast batch path can be used.

        Fast path requires:
        - turnover=False
        - chars=None
        - banding=None (no banding threshold)
        - Monthly rebalancing (staggered) OR non-staggered rebalancing

        Fast path NOW SUPPORTS (Phase 14):
        - rating filter (applied at formation date only, no look-ahead bias)
        - subset_filter (applied at formation date only, no look-ahead bias)

        Fast path NOW SUPPORTS (Phase 15):
        - Non-staggered rebalancing (quarterly, semi-annual, annual)
        """
        if self.turnover:
            return False
        if self.chars is not None and len(self.chars) > 0:
            return False
        if self.banding_threshold is not None:
            return False
        # rating and subset_filter are now supported in fast path
        # Non-staggered rebalancing is now supported in fast path
        return True

    def _fit_fast_batch(self) -> BatchResults:
        """
        Ultra-fast batch processing using numba kernels.

        Processes ALL signals in parallel using vectorized operations.
        Available when turnover=False, chars=None, banding=None.

        Supports rating and subset_filter by applying filters at formation date only
        (no look-ahead bias). Filters set signal to NaN for excluded observations,
        so they won't be ranked. Returns are collected from ALL bonds that were
        assigned to portfolios, regardless of their filter status at return date.

        Supports non-staggered rebalancing (Phase 15):
        - quarterly, semi-annual, annual rebalancing frequencies
        - Uses specialized numba kernels for non-staggered return computation
        """
        # Route to non-staggered fast path if applicable
        if self._is_nonstaggered:
            return self._fit_fast_batch_nonstaggered()

        import numpy as np
        from .numba_core import (
            compute_ranks_all_signals,
            build_rank_lookups_all_signals,
            compute_ls_returns_all_signals_hp1,
            compute_ls_returns_all_signals_staggered,
            build_vw_lookup_and_dynamic_weights
        )
        from .constants import RatingBounds

        results = BatchResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        t_start = time.time()

        filter_desc = []
        if self.rating is not None:
            filter_desc.append(f"rating={self.rating}")
        if self.subset_filter is not None:
            filter_desc.append(f"subset_filter={list(self.subset_filter.keys())}")
        filter_str = f" with filters: {', '.join(filter_desc)}" if filter_desc else ""

        if self.verbose:
            print(f"FAST BATCH PATH: Processing {len(self.signals)} signals with numba{filter_str}...")

        # =====================================================================
        # Step 1: Extract numpy arrays from DataFrame (ONCE)
        # =====================================================================
        t_extract = time.time()
        data = self.data

        # Build date and ID mappings
        dates = data['date'].unique()
        dates = np.sort(dates)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        n_dates = len(dates)

        ids = data['ID'].unique()
        id_to_idx = {bond_id: i for i, bond_id in enumerate(ids)}
        n_ids = len(ids)

        # Extract arrays
        date_idx = data['date'].map(date_to_idx).values.astype(np.int64)
        id_idx = data['ID'].map(id_to_idx).values.astype(np.int64)
        ret = data['ret'].values.astype(np.float64)
        vw = data['VW'].values.astype(np.float64)

        # =====================================================================
        # Step 1b: Build filter mask (Phase 14 - avoid look-ahead bias)
        # =====================================================================
        # Filter mask: True = observation passes filter at this date
        # We apply filter by setting signal to NaN for filtered-out observations.
        # This means filtered bonds won't be ranked at formation date.
        # BUT their returns are still collected if they were in a portfolio.
        filter_mask = np.ones(len(data), dtype=np.bool_)

        if self.rating is not None:
            rating_vals = data['RATING_NUM'].values
            if self.rating == 'IG':
                filter_mask &= (rating_vals <= RatingBounds.IG_MAX)
            elif self.rating == 'NIG':
                filter_mask &= (rating_vals > RatingBounds.IG_MAX)
            elif isinstance(self.rating, (tuple, list)):
                min_r, max_r = self.rating
                filter_mask &= (rating_vals >= min_r) & (rating_vals <= max_r)

        if self.subset_filter is not None:
            for col, (min_val, max_val) in self.subset_filter.items():
                if col not in data.columns:
                    raise ValueError(f"subset_filter column '{col}' not found in data")
                col_vals = data[col].values
                filter_mask &= (col_vals >= min_val) & (col_vals <= max_val)

        n_filtered = (~filter_mask).sum()
        if self.verbose and n_filtered > 0:
            pct_filtered = 100 * n_filtered / len(data)
            print(f"    Filter excludes {n_filtered:,} observations ({pct_filtered:.1f}%) from ranking")

        # Build signal matrix (n_obs, n_signals)
        # Apply filter mask: set signal to NaN for filtered-out observations
        n_signals = len(self.signals)
        signals_matrix = np.empty((len(data), n_signals), dtype=np.float64)
        for s_idx, signal in enumerate(self.signals):
            sig_vals = data[signal].values.astype(np.float64)
            # Set signal to NaN for observations that don't pass filter
            # This ensures they won't be ranked (rank computation skips NaN)
            sig_vals[~filter_mask] = np.nan
            signals_matrix[:, s_idx] = sig_vals

        # Build VW from d-1 (dynamic weights) - use existing numba function
        # Note: VW is NOT filtered - we use VW from ALL observations
        vw_lag = build_vw_lookup_and_dynamic_weights(
            date_idx, id_idx, vw, n_dates, n_ids
        )

        if self.verbose:
            print(f"    Data extracted in {time.time() - t_extract:.2f}s")

        # =====================================================================
        # Step 2: Compute ranks for ALL signals in parallel
        # =====================================================================
        t_ranks = time.time()
        ranks_all = compute_ranks_all_signals(
            date_idx, signals_matrix, n_dates, self.num_portfolios, n_signals
        )
        if self.verbose:
            print(f"    Ranks computed in {time.time() - t_ranks:.2f}s")

        # =====================================================================
        # Step 3: Build rank lookups for ALL signals
        # =====================================================================
        t_lookup = time.time()
        rank_lookups = build_rank_lookups_all_signals(
            date_idx, id_idx, ranks_all, n_dates, n_ids, n_signals
        )
        if self.verbose:
            print(f"    Rank lookups built in {time.time() - t_lookup:.2f}s")

        # =====================================================================
        # Step 4: Compute returns for ALL signals
        # =====================================================================
        t_returns = time.time()
        if self.holding_period == 1:
            ew_ls, vw_ls = compute_ls_returns_all_signals_hp1(
                date_idx, id_idx, ret, vw_lag, rank_lookups,
                n_dates, n_ids, self.num_portfolios, n_signals
            )
        else:
            ew_ls, vw_ls = compute_ls_returns_all_signals_staggered(
                date_idx, id_idx, ret, vw_lag, rank_lookups,
                n_dates, n_ids, self.num_portfolios, n_signals,
                self.holding_period
            )
        if self.verbose:
            print(f"    Returns computed in {time.time() - t_returns:.2f}s")

        # =====================================================================
        # Step 5: Package results into BatchResults format
        # =====================================================================
        t_package = time.time()

        # Create date index for output Series
        date_index = pd.DatetimeIndex(dates)

        for s_idx, signal in enumerate(self.signals):
            try:
                # Create simple result object with long-short returns
                ew_series = pd.Series(ew_ls[:, s_idx], index=date_index, name='ew_ls')
                vw_series = pd.Series(vw_ls[:, s_idx], index=date_index, name='vw_ls')

                # Drop NaN values
                ew_series = ew_series.dropna()
                vw_series = vw_series.dropna()

                # Create a minimal result object
                result = _FastBatchResult(
                    ew_ls=ew_series,
                    vw_ls=vw_series,
                    signal=signal
                )

                results.results[signal] = result
                results.timings[signal] = 0.0  # Individual timing not available in batch

            except Exception as e:
                results.errors[signal] = str(e)

        if self.verbose:
            print(f"    Results packaged in {time.time() - t_package:.2f}s")

        results.timings['total'] = time.time() - t_start

        if self.verbose:
            print(f"FAST BATCH PATH completed in {results.timings['total']:.2f}s")

        return results

    def _fit_fast_batch_nonstaggered(self) -> BatchResults:
        """
        Ultra-fast batch processing for non-staggered rebalancing using numba kernels.

        For quarterly, semi-annual, or annual rebalancing:
        - Computes ranks only at rebalancing dates
        - Computes returns for all holding period months
        - Much simpler than staggered (no cohort averaging needed)
        """
        import numpy as np
        from .numba_core import (
            compute_ranks_at_rebal_dates,
            build_rank_lookup_nonstaggered,
            compute_nonstaggered_returns_fast,
            build_vw_lookup_table,
            compute_nonstaggered_ls_returns,
        )
        from .utils_optimized import _get_rebalancing_dates
        from .constants import RatingBounds

        results = BatchResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        t_start = time.time()

        # Build frequency description
        freq = self.rebalance_frequency
        if isinstance(freq, str):
            freq_str = freq
            freq_months = {'quarterly': 3, 'semi-annual': 6, 'annual': 12}.get(freq, 1)
        else:
            freq_months = freq
            freq_str = {3: 'quarterly', 6: 'semi-annual', 12: 'annual'}.get(freq, f'{freq}m')

        filter_desc = []
        if self.rating is not None:
            filter_desc.append(f"rating={self.rating}")
        if self.subset_filter is not None:
            filter_desc.append(f"subset_filter={list(self.subset_filter.keys())}")
        filter_str = f" with filters: {', '.join(filter_desc)}" if filter_desc else ""

        if self.verbose:
            print(f"FAST BATCH PATH (non-staggered {freq_str}): Processing {len(self.signals)} signals{filter_str}...")

        # =====================================================================
        # Step 1: Extract numpy arrays from DataFrame (ONCE)
        # =====================================================================
        t_extract = time.time()
        data = self.data

        # Build date and ID mappings
        dates = data['date'].unique()
        dates = np.sort(dates)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        n_dates = len(dates)

        ids = data['ID'].unique()
        id_to_idx = {bond_id: i for i, bond_id in enumerate(ids)}
        n_ids = len(ids)

        # Get rebalancing dates
        datelist = [pd.Timestamp(d) for d in dates]
        rebal_date_indices = _get_rebalancing_dates(datelist, freq_str, self.rebalance_month)
        rebal_date_indices = np.array(rebal_date_indices, dtype=np.int64)
        n_rebal = len(rebal_date_indices)

        if self.verbose:
            print(f"    Rebalancing dates: {n_rebal} (frequency={freq_str}, month={self.rebalance_month})")

        # Extract arrays
        date_idx = data['date'].map(date_to_idx).values.astype(np.int64)
        id_idx = data['ID'].map(id_to_idx).values.astype(np.int64)
        ret = data['ret'].values.astype(np.float64)
        vw = data['VW'].values.astype(np.float64)

        # =====================================================================
        # Step 1b: Build filter mask
        # =====================================================================
        filter_mask = np.ones(len(data), dtype=np.bool_)

        if self.rating is not None:
            rating_vals = data['RATING_NUM'].values
            if self.rating == 'IG':
                filter_mask &= (rating_vals <= RatingBounds.IG_MAX)
            elif self.rating == 'NIG':
                filter_mask &= (rating_vals > RatingBounds.IG_MAX)
            elif isinstance(self.rating, (tuple, list)):
                min_r, max_r = self.rating
                filter_mask &= (rating_vals >= min_r) & (rating_vals <= max_r)

        if self.subset_filter is not None:
            for col, (min_val, max_val) in self.subset_filter.items():
                if col not in data.columns:
                    raise ValueError(f"subset_filter column '{col}' not found in data")
                col_vals = data[col].values
                filter_mask &= (col_vals >= min_val) & (col_vals <= max_val)

        n_filtered = (~filter_mask).sum()
        if self.verbose and n_filtered > 0:
            pct_filtered = 100 * n_filtered / len(data)
            print(f"    Filter excludes {n_filtered:,} observations ({pct_filtered:.1f}%) from ranking")

        # Build signal matrix with filter applied
        n_signals = len(self.signals)
        signals_matrix = np.empty((len(data), n_signals), dtype=np.float64)
        for s_idx, signal in enumerate(self.signals):
            sig_vals = data[signal].values.astype(np.float64)
            sig_vals[~filter_mask] = np.nan
            signals_matrix[:, s_idx] = sig_vals

        if self.verbose:
            print(f"    Data extracted in {time.time() - t_extract:.2f}s")

        # =====================================================================
        # Step 2: Build VW lookup table
        # =====================================================================
        t_vw = time.time()
        vw_lookup = build_vw_lookup_table(date_idx, id_idx, vw, n_dates, n_ids)
        if self.verbose:
            print(f"    VW lookup built in {time.time() - t_vw:.2f}s")

        # =====================================================================
        # Step 3-5: Process each signal
        # =====================================================================
        t_signals = time.time()
        date_index = pd.DatetimeIndex(dates)

        for s_idx, signal in enumerate(self.signals):
            try:
                sig_vals = signals_matrix[:, s_idx]

                # Step 3: Compute ranks at rebalancing dates
                ranks = compute_ranks_at_rebal_dates(
                    date_idx, sig_vals, rebal_date_indices, n_dates, self.num_portfolios
                )

                # Step 4: Build rank lookup
                rank_lookup = build_rank_lookup_nonstaggered(
                    date_idx, id_idx, ranks, rebal_date_indices, n_dates, n_ids
                )

                # Step 5: Compute returns
                ew_ptf, vw_ptf = compute_nonstaggered_returns_fast(
                    date_idx, id_idx, ret, vw, rebal_date_indices,
                    self.holding_period, rank_lookup, n_dates, n_ids,
                    self.num_portfolios, True, vw_lookup
                )

                # Compute long-short returns
                ew_ls, vw_ls = compute_nonstaggered_ls_returns(
                    ew_ptf, vw_ptf, self.num_portfolios
                )

                # Create result
                ew_series = pd.Series(ew_ls, index=date_index, name='ew_ls').dropna()
                vw_series = pd.Series(vw_ls, index=date_index, name='vw_ls').dropna()

                result = _FastBatchResult(
                    ew_ls=ew_series,
                    vw_ls=vw_series,
                    signal=signal
                )

                results.results[signal] = result
                results.timings[signal] = 0.0

            except Exception as e:
                results.errors[signal] = str(e)

        if self.verbose:
            print(f"    All signals processed in {time.time() - t_signals:.2f}s")

        results.timings['total'] = time.time() - t_start

        if self.verbose:
            print(f"FAST BATCH PATH (non-staggered) completed in {results.timings['total']:.2f}s")

        return results

    def fit(self) -> BatchResults:
        """Run batch portfolio formation for all signals."""
        # Check if fast batch path can be used
        if self._can_use_fast_batch_path():
            return self._fit_fast_batch()

        results = BatchResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        t_start = time.time()
        n_workers = self._get_n_workers()

        if self.verbose:
            print(f"Processing {len(self.signals)} signals with {n_workers} worker(s)...")

        if n_workers == 1:
            # Sequential processing
            results = self._fit_sequential(results)
        else:
            # Parallel processing
            results = self._fit_parallel(results, n_workers)

        t_end = time.time()
        results.timings['total'] = t_end - t_start

        if self.verbose:
            self._print_summary(results)

        return results

    def _fit_sequential(self, results: BatchResults) -> BatchResults:
        """Sequential processing of signals."""
        # First, run one signal to get shared precompute
        shared_precomp = None

        if self.verbose and TQDM_AVAILABLE:
            signal_iter = tqdm(self.signals, desc="Processing", unit="signal")
        else:
            signal_iter = self.signals

        for i, signal in enumerate(signal_iter):
            t_signal_start = time.time()

            try:
                strategy = SingleSort(
                    holding_period=self.holding_period,
                    sort_var=signal,
                    num_portfolios=self.num_portfolios,
                    rebalance_frequency=self.rebalance_frequency,
                    rebalance_month=self.rebalance_month,
                    verbose=False
                )

                sf_config = StrategyFormationConfig(
                    data=DataConfig(
                        rating=self.rating,
                        subset_filter=self.subset_filter,
                        chars=self.chars
                    ),
                    formation=FormationConfig(
                        dynamic_weights=True,
                        compute_turnover=self.turnover,
                        banding_threshold=self.banding_threshold,
                        verbose=False,
                    )
                )

                sf = StrategyFormation(data=self.data, strategy=strategy, config=sf_config)

                if shared_precomp is not None:
                    sf._cached_precomp = shared_precomp

                result = sf.fit()

                # Extract shared precomp from first signal
                if shared_precomp is None and hasattr(sf, '_shareable_precomp'):
                    full_precomp = sf._shareable_precomp
                    shared_precomp = {
                        'It1': full_precomp.get('It1'),
                        'It2': full_precomp.get('It2'),
                        'It1m': full_precomp.get('It1m'),
                        'vw_map_t1m': full_precomp.get('vw_map_t1m'),
                    }

                results.results[signal] = result
                results.timings[signal] = time.time() - t_signal_start

                if self.verbose and not TQDM_AVAILABLE:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: {results.timings[signal]:.2f}s")

            except Exception as e:
                results.errors[signal] = str(e)
                if self.verbose:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: ERROR - {e}")

        return results

    def _get_minimal_data(self, signal: str) -> pd.DataFrame:
        """
        Extract only the required columns for a single signal.

        This reduces pickle size by 50-80% compared to sending the full DataFrame.
        """
        # Base required columns
        cols = list(REQUIRED_COLUMNS)

        # Add the signal column
        if signal not in cols:
            cols.append(signal)

        # Add characteristic columns if specified
        if self.chars:
            for char in self.chars:
                if char not in cols and char in self.data.columns:
                    cols.append(char)

        # Add subset_filter columns if specified
        if self.subset_filter:
            for col in self.subset_filter.keys():
                if col not in cols and col in self.data.columns:
                    cols.append(col)

        # Only include columns that exist in data
        cols = [c for c in cols if c in self.data.columns]

        return self.data[cols].copy()

    def _get_minimal_data_batch(self, signals: List[str]) -> pd.DataFrame:
        """
        Extract only required columns for a batch of signals.

        More efficient than calling _get_minimal_data for each signal
        when signals_per_worker > 1.
        """
        cols = list(REQUIRED_COLUMNS)

        # Add all signal columns
        for signal in signals:
            if signal not in cols:
                cols.append(signal)

        # Add characteristic columns if specified
        if self.chars:
            for char in self.chars:
                if char not in cols and char in self.data.columns:
                    cols.append(char)

        # Add subset_filter columns if specified
        if self.subset_filter:
            for col in self.subset_filter.keys():
                if col not in cols and col in self.data.columns:
                    cols.append(col)

        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def _fit_parallel(self, results: BatchResults, n_workers: int) -> BatchResults:
        """Parallel processing of signals with chunking and batching support."""

        # Determine start method based on platform
        start_method = _get_start_method()
        if self.verbose:
            print(f"  Platform: {platform.system()}, using '{start_method}' start method")
            if self.signals_per_worker > 1:
                print(f"  Signals per worker: {self.signals_per_worker}")
            if self.chunk_size:
                print(f"  Chunk size: {self.chunk_size} signals")

        # First, run one signal sequentially (warmup + first result)
        first_signal = self.signals[0]
        remaining_signals = self.signals[1:]

        if self.verbose:
            print(f"  Running first signal (warmup)...")

        t0 = time.time()
        try:
            strategy = SingleSort(
                holding_period=self.holding_period,
                sort_var=first_signal,
                num_portfolios=self.num_portfolios,
                rebalance_frequency=self.rebalance_frequency,
                rebalance_month=self.rebalance_month,
                verbose=False
            )
            sf_config = StrategyFormationConfig(
                data=DataConfig(
                    rating=self.rating,
                    subset_filter=self.subset_filter,
                    chars=self.chars
                ),
                formation=FormationConfig(
                    dynamic_weights=True,
                    compute_turnover=self.turnover,
                    banding_threshold=self.banding_threshold,
                    verbose=False,
                )
            )
            sf = StrategyFormation(data=self.data, strategy=strategy, config=sf_config)
            first_result = sf.fit()

            results.results[first_signal] = first_result
            results.timings[first_signal] = time.time() - t0

            if self.verbose:
                print(f"  First signal done: {results.timings[first_signal]:.2f}s")

        except Exception as e:
            results.errors[first_signal] = str(e)
            if self.verbose:
                print(f"  First signal FAILED: {e}")

        if not remaining_signals:
            return results

        # Determine effective chunk size
        effective_chunk_size = self.chunk_size if self.chunk_size else len(remaining_signals)

        # Process remaining signals in chunks
        total_remaining = len(remaining_signals)
        processed = 0

        for chunk_start in range(0, total_remaining, effective_chunk_size):
            chunk_end = min(chunk_start + effective_chunk_size, total_remaining)
            chunk_signals = remaining_signals[chunk_start:chunk_end]

            if self.verbose:
                if self.chunk_size:
                    print(f"\n  Processing chunk {chunk_start // effective_chunk_size + 1} "
                          f"({len(chunk_signals)} signals)...")
                else:
                    print(f"  Processing {len(chunk_signals)} remaining signals in parallel...")

            # Process this chunk
            self._process_chunk(
                chunk_signals, results, n_workers, start_method, processed, total_remaining
            )
            processed += len(chunk_signals)

            # Memory cleanup between chunks
            if self.chunk_size and chunk_end < total_remaining:
                gc.collect()

        return results

    def _process_chunk(self, signals: List[str], results: BatchResults,
                       n_workers: int, start_method: str,
                       offset: int, total: int):
        """Process a chunk of signals in parallel."""

        # Group signals into batches for workers
        if self.signals_per_worker > 1:
            # Batch mode: group signals for each worker
            signal_batches = []
            for i in range(0, len(signals), self.signals_per_worker):
                batch = signals[i:i + self.signals_per_worker]
                signal_batches.append(batch)

            # Prepare worker args with batched data
            worker_args = []
            for batch in signal_batches:
                batch_data = self._get_minimal_data_batch(batch)
                worker_args.append((
                    batch, batch_data, self.holding_period, self.num_portfolios,
                    self.turnover, self.chars, self.rating, self.subset_filter,
                    self.banding_threshold, self.rebalance_frequency, self.rebalance_month
                ))

            if self.verbose and offset == 0:
                # Show data reduction stats on first chunk
                full_size = self.data.memory_usage(deep=True).sum() / 1024 / 1024
                batch_size = worker_args[0][1].memory_usage(deep=True).sum() / 1024 / 1024
                reduction = (1 - batch_size / full_size) * 100
                print(f"  Data size: {full_size:.1f}MB → {batch_size:.1f}MB per worker ({reduction:.0f}% reduction)")
                print(f"  Worker batches: {len(worker_args)} (processing {len(signals)} signals)")

            # Execute batched workers
            mp_context = mp.get_context(start_method)
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                future_to_batch = {
                    executor.submit(_process_signal_batch, args): args[0]
                    for args in worker_args
                }

                completed = 0
                for future in as_completed(future_to_batch):
                    batch_signals = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        for sig_name, result, elapsed, error in batch_results:
                            if error is None:
                                results.results[sig_name] = result
                                results.timings[sig_name] = elapsed
                            else:
                                results.errors[sig_name] = error
                            completed += 1

                            if self.verbose and not TQDM_AVAILABLE:
                                status = "OK" if error is None else f"ERROR"
                                print(f"  [{offset + completed}/{total}] {sig_name}: {status}")

                    except Exception as e:
                        for sig in batch_signals:
                            results.errors[sig] = str(e)
                        completed += len(batch_signals)

        else:
            # Single signal mode (original behavior)
            worker_args = []
            for signal in signals:
                minimal_data = self._get_minimal_data(signal)
                worker_args.append((
                    signal, minimal_data, self.holding_period, self.num_portfolios,
                    self.turnover, self.chars, self.rating, self.subset_filter,
                    self.banding_threshold, self.rebalance_frequency, self.rebalance_month
                ))

            if self.verbose and offset == 0:
                full_size = self.data.memory_usage(deep=True).sum() / 1024 / 1024
                min_size = worker_args[0][1].memory_usage(deep=True).sum() / 1024 / 1024
                reduction = (1 - min_size / full_size) * 100
                print(f"  Data size: {full_size:.1f}MB → {min_size:.1f}MB per worker ({reduction:.0f}% reduction)")

            mp_context = mp.get_context(start_method)
            completed = 0
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                future_to_signal = {
                    executor.submit(_process_single_signal, args): args[0]
                    for args in worker_args
                }

                if self.verbose and TQDM_AVAILABLE:
                    futures_iter = tqdm(
                        as_completed(future_to_signal),
                        total=len(signals),
                        desc="Parallel processing"
                    )
                else:
                    futures_iter = as_completed(future_to_signal)

                for future in futures_iter:
                    signal = future_to_signal[future]
                    try:
                        sig_name, result, elapsed, error = future.result()
                        if error is None:
                            results.results[sig_name] = result
                            results.timings[sig_name] = elapsed
                        else:
                            results.errors[sig_name] = error
                        completed += 1

                        if self.verbose and not TQDM_AVAILABLE:
                            status = "OK" if error is None else f"ERROR: {error[:30]}"
                            print(f"  [{offset + completed}/{total}] {sig_name}: {status}")

                    except Exception as e:
                        results.errors[signal] = str(e)
                        completed += 1

    def _print_summary(self, results: BatchResults):
        """Print summary of batch processing."""
        n_workers = self._get_n_workers()
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total signals:    {len(self.signals)}")
        print(f"Successful:       {len(results.results)}")
        print(f"Failed:           {len(results.errors)}")
        print(f"Workers used:     {n_workers}")
        print(f"Total time:       {results.timings.get('total', 0):.2f}s")

        if results.results:
            avg_time = sum(results.timings.get(s, 0) for s in results.results) / len(results.results)
            print(f"Avg time/signal:  {avg_time:.2f}s")

            # Compute effective speedup
            sequential_estimate = avg_time * len(self.signals)
            actual_time = results.timings.get('total', 0)
            if actual_time > 0:
                speedup = sequential_estimate / actual_time
                print(f"Effective speedup: {speedup:.1f}x")

        if results.errors:
            print(f"\nFailed signals: {list(results.errors.keys())}")

        print(f"{'='*60}")


# =============================================================================
# Convenience function
# =============================================================================

def batch_single_sort(
    data: pd.DataFrame,
    signals: List[str],
    holding_period: int = 1,
    num_portfolios: int = 5,
    turnover: bool = True,
    n_jobs: int = -1,
    **kwargs
) -> BatchResults:
    """
    Convenience function for batch single-sort portfolio formation.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signals : list of str
        Signal column names
    holding_period : int
        Holding period
    num_portfolios : int
        Number of portfolios
    turnover : bool
        Compute turnover
    n_jobs : int
        Number of parallel workers (-1 for all cores)
    **kwargs
        Additional arguments passed to BatchStrategyFormation

    Returns
    -------
    BatchResults
        Batch results container
    """
    batch_sf = BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=holding_period,
        num_portfolios=num_portfolios,
        turnover=turnover,
        n_jobs=n_jobs,
        **kwargs
    )
    return batch_sf.fit()
