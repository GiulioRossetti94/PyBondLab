# -*- coding: utf-8 -*-
"""
BatchWithinFirmSortFormation: Multi-signal processing for WithinFirmSort.

Provides fast batch processing of multiple signals using:
- Ultra-fast numba path when turnover=False and chars=None
- Multiprocessing slow path when turnover=True or chars is set

Aligned with BatchStrategyFormation for consistent user experience:
- Column mapping with verbose output
- Warmup run for JIT compilation
- tqdm progress bars
- Memory optimization
- Summary output with timing statistics

Author: Giulio Rossetti & Alex Dickerson
Date: 2024
"""

import gc
import platform
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import pandas as pd

from .batch_base import (
    BaseBatchFormation,
    _get_start_method,
    _is_windows,
    TQDM_AVAILABLE,
    tqdm,
    REQUIRED_COLUMNS,
    DEFAULT_COLUMNS,
    _estimate_memory_components,
    _estimate_peak_memory_mb,
    _get_available_memory_mb,
    _suggest_parallel_config,
    _print_memory_config,
)
from .StrategyClass import WithinFirmSort
from .PyBondLab import StrategyFormation
from .config import StrategyFormationConfig, DataConfig, FormationConfig
from .constants import ColumnNames


# =============================================================================
# Worker function for parallel processing (must be at module level for pickle)
# =============================================================================

def _process_withinfirm_signal(args: Tuple) -> Tuple[str, Any, float, Optional[str]]:
    """
    Process a single WithinFirmSort signal - worker function for parallel execution.

    Parameters
    ----------
    args : tuple
        (signal, data, firm_id_col, rating_bins, min_bonds_per_firm,
         turnover, chars, rating)

    Returns
    -------
    tuple
        (signal_name, result_or_none, elapsed_time, error_or_none)
    """
    (signal, data, firm_id_col, rating_bins, min_bonds_per_firm,
     turnover, chars, rating) = args

    t_start = time.time()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            strategy = WithinFirmSort(
                holding_period=1,
                sort_var=signal,
                firm_id_col=firm_id_col,
                min_bonds_per_firm=min_bonds_per_firm,
                rating_bins=rating_bins,
                num_portfolios=2,
                verbose=False
            )

            sf_config = StrategyFormationConfig(
                data=DataConfig(rating=rating, chars=chars),
                formation=FormationConfig(
                    dynamic_weights=True,
                    compute_turnover=turnover,
                    verbose=False,
                )
            )

            sf = StrategyFormation(data=data, strategy=strategy, config=sf_config)
            result = sf.fit()

            elapsed = time.time() - t_start
            return (signal, result, elapsed, None)

    except Exception as e:
        elapsed = time.time() - t_start
        return (signal, None, elapsed, str(e))


def _process_withinfirm_batch(args: Tuple) -> List[Tuple[str, Any, float, Optional[str]]]:
    """
    Process a batch of WithinFirmSort signals - worker function for parallel execution.

    Parameters
    ----------
    args : tuple
        (signals, data, firm_id_col, rating_bins, min_bonds_per_firm,
         turnover, chars, rating)

    Returns
    -------
    list
        List of (signal_name, result_or_none, elapsed_time, error_or_none) tuples
    """
    (signals, data, firm_id_col, rating_bins, min_bonds_per_firm,
     turnover, chars, rating) = args

    results = []
    for signal in signals:
        t_start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                strategy = WithinFirmSort(
                    holding_period=1,
                    sort_var=signal,
                    firm_id_col=firm_id_col,
                    min_bonds_per_firm=min_bonds_per_firm,
                    rating_bins=rating_bins,
                    num_portfolios=2,
                    verbose=False
                )

                sf_config = StrategyFormationConfig(
                    data=DataConfig(rating=rating, chars=chars),
                    formation=FormationConfig(
                        dynamic_weights=True,
                        compute_turnover=turnover,
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
class BatchWithinFirmResults:
    """
    Container for batch WithinFirmSort results.

    Provides dictionary-like access to individual signal results.
    """

    results: OrderedDict = field(default_factory=OrderedDict)
    signals: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, signal: str):
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
                ew_ls, vw_ls = result.get_long_short()
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
                rows.append(row)
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.set_index('signal')
        return df


# =============================================================================
# Batch WithinFirmSort Formation
# =============================================================================

class BatchWithinFirmSortFormation(BaseBatchFormation):
    """
    Batch processing for WithinFirmSort with multiple signals.

    Processes multiple signals efficiently using:
    - Fast numba path: When turnover=False and chars=None, processes all signals
      in parallel using vectorized numba kernels
    - Slow multiprocessing path: When turnover=True or chars is set, uses
      ProcessPoolExecutor to run StrategyFormation for each signal

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with columns: date, ID, ret, VW, RATING_NUM, PERMNO, and signal columns
    signals : List[str]
        Column names to use as sorting signals
    firm_id_col : str, default='PERMNO'
        Column name for firm identifier
    rating_bins : list, optional
        Rating bin edges for creating rating terciles
        Default: [-np.inf, 7, 10, np.inf] (IG/BBB/NIG)
    min_bonds_per_firm : int, default=2
        Minimum bonds per firm-date-rating group
    turnover : bool, default=False
        Compute turnover statistics (uses slow path)
    chars : List[str], optional
        Characteristics to aggregate (uses slow path)
    rating : str or tuple, optional
        Rating filter: 'IG', 'NIG', or (min, max) tuple
    subset_filter : Dict[str, Tuple[float, float]], optional
        Characteristic-based filters: {col_name: (min, max)}
    columns : Dict[str, str], optional
        Column name mapping: {'pbl_name': 'your_col_name'}
        Example: {'ID': 'cusip', 'ret': 'ret_vw', 'VW': 'mcap_e', 'RATING_NUM': 'spc_rat'}
    n_jobs : int, default=1
        Number of parallel workers (for slow path)
    signals_per_worker : int, default=1
        Number of signals per worker (reduces overhead)
    chunk_size : int, optional
        Process in chunks to limit memory. If 'auto', automatically determines
        based on available system memory.
    verbose : bool, default=True
        Show progress output

    Examples
    --------
    >>> batch = BatchWithinFirmSortFormation(
    ...     data=data,
    ...     signals=['signal1', 'signal2', 'signal3'],
    ...     columns={'ID': 'cusip', 'ret': 'ret_vw', 'VW': 'mcap_e'},
    ...     firm_id_col='PERMNO',
    ...     turnover=False,
    ...     n_jobs=4,
    ...     verbose=True
    ... )
    >>> results = batch.fit()
    >>> ew_ls, vw_ls = results['signal1'].get_long_short()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: List[str],
        firm_id_col: str = 'PERMNO',
        rating_bins: Optional[List[float]] = None,
        min_bonds_per_firm: int = 2,
        turnover: bool = False,
        chars: Optional[List[str]] = None,
        rating: Optional[Union[str, Tuple[int, int]]] = None,
        subset_filter: Optional[Dict[str, Tuple[float, float]]] = None,
        columns: Optional[Dict[str, str]] = None,
        n_jobs: int = 1,
        signals_per_worker: int = 1,
        chunk_size: Optional[Union[int, str]] = None,
        verbose: bool = True,
    ):
        # Validate parameter types (catch common mistakes early)
        if not isinstance(signals, (list, tuple)):
            raise TypeError(
                f"signals must be a list of column names, got {type(signals).__name__}: {signals!r}. "
                f"Example: signals=['signal1', 'signal2']"
            )
        if signals and not all(isinstance(s, str) for s in signals):
            bad = [s for s in signals if not isinstance(s, str)]
            raise TypeError(
                f"All signals must be strings (column names), got non-string values: {bad}"
            )
        if not isinstance(firm_id_col, str):
            raise TypeError(
                f"firm_id_col must be a string (column name), got {type(firm_id_col).__name__}: {firm_id_col!r}"
            )
        if not isinstance(min_bonds_per_firm, (int, np.integer)):
            raise TypeError(
                f"min_bonds_per_firm must be an integer, got {type(min_bonds_per_firm).__name__}: {min_bonds_per_firm!r}"
            )
        if not isinstance(n_jobs, (int, np.integer)):
            raise TypeError(
                f"n_jobs must be an integer, got {type(n_jobs).__name__}: {n_jobs!r}"
            )

        # WithinFirmSort-specific parameters
        self.firm_id_col = firm_id_col
        self.rating_bins = rating_bins if rating_bins is not None else [-np.inf, 7, 10, np.inf]
        self.min_bonds_per_firm = min_bonds_per_firm
        self.turnover = turnover
        self.chars = chars
        self.rating = rating
        self.subset_filter = subset_filter

        # Required columns for memory estimation (includes firm_id_col)
        required_cols = ['date', 'ID', 'ret', 'VW', 'RATING_NUM', firm_id_col]
        if chars:
            required_cols = list(set(required_cols + chars))

        # Handle n_jobs
        requested_workers = self._get_n_workers_from_param(n_jobs)

        # Auto-tune parallel configuration
        effective_chunk_size = None
        effective_signals_per_worker = max(1, signals_per_worker)
        effective_max_in_flight = None

        if chunk_size == 'auto' or (n_jobs != 1 and len(signals) > 5):
            # Use auto-tuning for parallel processing
            parallel_config = _suggest_parallel_config(
                data, len(signals), requested_workers,
                required_columns=required_cols, verbose=False
            )

            if chunk_size == 'auto':
                # Full auto mode: use all suggested values
                effective_n_jobs = parallel_config['n_workers']
                effective_chunk_size = parallel_config['chunk_size']
                effective_signals_per_worker = parallel_config['signals_per_worker']
                effective_max_in_flight = parallel_config['max_in_flight']

                if verbose:
                    _print_memory_config(parallel_config, len(signals), verbose=True)
            else:
                # Manual chunk_size but still use auto-tuning for other params
                effective_n_jobs = n_jobs
                if isinstance(chunk_size, (int, np.integer)):
                    effective_chunk_size = int(chunk_size)
                effective_signals_per_worker = max(1, signals_per_worker) if signals_per_worker != 1 else parallel_config['signals_per_worker']
                effective_max_in_flight = parallel_config['max_in_flight']

                # Print warnings if memory is tight
                for warning in parallel_config['warnings']:
                    if verbose:
                        print(f"  [!] {warning}")
        else:
            # Sequential or small batch - no auto-tuning needed
            effective_n_jobs = n_jobs
            if isinstance(chunk_size, (int, np.integer)):
                effective_chunk_size = int(chunk_size)

        # Initialize base class
        super().__init__(
            data=data,
            signals=signals,
            columns=columns,
            n_jobs=effective_n_jobs,
            signals_per_worker=effective_signals_per_worker,
            chunk_size=effective_chunk_size,
            verbose=verbose,
        )

        # Store max_in_flight for lazy arg preparation
        self.max_in_flight = effective_max_in_flight

        # Store config
        self.config = {
            'firm_id_col': firm_id_col,
            'rating_bins': self.rating_bins,
            'min_bonds_per_firm': min_bonds_per_firm,
            'turnover': turnover,
            'chars': chars,
            'rating': rating,
            'subset_filter': subset_filter,
            'n_jobs': effective_n_jobs,
            'signals_per_worker': effective_signals_per_worker,
            'chunk_size': effective_chunk_size,
            'is_within_firm': True,
            'holding_period': 1,
        }

    def _get_n_workers_from_param(self, n_jobs: int) -> int:
        """Convert n_jobs parameter to actual worker count."""
        if n_jobs == 1:
            return 1
        elif n_jobs == -1:
            return mp.cpu_count()
        elif n_jobs < -1:
            return max(1, mp.cpu_count() + 1 + n_jobs)
        else:
            return min(n_jobs, mp.cpu_count())

    def _get_required_columns(self) -> List[str]:
        """Get required columns including firm ID."""
        cols = REQUIRED_COLUMNS.copy()
        if self.firm_id_col not in cols:
            cols.append(self.firm_id_col)
        return cols

    def _can_use_fast_path(self) -> bool:
        """Check if fast batch path can be used."""
        if self.turnover:
            return False
        if self.chars is not None:
            return False
        return True

    def _get_minimal_data(self, signal: str) -> pd.DataFrame:
        """Extract minimal columns for a single signal."""
        cols = self._get_required_columns()
        if signal not in cols:
            cols.append(signal)
        if self.chars:
            for char in self.chars:
                if char not in cols and char in self.data.columns:
                    cols.append(char)
        if self.subset_filter:
            for col in self.subset_filter.keys():
                if col not in cols and col in self.data.columns:
                    cols.append(col)
        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def _get_minimal_data_batch(self, signals: List[str]) -> pd.DataFrame:
        """Extract minimal columns for a batch of signals."""
        cols = self._get_required_columns()
        for signal in signals:
            if signal not in cols:
                cols.append(signal)
        if self.chars:
            for char in self.chars:
                if char not in cols and char in self.data.columns:
                    cols.append(char)
        if self.subset_filter:
            for col in self.subset_filter.keys():
                if col not in cols and col in self.data.columns:
                    cols.append(col)
        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def fit(self) -> BatchWithinFirmResults:
        """
        Run batch portfolio formation for all signals.

        Returns
        -------
        BatchWithinFirmResults
            Container with results for all signals
        """
        # Check if fast batch path can be used
        if self._can_use_fast_path():
            return self._fit_fast_batch()

        # Reset timing
        self.timings = {}
        self.errors = {}

        t_start = time.time()
        n_workers = self._get_n_workers()

        if self.verbose:
            print(f"Processing {len(self.signals)} signals with {n_workers} worker(s)...")
            if self.turnover:
                print("  (turnover=True requires slow path)")
            if self.chars:
                print(f"  (chars={self.chars} requires slow path)")

        if n_workers == 1:
            results = self._fit_sequential()
        else:
            results = self._fit_parallel(n_workers)

        total_time = time.time() - t_start
        results.timings['total'] = total_time

        if self.verbose:
            n_success = len(results.results)
            n_failed = len(results.errors)
            self._print_summary(results, n_success, n_failed, total_time)

        return results

    def _fit_fast_batch(self) -> BatchWithinFirmResults:
        """
        Ultra-fast batch processing using vectorized numba kernels.

        Processes all signals in parallel:
        1. Pre-compute rating terciles (ONCE)
        2. Pre-compute firm groupings (ONCE)
        3. Compute HIGH/LOW assignments for ALL signals at once
        4. Aggregate returns for ALL signals in parallel
        """
        from .numba_core import (
            compute_withinfirm_assignments_all_dates,
            compute_within_firm_aggregation_with_lookup
        )
        from .PyBondLab import build_strategy_results

        t_start = time.time()

        results = BatchWithinFirmResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        if self.verbose:
            print(f"FAST BATCH PATH: Processing {len(self.signals)} signals...")

        # Get date list
        datelist = sorted(self.data[ColumnNames.DATE].unique())
        date_to_idx = {d: i for i, d in enumerate(datelist)}
        n_dates = len(datelist)

        # Filter to valid dates
        valid_mask = self.data[ColumnNames.DATE].isin(datelist)
        data = self.data[valid_mask].copy()

        if data.empty:
            if self.verbose:
                print("No valid data - returning empty results")
            return results

        # Apply rating filter if specified
        if self.rating is not None:
            from .PyBondLab import get_rating_bounds
            if isinstance(self.rating, str):
                min_r, max_r = get_rating_bounds(self.rating)
            else:
                min_r, max_r = self.rating
            data = data[
                (data[ColumnNames.RATING] >= min_r) &
                (data[ColumnNames.RATING] <= max_r)
            ]

        # Apply subset filter if specified
        if self.subset_filter is not None:
            for col, (min_val, max_val) in self.subset_filter.items():
                data = data[(data[col] >= min_val) & (data[col] <= max_val)]

        if data.empty:
            if self.verbose:
                print("No data after filtering - returning empty results")
            return results

        # Create ID mapping
        unique_ids = data[ColumnNames.ID].unique()
        id_to_idx = {id_: i for i, id_ in enumerate(unique_ids)}
        n_ids = len(unique_ids)

        # Create firm mapping
        unique_firms = data[self.firm_id_col].dropna().unique()
        firm_to_idx = {f: i for i, f in enumerate(unique_firms)}
        n_firms = len(unique_firms)

        # Create rating terciles (ONCE for all signals)
        rating_terc = pd.cut(
            pd.to_numeric(data[ColumnNames.RATING], errors='coerce'),
            bins=self.rating_bins,
            labels=[1, 2, 3],
            include_lowest=True
        ).astype(float).fillna(0).values.astype(np.int64)

        # Convert to numpy arrays (ONCE)
        date_idx = data[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
        id_idx = data[ColumnNames.ID].map(id_to_idx).values.astype(np.int64)
        firm_idx = data[self.firm_id_col].astype(object).map(firm_to_idx).fillna(-1).values.astype(np.int64)
        ret = data[ColumnNames.RETURN].values.astype(np.float64)
        vw = data[ColumnNames.VALUE_WEIGHT].values.astype(np.float64)

        # Sort by (date, rating_terc, firm) for group processing (ONCE)
        sort_order = np.lexsort((firm_idx, rating_terc, date_idx))
        date_idx_sorted = date_idx[sort_order]
        rating_terc_sorted = rating_terc[sort_order].astype(np.float64)
        firm_idx_sorted = firm_idx[sort_order]
        id_idx_sorted = id_idx[sort_order]
        vw_sorted = vw[sort_order]

        # Find group boundaries (ONCE for all signals)
        n_obs = len(date_idx_sorted)
        group_keys = date_idx_sorted * 1000000 + rating_terc_sorted.astype(np.int64) * 10000 + firm_idx_sorted
        group_changes = np.concatenate([
            [0],
            np.where(np.diff(group_keys) != 0)[0] + 1,
            [n_obs]
        ])
        group_starts = group_changes[:-1].astype(np.int64)
        group_ends = group_changes[1:].astype(np.int64)

        # Build VW lookup (ONCE for all signals)
        vw_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)
        for i in range(len(data)):
            d = date_idx[i]
            b = id_idx[i]
            vw_lookup[d, b] = vw[i]

        if self.verbose:
            print(f"  Data preparation: {time.time() - t_start:.2f}s")

        t_signals = time.time()

        # Process each signal with progress bar
        if self.verbose and TQDM_AVAILABLE:
            signal_iter = tqdm(enumerate(self.signals), total=len(self.signals), desc="Processing signals")
        else:
            signal_iter = enumerate(self.signals)

        for sig_idx, signal_name in signal_iter:
            t_sig_start = time.time()

            try:
                # Get signal values (apply sort order)
                signal_raw = data[signal_name].values.astype(np.float64)
                signal_sorted = signal_raw[sort_order]

                # Compute HIGH/LOW assignments for this signal
                ptf_rank = compute_withinfirm_assignments_all_dates(
                    signal_sorted, vw_sorted, group_starts, group_ends, self.min_bonds_per_firm
                )

                # Build rank lookup table
                rank_lookup = np.zeros((n_dates, n_ids, 3), dtype=np.float64)
                for i in range(n_obs):
                    d = date_idx_sorted[i]
                    b = id_idx_sorted[i]
                    rank_lookup[d, b, 0] = ptf_rank[i]
                    rank_lookup[d, b, 1] = rating_terc_sorted[i]
                    rank_lookup[d, b, 2] = firm_idx_sorted[i]

                # Aggregate returns
                (ew_long_short, vw_long_short,
                 ew_high_ret, ew_low_ret,
                 vw_high_ret, vw_low_ret) = compute_within_firm_aggregation_with_lookup(
                    date_idx, id_idx, firm_idx, ret, vw,
                    rank_lookup, vw_lookup, n_dates, n_ids, n_firms
                )

                # Build result DataFrames
                ptf_labels = ['LOW', 'HIGH']

                ew_port = pd.DataFrame(
                    np.column_stack([ew_low_ret, ew_high_ret]),
                    index=datelist,
                    columns=ptf_labels
                )
                vw_port = pd.DataFrame(
                    np.column_stack([vw_low_ret, vw_high_ret]),
                    index=datelist,
                    columns=ptf_labels
                )

                prefix = 'EWEA'
                vw_prefix = 'VWEA'

                ewls_df = pd.DataFrame(ew_long_short, index=datelist, columns=[f'{prefix}_{signal_name}'])
                vwls_df = pd.DataFrame(vw_long_short, index=datelist, columns=[f'{vw_prefix}_{signal_name}'])
                ew_long_df = pd.DataFrame(ew_high_ret, index=datelist, columns=[f'LONG_{prefix}_{signal_name}'])
                vw_long_df = pd.DataFrame(vw_high_ret, index=datelist, columns=[f'LONG_{vw_prefix}_{signal_name}'])
                ew_short_df = pd.DataFrame(ew_low_ret, index=datelist, columns=[f'SHORT_{prefix}_{signal_name}'])
                vw_short_df = pd.DataFrame(vw_low_ret, index=datelist, columns=[f'SHORT_{vw_prefix}_{signal_name}'])

                # Build StrategyResults
                result = build_strategy_results(
                    ewport_df=ew_port,
                    vwport_df=vw_port,
                    ewls_df=ewls_df,
                    vwls_df=vwls_df,
                    ewls_long_df=ew_long_df,
                    vwls_long_df=vw_long_df,
                    ewls_short_df=ew_short_df,
                    vwls_short_df=vw_short_df,
                    turnover_ew_df=None,
                    turnover_vw_df=None,
                    chars_ew=None,
                    chars_vw=None,
                )

                # Create result wrapper
                results.results[signal_name] = _BatchResult(result, datelist, signal_name)
                results.timings[signal_name] = time.time() - t_sig_start

            except Exception as e:
                results.errors[signal_name] = str(e)
                if self.verbose and not TQDM_AVAILABLE:
                    warnings.warn(f"Error processing {signal_name}: {e}", RuntimeWarning, stacklevel=2)

        if self.verbose:
            print(f"  Signal processing: {time.time() - t_signals:.2f}s")

        total_time = time.time() - t_start
        results.timings['total'] = total_time

        if self.verbose:
            n_success = len(results.results)
            n_failed = len(results.errors)
            self._print_summary(results, n_success, n_failed, total_time)

        return results

    def _fit_slow_batch(self) -> BatchWithinFirmResults:
        """Slow batch - called by _fit_sequential and _fit_parallel."""
        # This is handled by _fit_sequential and _fit_parallel
        pass

    def _fit_sequential(self) -> BatchWithinFirmResults:
        """Sequential processing of signals with warmup."""
        results = BatchWithinFirmResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        if self.verbose and TQDM_AVAILABLE:
            signal_iter = tqdm(self.signals, desc="Processing", unit="signal")
        else:
            signal_iter = self.signals

        for i, signal in enumerate(signal_iter):
            t_signal_start = time.time()

            try:
                strategy = WithinFirmSort(
                    holding_period=1,
                    sort_var=signal,
                    firm_id_col=self.firm_id_col,
                    min_bonds_per_firm=self.min_bonds_per_firm,
                    rating_bins=self.rating_bins,
                    num_portfolios=2,
                    verbose=False
                )

                sf_config = StrategyFormationConfig(
                    data=DataConfig(
                        rating=self.rating,
                        chars=self.chars,
                    ),
                    formation=FormationConfig(
                        dynamic_weights=True,
                        compute_turnover=self.turnover,
                        verbose=False,
                    )
                )

                sf = StrategyFormation(data=self.data, strategy=strategy, config=sf_config)
                result = sf.fit()

                results.results[signal] = result
                results.timings[signal] = time.time() - t_signal_start

                if self.verbose and not TQDM_AVAILABLE:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: {results.timings[signal]:.2f}s")

            except Exception as e:
                results.errors[signal] = str(e)
                if self.verbose:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: ERROR - {e}")

            gc.collect()

        return results

    def _fit_parallel(self, n_workers: int) -> BatchWithinFirmResults:
        """Parallel processing of signals with warmup and chunking."""
        results = BatchWithinFirmResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        # Platform-aware start method
        start_method = _get_start_method()
        if self.verbose:
            print(f"  Platform: {platform.system()}, using '{start_method}' start method")

        # Warmup: run first signal sequentially
        first_signal = self.signals[0]
        remaining_signals = self.signals[1:]

        if self.verbose:
            print(f"  Running first signal (warmup)...")

        t0 = time.time()
        try:
            strategy = WithinFirmSort(
                holding_period=1,
                sort_var=first_signal,
                firm_id_col=self.firm_id_col,
                min_bonds_per_firm=self.min_bonds_per_firm,
                rating_bins=self.rating_bins,
                num_portfolios=2,
                verbose=False
            )
            sf_config = StrategyFormationConfig(
                data=DataConfig(
                    rating=self.rating,
                    chars=self.chars,
                ),
                formation=FormationConfig(
                    dynamic_weights=True,
                    compute_turnover=self.turnover,
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

        # Determine effective chunk size (process in chunks to limit memory)
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

    def _prepare_single_arg(self, signal: str) -> Tuple:
        """Prepare worker argument for a single signal (lazy preparation)."""
        minimal_data = self._get_minimal_data(signal)
        return (
            signal, minimal_data, self.firm_id_col, self.rating_bins,
            self.min_bonds_per_firm, self.turnover, self.chars,
            self.rating
        )

    def _prepare_batch_arg(self, signals: List[str]) -> Tuple:
        """Prepare worker argument for a batch of signals (lazy preparation)."""
        batch_data = self._get_minimal_data_batch(signals)
        return (
            signals, batch_data, self.firm_id_col, self.rating_bins,
            self.min_bonds_per_firm, self.turnover, self.chars,
            self.rating
        )

    def _process_chunk(self, signals: List[str], results: BatchWithinFirmResults,
                       n_workers: int, start_method: str,
                       offset: int, total: int):
        """
        Process a chunk of signals in parallel with lazy arg preparation.

        Uses max_in_flight to limit memory by preparing worker args one at a time
        instead of all at once. This prevents memory spikes when processing
        many signals.
        """
        from concurrent.futures import wait, FIRST_COMPLETED

        # Determine max concurrent submissions
        max_in_flight = getattr(self, 'max_in_flight', None) or n_workers

        # Group signals into batches for workers (signals_per_worker)
        if self.signals_per_worker > 1:
            signal_batches = []
            for i in range(0, len(signals), self.signals_per_worker):
                batch = signals[i:i + self.signals_per_worker]
                signal_batches.append(batch)
            work_items = signal_batches
            is_batch_mode = True
        else:
            work_items = signals
            is_batch_mode = False

        # Show stats on first chunk (use sample data to show reduction)
        if self.verbose and offset == 0:
            full_size = self.data.memory_usage(deep=True).sum() / 1024 / 1024
            # Get sample minimal data size
            if is_batch_mode:
                sample_data = self._get_minimal_data_batch(work_items[0])
            else:
                sample_data = self._get_minimal_data(work_items[0])
            min_size = sample_data.memory_usage(deep=True).sum() / 1024 / 1024
            reduction = (1 - min_size / full_size) * 100
            print(f"  Data size: {full_size:.1f}MB -> {min_size:.1f}MB per worker ({reduction:.0f}% reduction)")
            if max_in_flight < len(work_items):
                print(f"  Max in-flight: {max_in_flight} (lazy arg preparation)")
            del sample_data  # Free sample

        mp_context = mp.get_context(start_method)
        completed = 0
        pending_futures = {}  # future -> work_item (signal or batch)

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            work_iter = iter(work_items)
            work_exhausted = False

            # Submit initial batch up to max_in_flight
            for _ in range(min(max_in_flight, len(work_items))):
                try:
                    item = next(work_iter)
                except StopIteration:
                    work_exhausted = True
                    break

                # Prepare and submit (lazy - one at a time)
                if is_batch_mode:
                    arg = self._prepare_batch_arg(item)
                    future = executor.submit(_process_withinfirm_batch, arg)
                else:
                    arg = self._prepare_single_arg(item)
                    future = executor.submit(_process_withinfirm_signal, arg)
                pending_futures[future] = item
                del arg  # Allow GC of the arg tuple

            # Process completions and submit new work
            while pending_futures:
                # Wait for at least one to complete
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    item = pending_futures.pop(future)

                    try:
                        if is_batch_mode:
                            batch_results = future.result()
                            for sig_name, result, elapsed, error in batch_results:
                                if error is None:
                                    results.results[sig_name] = result
                                    results.timings[sig_name] = elapsed
                                else:
                                    results.errors[sig_name] = error
                                completed += 1
                                if self.verbose and not TQDM_AVAILABLE:
                                    status = "OK" if error is None else "ERROR"
                                    print(f"  [{offset + completed}/{total}] {sig_name}: {status}")
                        else:
                            sig_name, result, elapsed, error = future.result()
                            if error is None:
                                results.results[sig_name] = result
                                results.timings[sig_name] = elapsed
                            else:
                                results.errors[sig_name] = error
                            completed += 1
                            if self.verbose and not TQDM_AVAILABLE:
                                status = "OK" if error is None else "ERROR"
                                print(f"  [{offset + completed}/{total}] {sig_name}: {status}")

                    except Exception as e:
                        if is_batch_mode:
                            for sig in item:
                                results.errors[sig] = str(e)
                            completed += len(item)
                        else:
                            results.errors[item] = str(e)
                            completed += 1

                    # Submit new work if available (lazy preparation)
                    if not work_exhausted:
                        try:
                            new_item = next(work_iter)
                            if is_batch_mode:
                                arg = self._prepare_batch_arg(new_item)
                                new_future = executor.submit(_process_withinfirm_batch, arg)
                            else:
                                arg = self._prepare_single_arg(new_item)
                                new_future = executor.submit(_process_withinfirm_signal, arg)
                            pending_futures[new_future] = new_item
                            del arg  # Allow GC
                        except StopIteration:
                            work_exhausted = True


class _BatchResult:
    """Wrapper to provide consistent API for batch results."""

    def __init__(self, strategy_result, datelist, signal_name):
        self.ea = strategy_result
        self.ep = strategy_result  # Same for now (no filter support)
        self.datelist = datelist
        self.signal_name = signal_name
        self.results = strategy_result

    def get_long_short(self):
        """Get EW and VW long-short returns."""
        return self.ea.get_long_short()

    def get_turnover(self):
        """Get turnover statistics."""
        return None, None  # Fast path doesn't compute turnover

    def get_characteristics(self):
        """Get characteristics."""
        return None  # Fast path doesn't compute characteristics

    def get_portfolio_returns(self):
        """Get portfolio returns."""
        return self.ea.get_portfolio_returns()
