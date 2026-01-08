# -*- coding: utf-8 -*-
"""
Batch Anomaly Assay for PyBondLab.

This module provides efficient batch processing for anomaly assaying
across multiple signals, using parallel processing to achieve significant speedups.

Example Usage
-------------
>>> from PyBondLab import BatchAssayAnomaly
>>>
>>> batch = BatchAssayAnomaly(
...     data=data,
...     signals=['cs', 'ytm', 'sze', 'ami'],
...     specs=TIER2_SPECS,
...     n_jobs=-1,  # Use all CPU cores
... )
>>> results = batch.fit()
>>>
>>> # Access individual signal results
>>> results['cs'].summary()
>>> results['cs'].returns_df
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
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle

# =============================================================================
# Pickle compatibility check
# =============================================================================

def _can_pickle_specs(specs: dict) -> Tuple[bool, str]:
    """
    Check if specs dictionary can be pickled (required for multiprocessing).

    Lambda functions in bp_universes cannot be pickled with standard pickle.

    Returns
    -------
    tuple
        (can_pickle, reason_if_not)
    """
    try:
        pickle.dumps(specs)
        return True, ""
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        # Check specifically for lambda functions
        if 'bp_universes' in specs:
            for name, func in specs['bp_universes'].items():
                if func is not None and callable(func):
                    if hasattr(func, '__name__') and func.__name__ == '<lambda>':
                        return False, f"bp_universes['{name}'] contains a lambda function"
        return False, str(e)


# =============================================================================
# Shared utilities from batch_base
# =============================================================================

from .batch_base import (
    _get_start_method,
    _is_windows,
    TQDM_AVAILABLE,
    tqdm,
    _estimate_memory_components,
    _estimate_peak_memory_mb,
    _get_available_memory_mb,
    _suggest_parallel_config,
    _print_memory_config,
)

# Import anomaly assay components
from .anomaly_assay_fast import assay_anomaly_fast, AnomalyAssayResult


# =============================================================================
# Worker function for parallel processing (must be at module level for pickle)
# =============================================================================

def _process_single_signal_assay(args: Tuple) -> Tuple[str, Any, float, Optional[str]]:
    """
    Process a single signal for anomaly assay - worker function for parallel execution.

    Parameters
    ----------
    args : tuple
        (signal, data, specs, holding_period, dynamic_weights, skip_invalid,
         IDvar, DATEvar, RETvar, VWvar, RATINGvar)

    Returns
    -------
    tuple
        (signal_name, result_or_none, elapsed_time, error_or_none)
    """
    (signal, data, specs, holding_period, dynamic_weights, skip_invalid,
     IDvar, DATEvar, RETvar, VWvar, RATINGvar) = args

    t_start = time.time()

    try:
        # Suppress warnings in worker
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = assay_anomaly_fast(
                data=data,
                signal=signal,
                specs=specs,
                holding_period=holding_period,
                dynamic_weights=dynamic_weights,
                skip_invalid=skip_invalid,
                verbose=False,
                IDvar=IDvar,
                DATEvar=DATEvar,
                RETvar=RETvar,
                VWvar=VWvar,
                RATINGvar=RATINGvar,
            )

            elapsed = time.time() - t_start
            return (signal, result, elapsed, None)

    except Exception as e:
        elapsed = time.time() - t_start
        return (signal, None, elapsed, str(e))


def _process_signal_batch_assay(args: Tuple) -> List[Tuple[str, Any, float, Optional[str]]]:
    """
    Process multiple signals in a single worker - reduces overhead.

    Parameters
    ----------
    args : tuple
        (signals_list, data, specs, holding_period, dynamic_weights, skip_invalid,
         IDvar, DATEvar, RETvar, VWvar, RATINGvar)

    Returns
    -------
    list of tuple
        [(signal_name, result_or_none, elapsed_time, error_or_none), ...]
    """
    (signals_list, data, specs, holding_period, dynamic_weights, skip_invalid,
     IDvar, DATEvar, RETvar, VWvar, RATINGvar) = args

    results = []
    for signal in signals_list:
        t_start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = assay_anomaly_fast(
                    data=data,
                    signal=signal,
                    specs=specs,
                    holding_period=holding_period,
                    dynamic_weights=dynamic_weights,
                    skip_invalid=skip_invalid,
                    verbose=False,
                    IDvar=IDvar,
                    DATEvar=DATEvar,
                    RETvar=RETvar,
                    VWvar=VWvar,
                    RATINGvar=RATINGvar,
                )

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
class BatchAssayResults:
    """
    Container for batch anomaly assay results.

    Provides dictionary-like access to individual signal results,
    plus aggregate statistics across all signals.
    """

    results: OrderedDict = field(default_factory=OrderedDict)
    signals: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, signal: str) -> AnomalyAssayResult:
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
                summary = result.summary(lag=0)
                n_specs = len(summary)
                n_sig_5 = (summary['p_value'] < 0.05).sum()
                n_sig_1 = (summary['p_value'] < 0.01).sum()
                mean_t = summary['t_stat'].abs().mean()
                max_t = summary['t_stat'].abs().max()
                best_spec = summary.loc[summary['t_stat'].abs().idxmax(), 'spec_id']

                row = {
                    'signal': signal,
                    'n_specs': n_specs,
                    'n_sig_5pct': n_sig_5,
                    'n_sig_1pct': n_sig_1,
                    'pct_sig_5pct': 100 * n_sig_5 / n_specs,
                    'pct_sig_1pct': 100 * n_sig_1 / n_specs,
                    'mean_abs_t': mean_t,
                    'max_abs_t': max_t,
                    'best_spec': best_spec,
                }
                rows.append(row)
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index('signal')
        return df

    def get_all_returns(self, spec_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get returns for all signals as a DataFrame.

        Parameters
        ----------
        spec_id : str, optional
            If specified, get returns for this specific specification.
            If None, gets the best specification for each signal.

        Returns
        -------
        pd.DataFrame
            Returns DataFrame with signals as columns
        """
        returns_dict = {}
        for signal, result in self.results.items():
            try:
                if spec_id is not None:
                    if spec_id in result.returns_df.columns:
                        returns_dict[signal] = result.returns_df[spec_id]
                else:
                    # Get best spec (highest |t-stat|)
                    summary = result.summary(lag=0)
                    best_spec = summary.loc[summary['t_stat'].abs().idxmax(), 'spec_id']
                    returns_dict[signal] = result.returns_df[best_spec]
            except Exception:
                continue

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signals': self.signals,
            'config': self.config,
            'timings': self.timings,
            'errors': self.errors,
            'summary': self.summary_df.to_dict() if len(self.results) > 0 else {},
        }


# =============================================================================
# Batch Anomaly Assay
# =============================================================================

class BatchAssayAnomaly:
    """
    Batch processing for anomaly assaying across multiple signals.

    Uses assay_anomaly_fast (optimized numba implementation) with
    optional parallel processing across signals.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signals : list of str
        Column names to use as sorting signals
    specs : dict
        Specification grid dictionary containing:
        - 'weighting': ['EW', 'VW']
        - 'portfolio_structures': [(n_ports, name, breakpoints), ...]
        - 'rating_filters': {'name': filter, ...}
        - 'bp_universes': {'name': func_or_none, ...}
        - 'maturity_filters': {'name': (min, max), ...}
    holding_period : int, default=1
        Holding period in months
    dynamic_weights : bool, default=True
        Use dynamic (d-1) weights for VW portfolios
    skip_invalid : bool, default=True
        Skip invalid specification combinations (e.g., ig_only bp + hy filter)
    IDvar : str, optional
        Column name for bond identifier (default: 'ID')
    DATEvar : str, optional
        Column name for date (default: 'date')
    RETvar : str, optional
        Column name for returns (default: 'ret')
    VWvar : str, optional
        Column name for value weights (default: 'VW')
    RATINGvar : str, optional
        Column name for rating (default: 'RATING_NUM')
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all cores, 1 for sequential.
    signals_per_worker : int, default=1
        Number of signals to process per worker. Higher values reduce overhead.
    chunk_size : int or 'auto', optional
        Process signals in chunks to limit memory. 'auto' enables auto-tuning.
    verbose : bool, default=True
        Whether to show progress

    Notes
    -----
    **Lambda functions in bp_universes**: If `specs['bp_universes']` contains
    lambda functions, parallel processing is not possible (lambda functions
    cannot be pickled). In this case, the class will automatically fall back
    to sequential processing with a warning message.

    To enable parallel processing, define breakpoint universe functions at
    module level instead of using lambdas:

        # Instead of:
        'bp_universes': {'ig_only': lambda df: df['RATING_NUM'] <= 10}

        # Define at module level:
        def ig_only_filter(df):
            return df['RATING_NUM'] <= 10

        # Then use:
        'bp_universes': {'ig_only': ig_only_filter}

    Examples
    --------
    >>> # Basic usage
    >>> batch = BatchAssayAnomaly(
    ...     data=data,
    ...     signals=['cs', 'ytm', 'sze'],
    ...     specs=specs,
    ...     n_jobs=4
    ... )
    >>> results = batch.fit()
    >>>
    >>> # Access individual results
    >>> results['cs'].summary()
    >>> results['cs'].returns_df
    >>>
    >>> # Get summary across all signals
    >>> results.summary_df
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: List[str],
        specs: Dict[str, Any],
        holding_period: int = 1,
        dynamic_weights: bool = True,
        skip_invalid: bool = True,
        IDvar: str = None,
        DATEvar: str = None,
        RETvar: str = None,
        VWvar: str = None,
        RATINGvar: str = None,
        n_jobs: int = 1,
        signals_per_worker: int = 1,
        chunk_size: Optional[Union[int, str]] = None,
        verbose: bool = True,
    ):
        # Validate inputs
        if not isinstance(signals, (list, tuple)):
            raise TypeError(
                f"signals must be a list of column names, got {type(signals).__name__}"
            )
        if not all(isinstance(s, str) for s in signals):
            raise TypeError("All signals must be strings (column names)")

        # Store parameters
        self.data = data
        self.signals = list(signals)
        self.specs = specs
        self.holding_period = holding_period
        self.dynamic_weights = dynamic_weights
        self.skip_invalid = skip_invalid
        self.verbose = verbose

        # Column names
        self.IDvar = IDvar or 'ID'
        self.DATEvar = DATEvar or 'date'
        self.RETvar = RETvar or 'ret'
        self.VWvar = VWvar or 'VW'
        self.RATINGvar = RATINGvar or 'RATING_NUM'

        # Validate required columns exist
        required = [self.DATEvar, self.IDvar, self.RETvar, self.VWvar, self.RATINGvar]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        # Validate signals exist
        missing_signals = [s for s in signals if s not in data.columns]
        if missing_signals:
            raise ValueError(f"Signal columns not found in data: {missing_signals}")

        # Parallel configuration
        requested_workers = self._get_n_workers_from_param(n_jobs)

        # Get required columns for memory estimation
        required_cols = self._get_required_columns()

        # Auto-tune parallel configuration
        if chunk_size == 'auto' or (n_jobs != 1 and len(signals) > 3):
            parallel_config = _suggest_parallel_config(
                self.data, len(signals), requested_workers,
                required_columns=required_cols, verbose=False
            )

            if chunk_size == 'auto':
                self.n_jobs = parallel_config['n_workers']
                self.chunk_size = parallel_config['chunk_size']
                self.signals_per_worker = parallel_config['signals_per_worker']
                self.max_in_flight = parallel_config['max_in_flight']

                if verbose:
                    _print_memory_config(parallel_config, len(signals), verbose=True)
            else:
                self.n_jobs = parallel_config['n_workers']
                self.chunk_size = chunk_size
                self.signals_per_worker = max(1, signals_per_worker)
                self.max_in_flight = parallel_config['max_in_flight']

                for warning in parallel_config['warnings']:
                    if verbose:
                        print(f"  [!] {warning}")
        else:
            self.n_jobs = n_jobs
            self.chunk_size = chunk_size
            self.signals_per_worker = max(1, signals_per_worker)
            self.max_in_flight = None

        self.config = {
            'holding_period': holding_period,
            'dynamic_weights': dynamic_weights,
            'skip_invalid': skip_invalid,
            'n_jobs': n_jobs,
            'signals_per_worker': signals_per_worker,
            'chunk_size': chunk_size,
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

    def _get_n_workers(self) -> int:
        """Determine number of worker processes."""
        return self._get_n_workers_from_param(self.n_jobs)

    def _get_required_columns(self) -> List[str]:
        """Get list of required columns for minimal data."""
        return [self.DATEvar, self.IDvar, self.RETvar, self.VWvar, self.RATINGvar, 'tmat']

    def _get_minimal_data(self, signal: str) -> pd.DataFrame:
        """Extract only required columns for a single signal."""
        cols = self._get_required_columns()
        if signal not in cols:
            cols.append(signal)
        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def _get_minimal_data_batch(self, signals: List[str]) -> pd.DataFrame:
        """Extract only required columns for a batch of signals."""
        cols = self._get_required_columns()
        for signal in signals:
            if signal not in cols:
                cols.append(signal)
        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def fit(self) -> BatchAssayResults:
        """Run batch anomaly assay for all signals."""
        results = BatchAssayResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        t_start = time.time()
        n_workers = self._get_n_workers()

        # Check if specs can be pickled (required for multiprocessing)
        can_pickle, pickle_reason = _can_pickle_specs(self.specs)

        if n_workers > 1 and not can_pickle:
            if self.verbose:
                print(f"Note: Falling back to sequential processing because {pickle_reason}")
                print("      (Lambda functions cannot be sent to worker processes)")
            n_workers = 1

        if self.verbose:
            print(f"Processing {len(self.signals)} signals with {n_workers} worker(s)...")

        if n_workers == 1:
            results = self._fit_sequential(results)
        else:
            results = self._fit_parallel(results, n_workers)

        results.timings['total'] = time.time() - t_start

        if self.verbose:
            self._print_summary(results)

        return results

    def _fit_sequential(self, results: BatchAssayResults) -> BatchAssayResults:
        """Sequential processing of signals."""
        if self.verbose and TQDM_AVAILABLE:
            signal_iter = tqdm(self.signals, desc="Processing", unit="signal")
        else:
            signal_iter = self.signals

        for i, signal in enumerate(signal_iter):
            t_signal_start = time.time()

            try:
                result = assay_anomaly_fast(
                    data=self.data,
                    signal=signal,
                    specs=self.specs,
                    holding_period=self.holding_period,
                    dynamic_weights=self.dynamic_weights,
                    skip_invalid=self.skip_invalid,
                    verbose=False,
                    IDvar=self.IDvar,
                    DATEvar=self.DATEvar,
                    RETvar=self.RETvar,
                    VWvar=self.VWvar,
                    RATINGvar=self.RATINGvar,
                )

                results.results[signal] = result
                results.timings[signal] = time.time() - t_signal_start

                if self.verbose and not TQDM_AVAILABLE:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: {results.timings[signal]:.2f}s")

            except Exception as e:
                results.errors[signal] = str(e)
                if self.verbose:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: ERROR - {e}")

        return results

    def _fit_parallel(self, results: BatchAssayResults, n_workers: int) -> BatchAssayResults:
        """Parallel processing of signals."""
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
            result = assay_anomaly_fast(
                data=self.data,
                signal=first_signal,
                specs=self.specs,
                holding_period=self.holding_period,
                dynamic_weights=self.dynamic_weights,
                skip_invalid=self.skip_invalid,
                verbose=False,
                IDvar=self.IDvar,
                DATEvar=self.DATEvar,
                RETvar=self.RETvar,
                VWvar=self.VWvar,
                RATINGvar=self.RATINGvar,
            )
            results.results[first_signal] = result
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

            self._process_chunk(
                chunk_signals, results, n_workers, start_method, processed, total_remaining
            )
            processed += len(chunk_signals)

            # Memory cleanup between chunks
            if self.chunk_size and chunk_end < total_remaining:
                gc.collect()

        return results

    def _process_chunk(self, signals: List[str], results: BatchAssayResults,
                       n_workers: int, start_method: str,
                       offset: int, total: int):
        """Process a chunk of signals in parallel."""
        max_in_flight = getattr(self, 'max_in_flight', None) or n_workers

        # Group signals into batches if signals_per_worker > 1
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

        # Show stats on first chunk
        if self.verbose and offset == 0:
            full_size = self.data.memory_usage(deep=True).sum() / 1024 / 1024
            if is_batch_mode:
                sample_data = self._get_minimal_data_batch(work_items[0])
            else:
                sample_data = self._get_minimal_data(work_items[0])
            min_size = sample_data.memory_usage(deep=True).sum() / 1024 / 1024
            reduction = (1 - min_size / full_size) * 100
            print(f"  Data size: {full_size:.1f}MB -> {min_size:.1f}MB per worker ({reduction:.0f}% reduction)")
            if max_in_flight < len(work_items):
                print(f"  Max in-flight: {max_in_flight} (lazy arg preparation)")
            del sample_data

        mp_context = mp.get_context(start_method)
        completed = 0
        pending_futures = {}

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

                if is_batch_mode:
                    arg = self._prepare_batch_arg(item)
                    future = executor.submit(_process_signal_batch_assay, arg)
                else:
                    arg = self._prepare_single_arg(item)
                    future = executor.submit(_process_single_signal_assay, arg)
                pending_futures[future] = item
                del arg

            # Process completions and submit new work
            while pending_futures:
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

                    # Submit new work if available
                    if not work_exhausted:
                        try:
                            new_item = next(work_iter)
                            if is_batch_mode:
                                arg = self._prepare_batch_arg(new_item)
                                new_future = executor.submit(_process_signal_batch_assay, arg)
                            else:
                                arg = self._prepare_single_arg(new_item)
                                new_future = executor.submit(_process_single_signal_assay, arg)
                            pending_futures[new_future] = new_item
                            del arg
                        except StopIteration:
                            work_exhausted = True

    def _prepare_single_arg(self, signal: str) -> Tuple:
        """Prepare worker argument for a single signal."""
        minimal_data = self._get_minimal_data(signal)
        return (
            signal, minimal_data, self.specs, self.holding_period,
            self.dynamic_weights, self.skip_invalid,
            self.IDvar, self.DATEvar, self.RETvar, self.VWvar, self.RATINGvar
        )

    def _prepare_batch_arg(self, signals: List[str]) -> Tuple:
        """Prepare worker argument for a batch of signals."""
        batch_data = self._get_minimal_data_batch(signals)
        return (
            signals, batch_data, self.specs, self.holding_period,
            self.dynamic_weights, self.skip_invalid,
            self.IDvar, self.DATEvar, self.RETvar, self.VWvar, self.RATINGvar
        )

    def _print_summary(self, results: BatchAssayResults):
        """Print summary of batch processing."""
        n_workers = self._get_n_workers()
        print(f"\n{'='*60}")
        print("BATCH ASSAY COMPLETE")
        print(f"{'='*60}")
        print(f"Total signals:    {len(self.signals)}")
        print(f"Successful:       {len(results.results)}")
        print(f"Failed:           {len(results.errors)}")
        print(f"Workers used:     {n_workers}")
        print(f"Total time:       {results.timings.get('total', 0):.2f}s")

        if results.results:
            signal_times = [t for s, t in results.timings.items() if s != 'total']
            if signal_times:
                avg_time = sum(signal_times) / len(signal_times)
                print(f"Avg time/signal:  {avg_time:.2f}s")

                # Speedup calculation
                if len(self.signals) >= 3:
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

def batch_assay_anomaly(
    data: pd.DataFrame,
    signals: List[str],
    specs: Dict[str, Any],
    holding_period: int = 1,
    n_jobs: int = -1,
    **kwargs
) -> BatchAssayResults:
    """
    Convenience function for batch anomaly assaying.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data
    signals : list of str
        Signal column names
    specs : dict
        Specification grid
    holding_period : int
        Holding period
    n_jobs : int
        Number of parallel workers (-1 for all cores)
    **kwargs
        Additional arguments passed to BatchAssayAnomaly

    Returns
    -------
    BatchAssayResults
        Batch results container
    """
    batch = BatchAssayAnomaly(
        data=data,
        signals=signals,
        specs=specs,
        holding_period=holding_period,
        n_jobs=n_jobs,
        **kwargs
    )
    return batch.fit()
