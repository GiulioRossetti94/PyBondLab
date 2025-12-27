# -*- coding: utf-8 -*-
"""
Base class for batch processing in PyBondLab.

This module provides shared functionality for BatchStrategyFormation and
BatchWithinFirmSortFormation, including:
- Column mapping with verbose output
- Platform-aware multiprocessing (fork vs spawn)
- Memory optimization (minimal data transfer)
- Progress bars (tqdm)
- Summary output with timing statistics

Authors: PyBondLab Team
Created: 2024
"""

import gc
import platform
import time
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd


# =============================================================================
# Memory Management Functions
# =============================================================================

def _estimate_memory_components(data: pd.DataFrame, n_workers: int,
                                  required_columns: List[str]) -> Tuple[float, float, float]:
    """
    Estimate memory components for batch processing.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_workers : int
        Number of parallel workers
    required_columns : list
        Required column names for minimal data

    Returns
    -------
    tuple
        (base_data_mb, minimal_data_mb, processing_overhead_per_worker_mb)
    """
    # Base data size
    base_data_mb = data.memory_usage(deep=True).sum() / 1024 / 1024

    # Minimal data size (only required columns)
    available_cols = [c for c in required_columns if c in data.columns]
    if available_cols:
        minimal_data_mb = data[available_cols].memory_usage(deep=True).sum() / 1024 / 1024
    else:
        # Fallback: estimate as fraction of full data
        minimal_data_mb = base_data_mb * 0.15

    # Processing overhead per worker (intermediate DataFrames, results, etc.)
    # Conservative estimate: 1.5x the minimal data size
    processing_overhead_mb = minimal_data_mb * 1.5

    return base_data_mb, minimal_data_mb, processing_overhead_mb


def _estimate_peak_memory_mb(data: pd.DataFrame, n_signals: int, n_workers: int,
                              chunk_size: Optional[int] = None,
                              required_columns: Optional[List[str]] = None) -> Tuple[float, float]:
    """
    Estimate peak memory usage for batch processing.

    Peak memory = base_data + chunk_size × minimal_data + n_workers × overhead

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_signals : int
        Number of signals to process
    n_workers : int
        Number of parallel workers
    chunk_size : int, optional
        If set, limits signals prepared at once
    required_columns : list, optional
        Required columns for minimal data estimation

    Returns
    -------
    tuple
        (minimal_data_mb, peak_total_mb)
    """
    if required_columns is None:
        required_columns = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']

    base_mb, minimal_mb, overhead_mb = _estimate_memory_components(
        data, n_workers, required_columns
    )

    # Effective chunk size (how many worker args prepared at once)
    effective_chunk = chunk_size if chunk_size else n_signals

    # Peak = base + chunk × minimal_data + concurrent_workers × overhead
    concurrent_workers = min(n_workers, effective_chunk)
    peak_mb = base_mb + (effective_chunk * minimal_mb) + (concurrent_workers * overhead_mb)

    return minimal_mb, peak_mb


def _get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024
    except ImportError:
        # If psutil not available, return a conservative estimate
        return 8000  # Assume 8GB available


def _suggest_chunk_size(data: pd.DataFrame, n_signals: int, n_workers: int,
                        target_memory_fraction: float = 0.7,
                        required_columns: Optional[List[str]] = None) -> Optional[int]:
    """
    Suggest a chunk_size to keep memory usage under control.

    Memory model:
        Peak = base_data + chunk_size × minimal_data + n_workers × overhead

    Solving for chunk_size:
        chunk_size = (target - base_data - n_workers × overhead) / minimal_data

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_signals : int
        Number of signals to process
    n_workers : int
        Number of parallel workers
    target_memory_fraction : float
        Target fraction of available memory to use (default 0.7 = 70%)
    required_columns : list, optional
        Required columns for minimal data estimation

    Returns
    -------
    int or None
        Suggested chunk_size, or None if no chunking needed
    """
    if required_columns is None:
        required_columns = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']

    available_mb = _get_available_memory_mb()
    target_mb = available_mb * target_memory_fraction

    base_mb, minimal_mb, overhead_mb = _estimate_memory_components(
        data, n_workers, required_columns
    )

    # Available budget for worker args after accounting for base data and worker overhead
    budget_for_args = target_mb - base_mb - (n_workers * overhead_mb)

    if budget_for_args <= 0:
        # Not enough memory even for n_workers - suggest minimal chunk
        return max(1, n_workers)

    # How many signals can we prepare at once?
    max_chunk = int(budget_for_args / minimal_mb) if minimal_mb > 0 else n_signals

    # Ensure chunk_size is at least n_workers (no point in smaller chunks)
    max_chunk = max(max_chunk, n_workers)

    if max_chunk >= n_signals:
        return None  # No chunking needed

    # Round up to a reasonable multiple for cleaner batching
    # Aim for 2-5 chunks total
    ideal_chunks = 3
    suggested = max(n_workers, (n_signals + ideal_chunks - 1) // ideal_chunks)

    # Don't exceed what memory allows
    return min(suggested, max_chunk)


# =============================================================================
# Platform-specific multiprocessing setup
# =============================================================================

def _get_start_method() -> str:
    """
    Determine the best multiprocessing start method for the current platform.

    - Linux/macOS: Use 'fork' for copy-on-write memory sharing (fastest)
    - Windows: Use 'spawn' (only option, requires pickle)
    """
    if platform.system() == 'Windows':
        return 'spawn'
    else:
        # Linux and macOS support fork
        return 'fork'


# =============================================================================
# Try to import tqdm for progress bars
# =============================================================================

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


# =============================================================================
# Required columns (PyBondLab internal names)
# =============================================================================

REQUIRED_COLUMNS = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']

# Default column name mapping (PyBondLab name -> default user name)
DEFAULT_COLUMNS = {
    'date': 'date',
    'ID': 'ID',
    'ret': 'ret',
    'VW': 'VW',
    'RATING_NUM': 'RATING_NUM',
}


# =============================================================================
# Base Batch Formation Class
# =============================================================================

class BaseBatchFormation(ABC):
    """
    Abstract base class for batch strategy formation.

    Provides shared functionality for:
    - Column mapping with verbose output
    - Platform-aware multiprocessing
    - Memory optimization
    - Progress tracking
    - Summary output

    Subclasses must implement:
    - _can_use_fast_path() - Check if fast numba path can be used
    - _fit_fast_batch() - Fast numba-based processing
    - _fit_slow_batch() - Multiprocessing-based processing
    - _get_minimal_data() - Extract minimal columns for worker
    - _get_required_columns() - Additional required columns beyond base
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: List[str],
        columns: Optional[Dict[str, str]] = None,
        n_jobs: int = 1,
        signals_per_worker: int = 1,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize base batch formation.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        signals : List[str]
            Signal column names
        columns : Dict[str, str], optional
            Column name mapping {pbl_name: user_name}
        n_jobs : int, default=1
            Number of parallel workers
        signals_per_worker : int, default=1
            Signals per worker (reduces overhead)
        chunk_size : int, optional
            Process in chunks to limit memory
        verbose : bool, default=True
            Show progress output
        """
        self.verbose = verbose
        self.signals = list(signals)
        self.n_jobs = n_jobs
        self.signals_per_worker = max(1, signals_per_worker)
        self.chunk_size = chunk_size

        # Build column mapping
        self.columns = DEFAULT_COLUMNS.copy()
        if columns is not None:
            self.columns.update(columns)

        # Prepare data (rename columns)
        self.data_raw = data
        self.data = self._prepare_data(data, signals)

        # Validate inputs
        self._validate_inputs(self.data, signals)

        # Initialize timing and results tracking
        self.timings: Dict[str, float] = {}
        self.errors: Dict[str, str] = {}

    def _prepare_data(self, data: pd.DataFrame, signals: List[str]) -> pd.DataFrame:
        """
        Prepare data by renaming columns to PyBondLab standard names.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data with user's column names
        signals : List[str]
            Signal column names (NOT renamed)

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
            return data

        # Check for conflicts: user columns that would overwrite existing columns
        # Drop existing target columns before renaming to avoid duplicates
        columns_to_drop = []
        for user_name, pbl_name in rename_map.items():
            if pbl_name in data.columns and pbl_name != user_name:
                # The target column already exists and is different from source
                # User's explicit mapping takes precedence, drop existing column
                columns_to_drop.append(pbl_name)

        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)

        # Rename columns
        data_prepared = data.rename(columns=rename_map)

        if self.verbose:
            renamed_str = ', '.join(f'{k}->{v}' for k, v in rename_map.items())
            print(f"Columns renamed: {renamed_str}")

        return data_prepared

    def _validate_inputs(self, data: pd.DataFrame, signals: List[str]):
        """Validate input data and signals."""
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        if not signals:
            raise ValueError("Must provide at least one signal")

        # Check required columns
        required = self._get_required_columns()
        missing = [col for col in required if col not in data.columns]
        if missing:
            user_cols = [self.columns.get(c, c) for c in missing]
            raise ValueError(
                f"Data missing required columns: {missing}. "
                f"Expected columns (based on 'columns' mapping): {user_cols}. "
                f"Use the 'columns' parameter to map your column names."
            )

        # Check signals exist
        missing_signals = [s for s in signals if s not in data.columns]
        if missing_signals:
            raise ValueError(f"Signal columns not found in data: {missing_signals}")

    def _get_required_columns(self) -> List[str]:
        """
        Get list of required columns.

        Override in subclass to add additional required columns.
        """
        return REQUIRED_COLUMNS.copy()

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

    @abstractmethod
    def _can_use_fast_path(self) -> bool:
        """Check if fast batch path can be used."""
        pass

    @abstractmethod
    def _fit_fast_batch(self):
        """Fast numba-based processing."""
        pass

    @abstractmethod
    def _fit_slow_batch(self):
        """Multiprocessing-based processing."""
        pass

    def _get_minimal_data(self, signal: str) -> pd.DataFrame:
        """
        Extract only required columns for a single signal.

        Override in subclass to add strategy-specific columns.

        Parameters
        ----------
        signal : str
            Signal column name

        Returns
        -------
        pd.DataFrame
            Minimal data for worker
        """
        cols = self._get_required_columns()
        if signal not in cols:
            cols.append(signal)
        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def _get_minimal_data_batch(self, signals: List[str]) -> pd.DataFrame:
        """
        Extract only required columns for a batch of signals.

        Parameters
        ----------
        signals : List[str]
            Signal column names

        Returns
        -------
        pd.DataFrame
            Minimal data for worker
        """
        cols = self._get_required_columns()
        for signal in signals:
            if signal not in cols:
                cols.append(signal)
        cols = [c for c in cols if c in self.data.columns]
        return self.data[cols].copy()

    def _print_data_size_stats(self, minimal_data: pd.DataFrame):
        """Print data size reduction statistics."""
        full_size = self.data.memory_usage(deep=True).sum() / 1024 / 1024
        min_size = minimal_data.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - min_size / full_size) * 100
        print(f"  Data size: {full_size:.1f}MB → {min_size:.1f}MB per worker ({reduction:.0f}% reduction)")

    def _print_summary(self, results, n_success: int, n_failed: int, total_time: float):
        """
        Print summary of batch processing.

        Parameters
        ----------
        results : Any
            Results object
        n_success : int
            Number of successful signals
        n_failed : int
            Number of failed signals
        total_time : float
            Total processing time
        """
        n_workers = self._get_n_workers()
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total signals:    {len(self.signals)}")
        print(f"Successful:       {n_success}")
        print(f"Failed:           {n_failed}")
        print(f"Workers used:     {n_workers}")
        print(f"Total time:       {total_time:.2f}s")

        if n_success > 0 and self.timings:
            signal_times = [t for s, t in self.timings.items() if s != 'total']
            if signal_times:
                avg_time = sum(signal_times) / len(signal_times)
                print(f"Avg time/signal:  {avg_time:.2f}s")

                # Compute effective speedup
                sequential_estimate = avg_time * len(self.signals)
                if total_time > 0:
                    speedup = sequential_estimate / total_time
                    print(f"Effective speedup: {speedup:.1f}x")

        if self.errors:
            print(f"\nFailed signals: {list(self.errors.keys())}")

        print(f"{'='*60}")

    def fit(self):
        """
        Run batch portfolio formation for all signals.

        Returns
        -------
        Results object (type depends on subclass)
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

        if n_workers == 1:
            results = self._fit_sequential()
        else:
            results = self._fit_parallel(n_workers)

        total_time = time.time() - t_start
        self.timings['total'] = total_time

        if self.verbose:
            n_success = len([s for s in self.signals if s not in self.errors])
            n_failed = len(self.errors)
            self._print_summary(results, n_success, n_failed, total_time)

        return results

    def _fit_sequential(self):
        """
        Sequential processing of signals.

        Override in subclass for strategy-specific logic.
        """
        return self._fit_slow_batch()

    def _fit_parallel(self, n_workers: int):
        """
        Parallel processing of signals.

        Override in subclass for strategy-specific parallel logic.
        """
        return self._fit_slow_batch()
