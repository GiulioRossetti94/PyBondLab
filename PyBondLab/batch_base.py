# -*- coding: utf-8 -*-
"""
Base class for batch processing in PyBondLab.

This module provides shared functionality for BatchStrategyFormation and
BatchWithinFirmSortFormation, including:
- Column mapping with verbose output
- Platform-aware multiprocessing (fork vs spawn)
- Memory optimization (minimal data transfer, lazy arg preparation)
- Auto-tuning of parallel parameters
- Progress bars (tqdm)
- Summary output with timing statistics

Authors: PyBondLab Team
Created: 2024
"""

import gc
import platform
import time
import warnings
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd


# =============================================================================
# Platform Detection
# =============================================================================

def _is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == 'Windows'


def _is_fork_safe() -> bool:
    """Check if fork-based multiprocessing is available (Linux/macOS)."""
    return platform.system() != 'Windows'


# =============================================================================
# Memory Management Functions
# =============================================================================

def _get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024
    except ImportError:
        # If psutil not available, return a conservative estimate
        return 8000  # Assume 8GB available


def _get_total_memory_mb() -> float:
    """Get total system memory in MB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 / 1024
    except ImportError:
        return 16000  # Assume 16GB total


def _estimate_memory_components(data: pd.DataFrame, n_workers: int,
                                  required_columns: List[str]) -> Dict[str, float]:
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
    dict
        Memory components in MB:
        - base_data_mb: Size of full DataFrame in main process
        - minimal_data_mb: Size of minimal data per signal
        - pickle_overhead_mb: Overhead from pickle serialization (2x on Windows)
        - worker_python_mb: Python runtime per worker process
        - processing_overhead_mb: Intermediate DataFrames during processing
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

    # Platform-specific overhead
    if _is_windows():
        # Windows 'spawn' requires full pickle serialization
        # Data is pickled (copy), sent, unpickled (another copy)
        pickle_overhead_factor = 2.0
        worker_python_mb = 150.0  # Each worker loads full Python runtime
    else:
        # Linux/macOS 'fork' uses copy-on-write
        # Much less overhead, but still some for modified pages
        pickle_overhead_factor = 1.2
        worker_python_mb = 50.0  # Shared memory, less overhead

    pickle_overhead_mb = minimal_data_mb * pickle_overhead_factor

    # Processing overhead (intermediate DataFrames, results, etc.)
    processing_overhead_mb = minimal_data_mb * 1.5

    return {
        'base_data_mb': base_data_mb,
        'minimal_data_mb': minimal_data_mb,
        'pickle_overhead_mb': pickle_overhead_mb,
        'worker_python_mb': worker_python_mb,
        'processing_overhead_mb': processing_overhead_mb,
    }


def _estimate_peak_memory_mb(data: pd.DataFrame, n_signals: int, n_workers: int,
                              chunk_size: Optional[int] = None,
                              max_in_flight: Optional[int] = None,
                              required_columns: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Estimate peak memory usage for batch processing.

    Uses a realistic memory model that accounts for:
    - Base data in main process
    - Worker args prepared at once (limited by max_in_flight)
    - Pickle serialization overhead (platform-dependent)
    - Worker Python runtime overhead
    - Processing overhead in workers

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_signals : int
        Number of signals to process
    n_workers : int
        Number of parallel workers
    chunk_size : int, optional
        If set, limits signals per chunk
    max_in_flight : int, optional
        Maximum concurrent tasks (limits prepared args)
    required_columns : list, optional
        Required columns for minimal data estimation

    Returns
    -------
    dict
        Memory estimates in MB:
        - minimal_data_mb: Per-signal minimal data size
        - peak_mb: Estimated peak memory usage
        - available_mb: Available system memory
        - is_safe: Whether peak is under safe threshold
    """
    if required_columns is None:
        required_columns = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']

    components = _estimate_memory_components(data, n_workers, required_columns)

    # Effective limits
    effective_chunk = chunk_size if chunk_size else n_signals
    effective_in_flight = max_in_flight if max_in_flight else min(n_workers, effective_chunk)

    # Peak memory calculation:
    # 1. Base data stays in main process
    # 2. max_in_flight worker args are prepared at once
    # 3. Each arg goes through pickle (overhead)
    # 4. Workers have Python runtime + processing overhead

    base_mb = components['base_data_mb']
    args_mb = effective_in_flight * components['minimal_data_mb']
    pickle_mb = effective_in_flight * components['pickle_overhead_mb']
    workers_mb = n_workers * (components['worker_python_mb'] + components['processing_overhead_mb'])

    peak_mb = base_mb + args_mb + pickle_mb + workers_mb
    available_mb = _get_available_memory_mb()

    # Safe threshold: 70% on Linux/macOS, 50% on Windows (more conservative)
    safe_fraction = 0.50 if _is_windows() else 0.70
    safe_threshold = available_mb * safe_fraction

    return {
        'minimal_data_mb': components['minimal_data_mb'],
        'base_data_mb': base_mb,
        'args_mb': args_mb,
        'pickle_mb': pickle_mb,
        'workers_mb': workers_mb,
        'peak_mb': peak_mb,
        'available_mb': available_mb,
        'safe_threshold_mb': safe_threshold,
        'is_safe': peak_mb <= safe_threshold,
    }


def _suggest_parallel_config(data: pd.DataFrame, n_signals: int,
                              requested_workers: int,
                              required_columns: Optional[List[str]] = None,
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Suggest optimal parallel configuration based on available memory.

    This is the main entry point for auto-tuning. It determines:
    - n_workers: May be reduced if memory is tight
    - chunk_size: How many signals per chunk
    - signals_per_worker: How many signals each worker processes
    - max_in_flight: Maximum concurrent task submissions

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_signals : int
        Number of signals to process
    requested_workers : int
        Requested number of workers (may be reduced)
    required_columns : list, optional
        Required columns for minimal data estimation
    verbose : bool
        Print memory diagnostics

    Returns
    -------
    dict
        Recommended configuration:
        - n_workers: Actual workers to use
        - chunk_size: Signals per chunk (None = no chunking)
        - signals_per_worker: Signals batched per worker
        - max_in_flight: Max concurrent submissions
        - memory_info: Dict with memory details
        - warnings: List of warning messages
    """
    if required_columns is None:
        required_columns = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']

    warnings = []
    available_mb = _get_available_memory_mb()
    total_mb = _get_total_memory_mb()

    # Get memory components
    components = _estimate_memory_components(data, requested_workers, required_columns)
    minimal_mb = components['minimal_data_mb']
    base_mb = components['base_data_mb']

    # Platform-specific safe threshold
    safe_fraction = 0.50 if _is_windows() else 0.70
    target_mb = available_mb * safe_fraction

    # Start with requested config
    n_workers = requested_workers

    # Auto-tune signals_per_worker based on signal count
    if n_signals >= 50:
        signals_per_worker = 3
    elif n_signals >= 20:
        signals_per_worker = 2
    else:
        signals_per_worker = 1

    # Calculate max_in_flight to limit prepared args
    # We want to limit the memory from prepared args to ~30% of target
    args_budget_mb = target_mb * 0.30
    max_in_flight = max(2, int(args_budget_mb / (minimal_mb * signals_per_worker)))
    max_in_flight = min(max_in_flight, n_workers, n_signals)

    # Calculate chunk_size
    # Chunk should be large enough for efficient parallel processing
    # but small enough to allow GC between chunks
    chunk_size = max(max_in_flight * signals_per_worker, n_workers * 2)

    # Estimate peak with these settings
    peak_info = _estimate_peak_memory_mb(
        data, n_signals, n_workers,
        chunk_size=chunk_size,
        max_in_flight=max_in_flight,
        required_columns=required_columns
    )

    # If still too high, reduce workers
    if not peak_info['is_safe'] and n_workers > 1:
        # Try reducing workers
        for try_workers in range(n_workers - 1, 0, -1):
            peak_info = _estimate_peak_memory_mb(
                data, n_signals, try_workers,
                chunk_size=chunk_size,
                max_in_flight=min(max_in_flight, try_workers),
                required_columns=required_columns
            )
            if peak_info['is_safe']:
                warnings.append(
                    f"Reduced workers from {n_workers} to {try_workers} due to memory constraints"
                )
                n_workers = try_workers
                max_in_flight = min(max_in_flight, try_workers)
                break

    # Final check - if still not safe, warn but proceed
    if not peak_info['is_safe']:
        warnings.append(
            f"Warning: Estimated peak memory ({peak_info['peak_mb']:.0f}MB) exceeds "
            f"safe threshold ({peak_info['safe_threshold_mb']:.0f}MB). "
            f"Consider reducing n_jobs or processing fewer signals at once."
        )

    # If no chunking needed (all signals fit safely)
    if chunk_size >= n_signals and peak_info['is_safe']:
        chunk_size = None

    return {
        'n_workers': n_workers,
        'chunk_size': chunk_size,
        'signals_per_worker': signals_per_worker,
        'max_in_flight': max_in_flight,
        'memory_info': {
            'available_mb': available_mb,
            'total_mb': total_mb,
            'base_data_mb': base_mb,
            'minimal_data_mb': minimal_mb,
            'peak_mb': peak_info['peak_mb'],
            'safe_threshold_mb': peak_info['safe_threshold_mb'],
            'is_safe': peak_info['is_safe'],
            'platform': 'Windows (spawn)' if _is_windows() else 'Linux/macOS (fork)',
        },
        'warnings': warnings,
    }


def _print_memory_config(config: Dict[str, Any], n_signals: int, verbose: bool = True):
    """
    Print memory configuration in a clean, human-readable format.

    Parameters
    ----------
    config : dict
        Configuration from _suggest_parallel_config
    n_signals : int
        Number of signals being processed
    verbose : bool
        If False, only print warnings
    """
    mem = config['memory_info']

    if verbose:
        print(f"\nMemory Configuration:")
        print(f"  System: {mem['total_mb']/1024:.1f} GB RAM | "
              f"Available: {mem['available_mb']/1024:.1f} GB | "
              f"Target: {mem['safe_threshold_mb']/1024:.1f} GB ({mem['platform']})")
        print(f"  Data: {mem['base_data_mb']:.0f} MB total | "
              f"{mem['minimal_data_mb']:.0f} MB per signal")
        print(f"\nParallel Settings:")
        print(f"  Workers: {config['n_workers']} | "
              f"Signals/worker: {config['signals_per_worker']} | "
              f"Max in-flight: {config['max_in_flight']}")

        if config['chunk_size']:
            n_chunks = (n_signals + config['chunk_size'] - 1) // config['chunk_size']
            print(f"  Chunk size: {config['chunk_size']} signals ({n_chunks} chunks)")
        else:
            print(f"  Chunk size: None (all signals fit in memory)")

        status = "[OK]" if mem['is_safe'] else "[!] HIGH"
        print(f"  Est. peak memory: {mem['peak_mb']/1024:.1f} GB {status}")

    # Always emit warnings
    for warning in config['warnings']:
        warnings.warn(warning, UserWarning, stacklevel=2)


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
        print(f"  Data size: {full_size:.1f}MB -> {min_size:.1f}MB per worker ({reduction:.0f}% reduction)")

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

                # Compute effective speedup (only show if meaningful)
                if len(self.signals) >= 3:
                    sequential_estimate = avg_time * len(self.signals)
                    if total_time > 0:
                        speedup = sequential_estimate / total_time
                        if speedup >= 1.0:
                            print(f"Effective speedup: {speedup:.1f}x")
                        else:
                            # Parallel overhead exceeded benefit
                            print(f"Note: Parallel overhead exceeded benefit for this dataset ({speedup:.1f}x)")

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
