# -*- coding: utf-8 -*-
"""
Batch Strategy Formation for PyBondLab.

This module provides efficient batch processing for multiple signals,
using parallel processing to achieve significant speedups.

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
>>> results['momentum'].factor_df
>>> results['momentum'].ew_turnover_df
>>>
>>> # Summary across all signals
>>> results.summary_df

Authors: PyBondLab Team
Created: 2024
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import OrderedDict
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

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
         chars, rating, banding_threshold, shared_precomp)

    Returns
    -------
    tuple
        (signal_name, result_or_none, elapsed_time, error_or_none)
    """
    (signal, data, holding_period, num_portfolios, turnover,
     chars, rating, banding_threshold, shared_precomp) = args

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
                verbose=False
            )

            # Create config
            sf_config = StrategyFormationConfig(
                data=DataConfig(
                    rating=rating,
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
            sf = StrategyFormation(
                data=data,
                strategy=strategy,
                config=sf_config
            )

            # Pass shared precompute if available
            if shared_precomp is not None:
                sf._cached_precomp = shared_precomp

            result = sf.fit()

            elapsed = time.time() - t_start
            return (signal, result, elapsed, None)

    except Exception as e:
        elapsed = time.time() - t_start
        return (signal, None, elapsed, str(e))


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
        Rating filter
    banding : int, optional
        Banding parameter
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all cores, 1 for sequential.
    verbose : bool, default=True
        Whether to show progress
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
        banding: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        self._validate_inputs(data, signals)

        self.data = data
        self.signals = list(signals)
        self.holding_period = holding_period
        self.num_portfolios = num_portfolios
        self.turnover = turnover
        self.chars = chars
        self.rating = rating
        self.banding = banding
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.banding_threshold = None
        if banding is not None:
            self.banding_threshold = banding / num_portfolios

        self.config = {
            'holding_period': holding_period,
            'num_portfolios': num_portfolios,
            'turnover': turnover,
            'chars': chars,
            'rating': rating,
            'banding': banding,
            'n_jobs': n_jobs,
        }

    def _validate_inputs(self, data: pd.DataFrame, signals: List[str]):
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        if not signals:
            raise ValueError("Must provide at least one signal")
        required = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")
        missing_signals = [s for s in signals if s not in data.columns]
        if missing_signals:
            raise ValueError(f"Signal columns not found in data: {missing_signals}")

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

    def fit(self) -> BatchResults:
        """Run batch portfolio formation for all signals."""
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
                    verbose=False
                )

                sf_config = StrategyFormationConfig(
                    data=DataConfig(rating=self.rating, chars=self.chars),
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

    def _fit_parallel(self, results: BatchResults, n_workers: int) -> BatchResults:
        """Parallel processing of signals using ProcessPoolExecutor."""

        # First, run one signal sequentially to get shared precompute
        first_signal = self.signals[0]
        remaining_signals = self.signals[1:]

        if self.verbose:
            print(f"  Running first signal to extract shared data...")

        t0 = time.time()
        try:
            strategy = SingleSort(
                holding_period=self.holding_period,
                sort_var=first_signal,
                num_portfolios=self.num_portfolios,
                verbose=False
            )
            sf_config = StrategyFormationConfig(
                data=DataConfig(rating=self.rating, chars=self.chars),
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

            # Extract shared precompute
            shared_precomp = None
            if hasattr(sf, '_shareable_precomp'):
                full_precomp = sf._shareable_precomp
                shared_precomp = {
                    'It1': full_precomp.get('It1'),
                    'It2': full_precomp.get('It2'),
                    'It1m': full_precomp.get('It1m'),
                    'vw_map_t1m': full_precomp.get('vw_map_t1m'),
                }

            if self.verbose:
                print(f"  First signal done: {results.timings[first_signal]:.2f}s")

        except Exception as e:
            results.errors[first_signal] = str(e)
            shared_precomp = None
            if self.verbose:
                print(f"  First signal FAILED: {e}")

        if not remaining_signals:
            return results

        # Process remaining signals in parallel
        if self.verbose:
            print(f"  Processing {len(remaining_signals)} remaining signals in parallel...")

        # Prepare arguments for workers
        worker_args = [
            (signal, self.data, self.holding_period, self.num_portfolios,
             self.turnover, self.chars, self.rating, self.banding_threshold,
             shared_precomp)
            for signal in remaining_signals
        ]

        # Use ProcessPoolExecutor for parallel execution
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_signal = {
                executor.submit(_process_single_signal, args): args[0]
                for args in worker_args
            }

            # Collect results as they complete
            if self.verbose and TQDM_AVAILABLE:
                futures_iter = tqdm(
                    as_completed(future_to_signal),
                    total=len(remaining_signals),
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
                        print(f"  [{completed}/{len(remaining_signals)}] {sig_name}: {status}")

                except Exception as e:
                    results.errors[signal] = str(e)
                    completed += 1

        return results

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
