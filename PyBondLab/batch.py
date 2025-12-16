# -*- coding: utf-8 -*-
"""
Batch Strategy Formation for PyBondLab.

This module provides efficient batch processing for multiple signals,
sharing computation where possible to achieve significant speedups.

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

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from collections import OrderedDict
import time

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
from .config import StrategyFormationConfig, DataConfig, FormationConfig, FilterConfig
from .results import FormationResults


# =============================================================================
# Batch Results Container
# =============================================================================

@dataclass
class BatchResults:
    """
    Container for batch strategy formation results.

    Provides dictionary-like access to individual signal results,
    plus aggregate statistics across all signals.

    Attributes
    ----------
    results : OrderedDict
        Mapping of signal_name -> FormationResults
    signals : list
        List of signal names (in order processed)
    config : dict
        Batch configuration used
    timings : dict
        Timing information for profiling
    errors : dict
        Any errors encountered during processing
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
        """Check if signal results exist."""
        return signal in self.results

    def __len__(self) -> int:
        """Number of successful signal results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over signal names."""
        return iter(self.results)

    def keys(self):
        """Get signal names."""
        return self.results.keys()

    def values(self):
        """Get all FormationResults."""
        return self.results.values()

    def items(self):
        """Get (signal, results) pairs."""
        return self.results.items()

    @property
    def successful_signals(self) -> List[str]:
        """List of signals that completed successfully."""
        return list(self.results.keys())

    @property
    def failed_signals(self) -> List[str]:
        """List of signals that failed."""
        return list(self.errors.keys())

    @property
    def summary_df(self) -> pd.DataFrame:
        """
        Summary DataFrame with key statistics for all signals.

        Returns
        -------
        pd.DataFrame
            Summary with columns: signal, ew_mean, vw_mean, ew_std, vw_std,
            ew_sharpe, vw_sharpe, ew_turnover, vw_turnover
        """
        rows = []

        for signal, result in self.results.items():
            try:
                ea = result.ea

                # Get long-short returns
                ew_ls, vw_ls = result.get_long_short(strategy='ea')

                row = {
                    'signal': signal,
                    'ew_mean': ew_ls.mean() * 12,  # Annualized
                    'vw_mean': vw_ls.mean() * 12,
                    'ew_std': ew_ls.std() * np.sqrt(12),
                    'vw_std': vw_ls.std() * np.sqrt(12),
                    'ew_sharpe': (ew_ls.mean() / ew_ls.std()) * np.sqrt(12) if ew_ls.std() > 0 else np.nan,
                    'vw_sharpe': (vw_ls.mean() / vw_ls.std()) * np.sqrt(12) if vw_ls.std() > 0 else np.nan,
                    'n_periods': len(ew_ls),
                }

                # Add turnover if available
                if ea.turnover is not None and ea.turnover.ew_turnover_df is not None:
                    row['ew_turnover'] = ea.turnover.ew_turnover_df.mean().mean()
                    row['vw_turnover'] = ea.turnover.vw_turnover_df.mean().mean()

                rows.append(row)

            except Exception as e:
                # Skip signals with errors in summary
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index('signal')
        return df

    def get_factor_returns(self, weight_type: str = 'ew') -> pd.DataFrame:
        """
        Get long-short factor returns for all signals as a DataFrame.

        Parameters
        ----------
        weight_type : {'ew', 'vw'}
            Equal-weighted or value-weighted

        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and signals as columns
        """
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
        """Export results as nested dictionary."""
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
    Batch processing for multiple signals with shared computation.

    This class efficiently processes multiple signals (sort variables)
    with the same configuration, sharing computation where possible
    to achieve significant speedups over sequential processing.

    Parameters
    ----------
    data : pd.DataFrame
        Bond panel data with columns: date, ID, ret, VW, RATING_NUM, and signal columns
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
        Rating filter ('IG', 'NIG', or tuple of bounds)
    banding : int, optional
        Banding parameter to reduce turnover
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all cores)
        Note: Currently sequential, parallelization planned for future
    verbose : bool, default=True
        Whether to show progress

    Attributes
    ----------
    data : pd.DataFrame
        Input data
    signals : list
        Signal columns
    config : dict
        Configuration parameters

    Examples
    --------
    >>> batch_sf = BatchStrategyFormation(
    ...     data=data,
    ...     signals=['momentum', 'value', 'size'],
    ...     holding_period=3,
    ...     num_portfolios=5,
    ...     turnover=True,
    ... )
    >>> results = batch_sf.fit()
    >>> print(results.summary_df)
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
        # Validate inputs
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

        # Compute banding threshold if banding is specified
        self.banding_threshold = None
        if banding is not None:
            self.banding_threshold = banding / num_portfolios

        # Store config for reference
        self.config = {
            'holding_period': holding_period,
            'num_portfolios': num_portfolios,
            'turnover': turnover,
            'chars': chars,
            'rating': rating,
            'banding': banding,
            'n_jobs': n_jobs,
        }

        # Shared precomputed data (populated during fit)
        self._shared_precomp = None

    def _validate_inputs(self, data: pd.DataFrame, signals: List[str]):
        """Validate input data and signals."""
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")

        if not signals:
            raise ValueError("Must provide at least one signal")

        # Check required columns
        required = ['date', 'ID', 'ret', 'VW', 'RATING_NUM']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        # Check signal columns exist
        missing_signals = [s for s in signals if s not in data.columns]
        if missing_signals:
            raise ValueError(f"Signal columns not found in data: {missing_signals}")

    def fit(self) -> BatchResults:
        """
        Run batch portfolio formation for all signals.

        Returns
        -------
        BatchResults
            Container with results for all signals
        """
        results = BatchResults(
            signals=self.signals.copy(),
            config=self.config.copy(),
        )

        t_start = time.time()

        # Create progress iterator
        if self.verbose and TQDM_AVAILABLE:
            signal_iter = tqdm(self.signals, desc="Processing signals", unit="signal")
        elif self.verbose:
            signal_iter = self.signals
            print(f"Processing {len(self.signals)} signals...")
        else:
            signal_iter = self.signals

        # Track shared precompute time
        shared_precomp_time = 0
        first_signal = True
        shared_precomp = None  # Will be extracted from first signal

        for i, signal in enumerate(signal_iter):
            t_signal_start = time.time()

            try:
                # Create strategy for this signal
                strategy = SingleSort(
                    holding_period=self.holding_period,
                    sort_var=signal,
                    num_portfolios=self.num_portfolios,
                    verbose=False
                )

                # Create config
                sf_config = StrategyFormationConfig(
                    data=DataConfig(
                        rating=self.rating,
                        chars=self.chars,
                    ),
                    formation=FormationConfig(
                        dynamic_weights=True,
                        compute_turnover=self.turnover,
                        banding_threshold=self.banding_threshold,
                        verbose=False,
                    )
                )

                # Create StrategyFormation
                sf = StrategyFormation(
                    data=self.data,
                    strategy=strategy,
                    config=sf_config
                )

                # OPTIMIZATION: Pass shared precompute data to skip redundant computation
                # After first signal, we reuse It1, It2, It1m, vw_map_t1m
                # NOTE: We do NOT pass It0 or vw_map_t0 - those depend on the signal
                if shared_precomp is not None:
                    sf._cached_precomp = shared_precomp

                # Run formation
                result = sf.fit()

                # OPTIMIZATION: Extract shared precompute from first signal
                # Only extract signal-INDEPENDENT data (It1, It2, It1m, vw_map_t1m)
                if first_signal and hasattr(sf, '_shareable_precomp'):
                    full_precomp = sf._shareable_precomp
                    # Only keep signal-independent data
                    shared_precomp = {
                        'It1': full_precomp.get('It1'),
                        'It2': full_precomp.get('It2'),
                        'It1m': full_precomp.get('It1m'),
                        'vw_map_t1m': full_precomp.get('vw_map_t1m'),
                    }
                    # Record time spent on shared precompute (first signal includes it)
                    shared_precomp_time = time.time() - t_signal_start

                # Store result
                results.results[signal] = result

                t_signal_end = time.time()
                results.timings[signal] = t_signal_end - t_signal_start

                # Update progress bar description if using tqdm
                if self.verbose and TQDM_AVAILABLE:
                    signal_iter.set_postfix({
                        'last': f"{results.timings[signal]:.2f}s",
                        'done': len(results.results),
                    })
                elif self.verbose and not TQDM_AVAILABLE:
                    print(f"  [{i+1}/{len(self.signals)}] {signal}: {results.timings[signal]:.2f}s")

                first_signal = False

            except Exception as e:
                # Record error and continue
                results.errors[signal] = str(e)
                if self.verbose:
                    if TQDM_AVAILABLE:
                        signal_iter.set_postfix({'error': signal})
                    else:
                        print(f"  [{i+1}/{len(self.signals)}] {signal}: ERROR - {e}")

        t_end = time.time()
        results.timings['total'] = t_end - t_start
        results.timings['shared_precomp'] = shared_precomp_time

        if self.verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: BatchResults):
        """Print summary of batch processing."""
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total signals:    {len(self.signals)}")
        print(f"Successful:       {len(results.results)}")
        print(f"Failed:           {len(results.errors)}")
        print(f"Total time:       {results.timings.get('total', 0):.2f}s")

        if results.results:
            avg_time = sum(results.timings.get(s, 0) for s in results.results) / len(results.results)
            print(f"Avg time/signal:  {avg_time:.2f}s")

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
    **kwargs
        Additional arguments passed to BatchStrategyFormation

    Returns
    -------
    BatchResults
        Batch results container

    Examples
    --------
    >>> results = batch_single_sort(
    ...     data,
    ...     signals=['momentum', 'value', 'size'],
    ...     holding_period=3,
    ...     num_portfolios=5
    ... )
    """
    batch_sf = BatchStrategyFormation(
        data=data,
        signals=signals,
        holding_period=holding_period,
        num_portfolios=num_portfolios,
        turnover=turnover,
        **kwargs
    )
    return batch_sf.fit()
