# -*- coding: utf-8 -*-
"""
BatchWithinFirmSortFormation: Multi-signal processing for WithinFirmSort.

Provides fast batch processing of multiple signals using:
- Ultra-fast numba path when turnover=False and chars=None
- Multiprocessing slow path when turnover=True or chars is set

Author: Claude
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor
import gc

from .StrategyClass import WithinFirmSort
from .PyBondLab import StrategyFormation
from .constants import ColumnNames


class BatchWithinFirmSortFormation:
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
        Bond panel data with columns: date, ID, ret, VW, RATING_NUM, and signal columns
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
        Column name mapping: {'expected_name': 'actual_name'}
    n_jobs : int, default=1
        Number of parallel workers (for slow path)
    verbose : bool, default=True
        Show progress output

    Examples
    --------
    >>> batch = BatchWithinFirmSortFormation(
    ...     data=data,
    ...     signals=['signal1', 'signal2', 'signal3'],
    ...     firm_id_col='PERMNO',
    ...     turnover=False,
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
        verbose: bool = True,
    ):
        self.data = data
        self.signals = signals
        self.firm_id_col = firm_id_col
        self.rating_bins = rating_bins if rating_bins is not None else [-np.inf, 7, 10, np.inf]
        self.min_bonds_per_firm = min_bonds_per_firm
        self.turnover = turnover
        self.chars = chars
        self.rating = rating
        self.subset_filter = subset_filter
        self.columns = columns or {}
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Apply column mapping
        self._apply_column_mapping()

        # Validate signals exist
        for sig in signals:
            if sig not in self.data.columns:
                raise ValueError(f"Signal '{sig}' not found in data columns")

    def _apply_column_mapping(self):
        """Apply column name mapping if provided."""
        if not self.columns:
            return

        rename_map = {}
        for expected, actual in self.columns.items():
            if actual in self.data.columns and expected != actual:
                rename_map[actual] = expected

        if rename_map:
            self.data = self.data.rename(columns=rename_map)

    def _can_use_fast_path(self) -> bool:
        """Check if fast batch path can be used."""
        if self.turnover:
            return False
        if self.chars is not None:
            return False
        return True

    def fit(self) -> Dict[str, 'StrategyFormation']:
        """
        Run batch portfolio formation for all signals.

        Returns
        -------
        Dict[str, StrategyFormation]
            Dictionary mapping signal names to StrategyFormation results
        """
        if self._can_use_fast_path():
            return self._fit_fast_batch()
        else:
            return self._fit_slow_batch()

    def _fit_fast_batch(self) -> Dict[str, 'StrategyFormation']:
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
            return {sig: None for sig in self.signals}

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
            return {sig: None for sig in self.signals}

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
        ).astype('Int64').fillna(0).values.astype(np.int64)

        # Convert to numpy arrays (ONCE)
        date_idx = data[ColumnNames.DATE].map(date_to_idx).values.astype(np.int64)
        id_idx = data[ColumnNames.ID].map(id_to_idx).values.astype(np.int64)
        firm_idx = data[self.firm_id_col].map(firm_to_idx).fillna(-1).values.astype(np.int64)
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

        results = {}

        # Process each signal
        for sig_idx, signal_name in enumerate(self.signals):
            if self.verbose:
                print(f"  [{sig_idx+1}/{len(self.signals)}] Processing {signal_name}...")

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

            # Create a mock StrategyFormation-like result
            results[signal_name] = _BatchResult(result, datelist, signal_name)

        if self.verbose:
            print(f"Batch processing complete: {len(results)} signals processed")

        return results

    def _fit_slow_batch(self) -> Dict[str, 'StrategyFormation']:
        """
        Slow batch processing using multiprocessing.

        Used when turnover=True or chars is set.
        Each worker runs a complete StrategyFormation for one signal.
        """
        if self.verbose:
            print(f"SLOW BATCH PATH: Processing {len(self.signals)} signals with {self.n_jobs} workers...")
            if self.turnover:
                print("  (turnover=True requires slow path)")
            if self.chars:
                print(f"  (chars={self.chars} requires slow path)")

        results = {}

        if self.n_jobs == 1:
            # Sequential processing
            for sig_idx, signal_name in enumerate(self.signals):
                if self.verbose:
                    print(f"  [{sig_idx+1}/{len(self.signals)}] Processing {signal_name}...")

                result = self._process_single_signal(signal_name)
                results[signal_name] = result

                gc.collect()
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self._process_single_signal, sig): sig
                    for sig in self.signals
                }

                for future in futures:
                    signal_name = futures[future]
                    try:
                        result = future.result()
                        results[signal_name] = result
                    except Exception as e:
                        if self.verbose:
                            print(f"  Error processing {signal_name}: {e}")
                        results[signal_name] = None

        if self.verbose:
            print(f"Batch processing complete: {len(results)} signals processed")

        return results

    def _process_single_signal(self, signal_name: str):
        """Process a single signal using StrategyFormation."""
        strategy = WithinFirmSort(
            holding_period=1,
            sort_var=signal_name,
            firm_id_col=self.firm_id_col,
            min_bonds_per_firm=self.min_bonds_per_firm,
            rating_bins=self.rating_bins,
            num_portfolios=2,
            verbose=False
        )

        sf = StrategyFormation(
            data=self.data,
            strategy=strategy,
            turnover=self.turnover,
            chars=self.chars,
            rating=self.rating,
            verbose=False
        )

        return sf.fit()


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
