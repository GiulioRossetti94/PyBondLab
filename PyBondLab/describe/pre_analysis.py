"""
Pre-analysis summary statistics.

This module provides the PreAnalysisStats class for computing
cross-sectional distribution statistics before portfolio formation.

Supports optional filtering to analyze how return filters (trim, winsorize,
price, bounce) affect the distribution of bond characteristics.

Author: Giulio Rossetti
"""

from __future__ import annotations

from typing import Sequence, Literal

import numpy as np
import pandas as pd

from .base import BaseDescribe
from .results import PreAnalysisResult
from .utils import (
    compute_cross_sectional_stats,
    compute_issuer_stats,
    aggregate_time_series_stats,
)


class PreAnalysisStats(BaseDescribe):
    """
    Compute pre-analysis summary statistics for panel data.

    For each time period t, this class computes cross-sectional statistics
    (mean, std, skewness, kurtosis, percentiles, etc.) of specified variables.
    Then it aggregates these cross-sectional statistics over time.

    Supports optional filtering to analyze how return-based filters affect
    the distribution of bond characteristics. When a filter is applied,
    bonds with filtered-out returns are excluded from the analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with observations over time.
    variables : str or list of str
        Variable(s) to compute statistics for (e.g., 'duration', 'maturity').
    date_col : str
        Column name for date identifier. Default is 'date'.
    id_col : str
        Column name for entity identifier. Default is 'ID'.
    issuer_col : str or None
        Column name for issuer identifier (e.g., 'PERMNO').
        If provided, issuer-level counts will be computed.
        Default is None.
    percentiles : list of float
        Percentiles to compute. Values should be between 0 and 100.
        Default is [5, 25, 50, 75, 95].
    filter_type : str or None
        Type of filter to apply: 'trim', 'wins', 'price', or 'bounce'.
        If None, no filtering is applied. Default is None.
    filter_value : float or list of float or None
        Filter threshold(s). Interpretation depends on filter_type:
        - trim: return threshold (e.g., 0.5 for 50% return cutoff)
        - wins: percentile (e.g., 1.0 for 1st/99th percentile)
        - price: price threshold
        - bounce: bounce threshold
        Can be a single value or [lower, upper] bounds.
    filter_location : str or None
        For winsorizing: 'both', 'left', or 'right'. Default is 'both'.
    ret_col : str
        Column name for returns. Default is 'ret'.
    rating : str or tuple or None
        Rating filter to apply:
        - 'IG': Investment grade bonds (RATING_NUM 1-10)
        - 'NIG': Non-investment grade bonds (RATING_NUM 11-22)
        - 'all' or None: No rating filter
        - tuple (min, max): Custom rating bounds
        Default is None.
    subset_filter : dict or None
        Dictionary of characteristic filters. Keys are column names,
        values are (min, max) tuples. Example:
        {'duration': (1, 10), 'maturity': (0, 30)}
        Default is None.
    rating_col : str
        Column name for numeric rating. Default is 'RATING_NUM'.

    Examples
    --------
    >>> from PyBondLab.describe import PreAnalysisStats
    >>>
    >>> # Without filtering
    >>> stats = PreAnalysisStats(
    ...     data=bond_data,
    ...     variables='duration',
    ...     issuer_col='PERMNO'
    ... )
    >>> result = stats.compute()
    >>>
    >>> # With trimming filter - exclude bonds with |ret| > 50%
    >>> stats_trim = PreAnalysisStats(
    ...     data=bond_data,
    ...     variables='duration',
    ...     filter_type='trim',
    ...     filter_value=[-0.5, 0.5]
    ... )
    >>> result_trim = stats_trim.compute()
    >>>
    >>> # With winsorizing filter - winsorize at 1st/99th percentile
    >>> stats_wins = PreAnalysisStats(
    ...     data=bond_data,
    ...     variables='duration',
    ...     filter_type='wins',
    ...     filter_value=1.0,
    ...     filter_location='both'
    ... )
    >>> result_wins = stats_wins.compute()
    >>>
    >>> # With rating filter (Investment Grade only)
    >>> stats_ig = PreAnalysisStats(
    ...     data=bond_data,
    ...     variables='duration',
    ...     rating='IG'
    ... )
    >>>
    >>> # With characteristic subset filter
    >>> stats_subset = PreAnalysisStats(
    ...     data=bond_data,
    ...     variables='duration',
    ...     subset_filter={'maturity': (1, 30), 'duration': (0, 15)}
    ... )
    """

    DEFAULT_PERCENTILES = [5, 25, 50, 75, 95]
    VALID_FILTER_TYPES = ('trim', 'wins', 'price', 'bounce')
    VALID_RATINGS = ('IG', 'NIG', 'all', None)

    def __init__(
        self,
        data: pd.DataFrame,
        variables: str | Sequence[str],
        date_col: str = 'date',
        id_col: str = 'ID',
        issuer_col: str | None = None,
        percentiles: Sequence[float] | None = None,
        filter_type: Literal['trim', 'wins', 'price', 'bounce'] | None = None,
        filter_value: float | Sequence[float] | None = None,
        filter_location: Literal['both', 'left', 'right'] = 'both',
        ret_col: str = 'ret',
        rating: Literal['IG', 'NIG', 'all'] | tuple[int, int] | None = None,
        subset_filter: dict[str, tuple[float, float]] | None = None,
        rating_col: str = 'RATING_NUM',
    ):
        super().__init__(
            data=data,
            date_col=date_col,
            id_col=id_col,
            issuer_col=issuer_col,
        )

        # Validate and store variables
        self.variables = self._validate_variables(variables)

        # Set percentiles
        if percentiles is None:
            self.percentiles = self.DEFAULT_PERCENTILES
        else:
            self.percentiles = list(percentiles)
            self._validate_percentiles()

        # Store filter parameters
        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_location = filter_location
        self.ret_col = ret_col

        # Store subsetting parameters
        self.rating = rating
        self.subset_filter = subset_filter
        self.rating_col = rating_col

        # Apply subsetting first (rating and characteristic filters)
        self._apply_subsetting()

        # Validate and apply return filters
        if filter_type is not None:
            self._validate_filter_params()
            # Apply filter to create filtered data
            self._apply_filter()

    def _apply_subsetting(self) -> None:
        """
        Apply rating and characteristic subset filters to the data.

        This method filters the data based on:
        1. Rating filter (IG, NIG, or custom bounds)
        2. Characteristic filters (e.g., duration, maturity bounds)

        The filtered data replaces self.data for subsequent analysis.
        """
        from PyBondLab.constants import get_rating_bounds

        n_original = len(self.data)
        subset_info = {}

        # Apply rating filter
        if self.rating is not None and self.rating != 'all':
            if self.rating_col not in self.data.columns:
                raise ValueError(
                    f"Rating column '{self.rating_col}' not found in data. "
                    f"Available columns: {list(self.data.columns)}"
                )

            # Get rating bounds
            if isinstance(self.rating, str):
                min_r, max_r = get_rating_bounds(self.rating)
            else:
                # Custom tuple bounds
                min_r, max_r = self.rating

            # Apply filter
            mask = (
                (self.data[self.rating_col] >= min_r) &
                (self.data[self.rating_col] <= max_r)
            )
            self.data = self.data[mask].copy()
            subset_info['rating'] = {
                'type': self.rating if isinstance(self.rating, str) else 'custom',
                'bounds': (min_r, max_r),
            }

        # Apply characteristic subset filters
        if self.subset_filter is not None:
            for col, (min_val, max_val) in self.subset_filter.items():
                if col not in self.data.columns:
                    raise ValueError(
                        f"Subset filter column '{col}' not found in data. "
                        f"Available columns: {list(self.data.columns)}"
                    )

                mask = (
                    (self.data[col] >= min_val) &
                    (self.data[col] <= max_val)
                )
                self.data = self.data[mask].copy()

            subset_info['characteristics'] = self.subset_filter

        # Store subsetting info
        n_after = len(self.data)
        if n_original != n_after:
            self.subset_info = {
                'n_original': n_original,
                'n_after_subset': n_after,
                'n_excluded': n_original - n_after,
                **subset_info,
            }
        else:
            self.subset_info = None

        # Update dates after subsetting
        self._prepare_dates()

    def _validate_filter_params(self) -> None:
        """Validate filter parameters."""
        if self.filter_type not in self.VALID_FILTER_TYPES:
            raise ValueError(
                f"Invalid filter_type '{self.filter_type}'. "
                f"Must be one of: {self.VALID_FILTER_TYPES}"
            )

        if self.filter_value is None:
            raise ValueError(
                f"filter_value must be provided when filter_type='{self.filter_type}'"
            )

        # Validate return column exists
        if self.ret_col not in self.data.columns:
            raise ValueError(
                f"Return column '{self.ret_col}' not found in data. "
                f"Available columns: {list(self.data.columns)}"
            )

        # For price filter, check PRICE column exists
        if self.filter_type == 'price' and 'PRICE' not in self.data.columns:
            raise ValueError(
                "Price filter requires 'PRICE' column in data."
            )

    def _apply_filter(self) -> None:
        """
        Apply the specified filter to create a mask for valid observations.

        This method creates a boolean mask indicating which observations
        survive the filter. For trim/price/bounce, this excludes extreme
        values. For winsorize, all observations are kept but we still
        identify which ones would have been excluded.

        The filtered data is stored in self.filtered_data.
        """
        from PyBondLab.FilterClass import Filter

        # Create a copy of data for filtering
        data_copy = self.data.copy()

        # Apply filter using the Filter class
        filter_obj = Filter(
            data=data_copy,
            adj=self.filter_type,
            w=self.filter_value,
            loc=self.filter_location,
            price_threshold=100.0  # Default price threshold
        )

        # Apply filters - this adds ret_{adj} column
        filtered_data = filter_obj.apply_filters()

        # Determine which column to check for valid observations
        adj_col = f'{self.ret_col}_{self.filter_type}'

        if self.filter_type in ('trim', 'price', 'bounce'):
            # For these filters, NaN in adjusted column means excluded
            # Keep only rows where adjusted return is not NaN
            self.filtered_data = filtered_data[filtered_data[adj_col].notna()].copy()
            self.filter_mask_col = adj_col
        elif self.filter_type == 'wins':
            # For winsorizing, all observations are kept
            # but we track how many were modified
            self.filtered_data = filtered_data.copy()
            self.filter_mask_col = adj_col
            # Track which values were winsorized
            self.winsorized_mask = (
                filtered_data[adj_col] != filtered_data[self.ret_col]
            )

        # Store filter info
        self.filter_applied = True
        self.n_excluded = len(self.data) - len(self.filtered_data)

    def _validate_percentiles(self) -> None:
        """Validate that percentiles are in valid range [0, 100]."""
        for p in self.percentiles:
            if not 0 <= p <= 100:
                raise ValueError(
                    f"Percentiles must be between 0 and 100, got {p}"
                )

    def compute(
        self,
        include_nw: bool = False,
        nw_lag: int = 0,
    ) -> PreAnalysisResult:
        """
        Compute pre-analysis summary statistics.

        Parameters
        ----------
        include_nw : bool
            If True, compute Newey-West t-statistics for time-series
            aggregates. Default is False.
        nw_lag : int
            Number of lags for Newey-West standard errors.
            Only used if include_nw=True. Default is 0.

        Returns
        -------
        PreAnalysisResult
            Result object containing:
            - cs_stats: Cross-sectional statistics for each period
            - ts_stats: Time-series aggregates of CS statistics
            - ts_stats_nw: NW t-stats (if include_nw=True)
        """
        cs_stats = {}
        ts_stats = {}
        ts_stats_nw = {} if include_nw else None

        for var in self.variables:
            # Compute cross-sectional stats for each period
            cs_df = self._compute_all_periods(var)
            cs_stats[var] = cs_df

            # Aggregate over time (simple statistics)
            ts_df = aggregate_time_series_stats(
                cs_df,
                include_nw=False,
                nw_lag=nw_lag,
            )
            ts_stats[var] = ts_df

            # Newey-West t-statistics if requested
            if include_nw:
                ts_nw_df = aggregate_time_series_stats(
                    cs_df,
                    include_nw=True,
                    nw_lag=nw_lag,
                )
                ts_stats_nw[var] = ts_nw_df

        # Build filter info dict if filter was applied
        filter_info = None
        if self.filter_type is not None:
            filter_info = {
                'type': self.filter_type,
                'value': self.filter_value,
                'location': self.filter_location,
                'n_excluded': getattr(self, 'n_excluded', 0),
                'n_original': len(self.data),
                'n_filtered': len(getattr(self, 'filtered_data', self.data)),
            }

        # Get subset info if subsetting was applied
        subset_info = getattr(self, 'subset_info', None)

        return PreAnalysisResult(
            cs_stats=cs_stats,
            ts_stats=ts_stats,
            ts_stats_nw=ts_stats_nw,
            variables=self.variables,
            n_periods=self.n_periods,
            date_range=(self.dates.min(), self.dates.max()),
            percentiles=self.percentiles,
            issuer_col=self.issuer_col,
            filter_info=filter_info,
            subset_info=subset_info,
        )

    def _compute_all_periods(self, variable: str) -> pd.DataFrame:
        """
        Compute cross-sectional statistics for all periods.

        Parameters
        ----------
        variable : str
            Variable to compute statistics for.

        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and CS statistics as columns.
        """
        results = []

        # Use filtered data if filter was applied, otherwise use original
        data_to_use = getattr(self, 'filtered_data', self.data)

        # Group data by date
        grouped = data_to_use.groupby(self.date_col)

        for date in self.dates:
            try:
                group = grouped.get_group(date)
            except KeyError:
                # No data for this date, skip
                continue

            # Compute statistics for this period
            stats = self._compute_period_stats(group, variable)
            stats['date'] = date
            results.append(stats)

        # Convert to DataFrame
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.set_index('date')

        # Order columns logically
        df = self._order_columns(df)

        return df

    def _compute_period_stats(
        self,
        group: pd.DataFrame,
        variable: str,
    ) -> dict:
        """
        Compute all statistics for a single period.

        Parameters
        ----------
        group : pd.DataFrame
            Data for a single time period.
        variable : str
            Variable to compute statistics for.

        Returns
        -------
        dict
            Dictionary of all statistics for this period.
        """
        values = group[variable]

        # Core cross-sectional statistics
        stats = compute_cross_sectional_stats(
            values=values,
            percentiles=self.percentiles,
        )

        # Total observations in this period (including those with missing var)
        total_obs = len(group)
        stats['pct'] = (stats['n'] / total_obs * 100) if total_obs > 0 else 0.0

        # Issuer statistics if issuer column is provided
        if self.issuer_col is not None:
            # Only count issuers with valid variable values
            valid_mask = group[variable].notna()
            valid_group = group[valid_mask]

            if len(valid_group) > 0:
                issuer_stats = compute_issuer_stats(
                    data=valid_group,
                    id_col=self.id_col,
                    issuer_col=self.issuer_col,
                )
                stats.update(issuer_stats)
            else:
                stats['n_issuers'] = 0
                stats['bonds_per_issuer'] = float('nan')

        return stats

    def _order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order DataFrame columns in a logical sequence.

        Order: mean, std, skew, kurt, min, percentiles..., max, n, pct, issuer stats
        """
        # Define column order
        first_cols = ['mean', 'std', 'skew', 'kurt', 'min']
        percentile_cols = [f'p{int(p)}' for p in sorted(self.percentiles)]
        last_cols = ['max', 'n', 'pct']

        if self.issuer_col is not None:
            last_cols.extend(['n_issuers', 'bonds_per_issuer'])

        # Build ordered column list (only include existing columns)
        ordered = []
        for col in first_cols + percentile_cols + last_cols:
            if col in df.columns:
                ordered.append(col)

        # Add any remaining columns not in our order
        remaining = [c for c in df.columns if c not in ordered]
        ordered.extend(remaining)

        return df[ordered]

    def __repr__(self) -> str:
        return (
            f"PreAnalysisStats("
            f"variables={self.variables}, "
            f"n_periods={self.n_periods}, "
            f"date_range=[{self.dates.min():%Y-%m-%d}, {self.dates.max():%Y-%m-%d}]"
            f")"
        )
