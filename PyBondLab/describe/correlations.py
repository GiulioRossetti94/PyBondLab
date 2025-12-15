"""
Cross-sectional correlation analysis.

This module provides the CorrelationStats class for computing
cross-sectional correlations between pairs of variables.

The methodology:
1. For each period t, compute cross-sectional Pearson and Spearman correlations
2. Aggregate correlations over time (time-series average)

Author: Giulio Rossetti
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .base import BaseDescribe
from .results import CorrelationResult
from .utils import compute_pairwise_correlations


class CorrelationStats(BaseDescribe):
    """
    Compute cross-sectional correlations between variable pairs.

    For each time period t, this class computes Pearson and Spearman
    correlations between all pairs of specified variables. Then it
    aggregates these correlations over time.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with observations over time.
    variables : list of str
        Variables to compute correlations for. All pairwise combinations
        will be computed.
    date_col : str
        Column name for date identifier. Default is 'date'.
    id_col : str
        Column name for entity identifier. Default is 'ID'.
    winsorize : bool
        If True, winsorize data for Pearson correlation to reduce
        outlier effects. Spearman is never winsorized. Default is True.
    winsorize_pct : float
        Percentile for winsorization (e.g., 1.0 for 1st/99th percentile).
        Default is 1.0.
    rating : str or tuple or None
        Rating filter: 'IG', 'NIG', 'all', or tuple (min, max).
        Default is None.
    subset_filter : dict or None
        Characteristic filters as {col: (min, max)}.
        Default is None.
    rating_col : str
        Column name for numeric rating. Default is 'RATING_NUM'.

    Examples
    --------
    >>> from PyBondLab.describe import CorrelationStats
    >>>
    >>> # Compute correlations between multiple variables
    >>> corr = CorrelationStats(
    ...     data=bond_data,
    ...     variables=['duration', 'maturity', 'YTM', 'size'],
    ...     winsorize=True,
    ...     winsorize_pct=1.0,
    ... )
    >>> result = corr.compute()
    >>>
    >>> # View time-series average correlation matrices
    >>> print(result.summary())
    >>>
    >>> # Get Pearson correlations over time
    >>> cs_corr = result.get_cs_correlations(method='pearson')
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Sequence[str],
        date_col: str = 'date',
        id_col: str = 'ID',
        winsorize: bool = True,
        winsorize_pct: float = 1.0,
        rating: Literal['IG', 'NIG', 'all'] | tuple[int, int] | None = None,
        subset_filter: dict[str, tuple[float, float]] | None = None,
        rating_col: str = 'RATING_NUM',
    ):
        super().__init__(
            data=data,
            date_col=date_col,
            id_col=id_col,
            issuer_col=None,
        )

        # Validate and store variables
        self.variables = self._validate_variables(variables)

        if len(self.variables) < 2:
            raise ValueError(
                "At least 2 variables are required for correlation analysis. "
                f"Got {len(self.variables)}: {self.variables}"
            )

        # Store correlation parameters
        self.winsorize = winsorize
        self.winsorize_pct = winsorize_pct

        # Store subsetting parameters
        self.rating = rating
        self.subset_filter = subset_filter
        self.rating_col = rating_col

        # Apply subsetting
        self._apply_subsetting()

        # Generate all variable pairs
        self.variable_pairs = list(combinations(self.variables, 2))

    def _apply_subsetting(self) -> None:
        """Apply rating and characteristic subset filters."""
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

            if isinstance(self.rating, str):
                min_r, max_r = get_rating_bounds(self.rating)
            else:
                min_r, max_r = self.rating

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

    def compute(self) -> CorrelationResult:
        """
        Compute cross-sectional correlations for all variable pairs.

        Returns
        -------
        CorrelationResult
            Result object containing:
            - pearson_cs: Pearson correlations over time
            - spearman_cs: Spearman correlations over time
            - n_obs_cs: Number of observations over time
            - pearson_avg: Time-series average Pearson correlation matrix
            - spearman_avg: Time-series average Spearman correlation matrix
        """
        # Initialize storage for cross-sectional correlations
        pearson_data = {self._pair_name(p): [] for p in self.variable_pairs}
        spearman_data = {self._pair_name(p): [] for p in self.variable_pairs}
        n_obs_data = {self._pair_name(p): [] for p in self.variable_pairs}
        dates_list = []

        # Group data by date
        grouped = self.data.groupby(self.date_col)

        # Compute correlations for each period
        for date in self.dates:
            try:
                group = grouped.get_group(date)
            except KeyError:
                continue

            dates_list.append(date)

            # Compute correlations for each pair
            for var_x, var_y in self.variable_pairs:
                pair_name = self._pair_name((var_x, var_y))

                corr_stats = compute_pairwise_correlations(
                    data=group,
                    var_x=var_x,
                    var_y=var_y,
                    winsorize=self.winsorize,
                    winsorize_pct=self.winsorize_pct,
                )

                pearson_data[pair_name].append(corr_stats['pearson'])
                spearman_data[pair_name].append(corr_stats['spearman'])
                n_obs_data[pair_name].append(corr_stats['n'])

        # Convert to DataFrames
        pearson_cs = pd.DataFrame(pearson_data, index=dates_list)
        spearman_cs = pd.DataFrame(spearman_data, index=dates_list)
        n_obs_cs = pd.DataFrame(n_obs_data, index=dates_list)

        # Compute time-series averages as correlation matrices
        pearson_avg = self._build_correlation_matrix(pearson_cs)
        spearman_avg = self._build_correlation_matrix(spearman_cs)

        return CorrelationResult(
            pearson_cs=pearson_cs,
            spearman_cs=spearman_cs,
            n_obs_cs=n_obs_cs,
            pearson_avg=pearson_avg,
            spearman_avg=spearman_avg,
            variables=self.variables,
            n_periods=len(dates_list),
            date_range=(self.dates.min(), self.dates.max()),
            winsorize=self.winsorize,
            winsorize_pct=self.winsorize_pct if self.winsorize else None,
            subset_info=getattr(self, 'subset_info', None),
        )

    def _pair_name(self, pair: tuple[str, str]) -> str:
        """Generate column name for a variable pair."""
        return f"{pair[0]}_{pair[1]}"

    def _build_correlation_matrix(
        self,
        cs_corr: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build correlation matrix from time-series average of pairwise correlations.

        Parameters
        ----------
        cs_corr : pd.DataFrame
            Cross-sectional correlations over time (columns are pair names).

        Returns
        -------
        pd.DataFrame
            Symmetric correlation matrix with variables as both index and columns.
        """
        n_vars = len(self.variables)
        matrix = np.ones((n_vars, n_vars))

        # Fill in the matrix
        for i, var_i in enumerate(self.variables):
            for j, var_j in enumerate(self.variables):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    pair_name = self._pair_name((var_i, var_j))
                    avg_corr = cs_corr[pair_name].mean()
                    matrix[i, j] = avg_corr
                    matrix[j, i] = avg_corr  # Symmetric

        return pd.DataFrame(
            matrix,
            index=self.variables,
            columns=self.variables,
        )

    def __repr__(self) -> str:
        return (
            f"CorrelationStats("
            f"variables={self.variables}, "
            f"n_pairs={len(self.variable_pairs)}, "
            f"n_periods={self.n_periods}, "
            f"winsorize={self.winsorize}"
            f")"
        )
