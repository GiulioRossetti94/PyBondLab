"""
Base class for data description statistics.

This module provides the abstract base class that defines common
functionality for PreAnalysisStats and PostAnalysisStats.

Author: Giulio Rossetti
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class BaseDescribe(ABC):
    """
    Abstract base class for computing descriptive statistics.

    This class defines the common interface and shared functionality
    for pre-analysis and post-analysis statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with observations over time.
    date_col : str
        Column name for date/time period identifier. Default is 'date'.
    id_col : str
        Column name for entity identifier (e.g., bond ID). Default is 'ID'.
    issuer_col : str or None
        Column name for issuer identifier (e.g., 'PERMNO').
        If provided, issuer-level statistics will be computed.
        Default is None.

    Attributes
    ----------
    data : pd.DataFrame
        The input data.
    date_col : str
        Date column name.
    id_col : str
        Entity ID column name.
    issuer_col : str or None
        Issuer column name.
    dates : pd.DatetimeIndex
        Sorted unique dates in the data.
    n_periods : int
        Number of unique time periods.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        id_col: str = 'ID',
        issuer_col: str | None = None,
    ):
        self.data = data
        self.date_col = date_col
        self.id_col = id_col
        self.issuer_col = issuer_col

        # Validate and prepare data
        self._validate_data()
        self._prepare_dates()

    def _validate_data(self) -> None:
        """
        Validate that required columns exist in the data.

        Raises
        ------
        ValueError
            If required columns are missing from the data.
        """
        required_cols = [self.date_col, self.id_col]
        missing = [col for col in required_cols if col not in self.data.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(self.data.columns)}"
            )

        if self.issuer_col is not None:
            if self.issuer_col not in self.data.columns:
                raise ValueError(
                    f"Issuer column '{self.issuer_col}' not found in data. "
                    f"Available columns: {list(self.data.columns)}"
                )

    def _prepare_dates(self) -> None:
        """Extract and sort unique dates from the data."""
        dates = pd.to_datetime(self.data[self.date_col])
        self.dates = pd.DatetimeIndex(sorted(dates.unique()))
        self.n_periods = len(self.dates)

    def _validate_variables(
        self,
        variables: str | Sequence[str],
    ) -> list[str]:
        """
        Validate that variables exist in the data and return as list.

        Parameters
        ----------
        variables : str or sequence of str
            Variable name(s) to validate.

        Returns
        -------
        list of str
            Validated variable names as a list.

        Raises
        ------
        ValueError
            If any variable is not found in the data.
        """
        # Convert single variable to list
        if isinstance(variables, str):
            variables = [variables]
        else:
            variables = list(variables)

        # Check all variables exist
        missing = [v for v in variables if v not in self.data.columns]
        if missing:
            raise ValueError(
                f"Variables not found in data: {missing}. "
                f"Available columns: {list(self.data.columns)}"
            )

        return variables

    @abstractmethod
    def compute(self, **kwargs):
        """
        Compute descriptive statistics.

        This method must be implemented by subclasses.

        Returns
        -------
        Result object
            A result container with computed statistics.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_periods={self.n_periods}, "
            f"date_range=[{self.dates.min():%Y-%m-%d}, {self.dates.max():%Y-%m-%d}]"
            f")"
        )
