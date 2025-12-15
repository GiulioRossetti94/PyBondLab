"""
Result containers for descriptive statistics.

This module provides result classes that store computed statistics
and provide methods for display, export, and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd


@dataclass
class PreAnalysisResult:
    """
    Container for pre-analysis summary statistics results.

    This class stores cross-sectional statistics computed for each period
    and their time-series aggregates, providing methods for display,
    LaTeX export, and visualization.

    Attributes
    ----------
    cs_stats : dict[str, pd.DataFrame]
        Cross-sectional statistics for each variable.
        Keys are variable names, values are DataFrames with dates as index.
    ts_stats : dict[str, pd.DataFrame]
        Time-series aggregates of CS statistics for each variable.
        Keys are variable names, values are DataFrames with CS stats as index.
    ts_stats_nw : dict[str, pd.DataFrame] or None
        Time-series aggregates with Newey-West t-statistics, if computed.
    variables : list[str]
        List of variables for which statistics were computed.
    n_periods : int
        Number of time periods in the data.
    date_range : tuple
        (start_date, end_date) tuple.
    percentiles : list[float]
        Percentiles that were computed.
    issuer_col : str or None
        Issuer column name, if issuer stats were computed.
    filter_info : dict or None
        Information about applied filter, if any. Contains:
        - type: filter type ('trim', 'wins', 'price', 'bounce')
        - value: filter threshold(s)
        - location: for winsorize ('both', 'left', 'right')
        - n_excluded: number of observations excluded
        - n_original: original observation count
        - n_filtered: filtered observation count
    subset_info : dict or None
        Information about applied subsetting, if any. Contains:
        - n_original: original observation count
        - n_after_subset: observation count after subsetting
        - n_excluded: number of observations excluded
        - rating: rating filter info (if applied)
        - characteristics: characteristic filters (if applied)

    Examples
    --------
    >>> result = stats.compute()
    >>> print(result.summary())                    # All variables
    >>> print(result.summary('duration'))          # Single variable
    >>> cs_df = result.get_cs_stats('duration')    # Cross-sectional over time
    >>> latex = result.to_latex()                  # Export to LaTeX
    >>> fig = result.plot('duration')              # Visualize
    """

    cs_stats: dict[str, pd.DataFrame]
    ts_stats: dict[str, pd.DataFrame]
    ts_stats_nw: dict[str, pd.DataFrame] | None
    variables: list[str]
    n_periods: int
    date_range: tuple
    percentiles: list[float] = field(default_factory=lambda: [5, 25, 50, 75, 95])
    issuer_col: str | None = None
    filter_info: dict | None = None
    subset_info: dict | None = None

    # =========================================================================
    # Summary and Display Methods
    # =========================================================================

    def summary(self, variable: str | None = None) -> pd.DataFrame:
        """
        Get summary table of time-series aggregates.

        Parameters
        ----------
        variable : str or None
            Variable name. If None and only one variable was analyzed,
            returns that variable's summary. If multiple variables,
            raises ValueError.

        Returns
        -------
        pd.DataFrame
            Summary table with CS statistics as rows and TS aggregates
            (mean, std, median, min, max) as columns.
        """
        var = self._resolve_variable(variable)

        # Use NW stats if available, otherwise simple stats
        if self.ts_stats_nw is not None and var in self.ts_stats_nw:
            return self.ts_stats_nw[var]
        else:
            return self.ts_stats[var]

    def get_cs_stats(self, variable: str | None = None) -> pd.DataFrame:
        """
        Get cross-sectional statistics over time.

        Parameters
        ----------
        variable : str or None
            Variable name. If None and only one variable was analyzed,
            returns that variable's CS stats.

        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and CS statistics as columns.
            Each row represents statistics for one time period.
        """
        var = self._resolve_variable(variable)
        return self.cs_stats[var]

    def get_ts_stats(self, variable: str | None = None) -> pd.DataFrame:
        """
        Get time-series aggregates (simple statistics only).

        Parameters
        ----------
        variable : str or None
            Variable name.

        Returns
        -------
        pd.DataFrame
            DataFrame with CS statistics as rows and TS aggregates as columns.
        """
        var = self._resolve_variable(variable)
        return self.ts_stats[var]

    def _resolve_variable(self, variable: str | None) -> str:
        """Resolve variable name, handling None for single-variable case."""
        if variable is None:
            if len(self.variables) == 1:
                return self.variables[0]
            else:
                raise ValueError(
                    f"Multiple variables analyzed: {self.variables}. "
                    f"Please specify which variable to use."
                )
        else:
            if variable not in self.variables:
                raise ValueError(
                    f"Variable '{variable}' not found. "
                    f"Available variables: {self.variables}"
                )
            return variable

    # =========================================================================
    # LaTeX Export
    # =========================================================================

    def to_latex(
        self,
        variable: str | None = None,
        caption: str | None = None,
        label: str | None = None,
        precision: int = 3,
    ) -> str:
        """
        Export summary statistics to LaTeX table format.

        Parameters
        ----------
        variable : str or None
            Variable to export. If None and single variable, uses that.
        caption : str or None
            LaTeX table caption. If None, generates a default caption.
        label : str or None
            LaTeX table label for referencing.
        precision : int
            Number of decimal places. Default is 3.

        Returns
        -------
        str
            LaTeX table as a string.
        """
        var = self._resolve_variable(variable)
        df = self.summary(var)

        if caption is None:
            caption = f"Summary Statistics: {var}"

        # Build LaTeX table
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        # Determine columns
        cols = df.columns.tolist()
        col_spec = "l" + "r" * len(cols)

        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\hline")

        # Header row
        header = " & ".join(["Statistic"] + [str(c) for c in cols])
        lines.append(header + " \\\\")
        lines.append("\\hline")

        # Data rows
        for idx in df.index:
            row_values = []
            for col in cols:
                val = df.loc[idx, col]
                if pd.isna(val):
                    row_values.append("--")
                elif abs(val) >= 1e4 or (abs(val) < 1e-4 and val != 0):
                    row_values.append(f"{val:.{precision}g}")
                else:
                    row_values.append(f"{val:.{precision}f}")

            row = " & ".join([str(idx)] + row_values)
            lines.append(row + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")

        # Add notes
        lines.append("\\begin{tablenotes}")
        lines.append("\\small")
        lines.append(
            f"\\item Note: Statistics computed over {self.n_periods} periods "
            f"from {self.date_range[0]:%Y-%m} to {self.date_range[1]:%Y-%m}."
        )
        if self.ts_stats_nw is not None:
            lines.append(
                "\\item t-statistics computed using Newey-West standard errors."
            )
        lines.append("\\end{tablenotes}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot(
        self,
        variable: str | None = None,
        stats: Sequence[str] = ('mean', 'std'),
        figsize: tuple = (12, 6),
        title: str | None = None,
    ):
        """
        Plot cross-sectional statistics over time.

        Parameters
        ----------
        variable : str or None
            Variable to plot.
        stats : sequence of str
            Which CS statistics to plot. Default is ('mean', 'std').
        figsize : tuple
            Figure size (width, height). Default is (12, 6).
        title : str or None
            Plot title. If None, generates a default title.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        var = self._resolve_variable(variable)
        cs_df = self.cs_stats[var]

        # Create figure with subplots for each statistic
        n_stats = len(stats)
        fig, axes = plt.subplots(n_stats, 1, figsize=figsize, sharex=True)

        if n_stats == 1:
            axes = [axes]

        for ax, stat in zip(axes, stats):
            if stat not in cs_df.columns:
                ax.text(
                    0.5, 0.5, f"'{stat}' not available",
                    ha='center', va='center', transform=ax.transAxes
                )
                continue

            series = cs_df[stat]
            ax.plot(series.index, series.values, linewidth=1.5)
            ax.set_ylabel(stat)
            ax.grid(True, alpha=0.3)

            # Add horizontal line at mean
            mean_val = series.mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5)

        # Title and x-label
        if title is None:
            title = f"Cross-Sectional Statistics Over Time: {var}"
        axes[0].set_title(title)
        axes[-1].set_xlabel("Date")

        plt.tight_layout()
        return fig

    # =========================================================================
    # Comparison Methods
    # =========================================================================

    def compare(
        self,
        other: 'PreAnalysisResult',
        variable: str | None = None,
    ) -> pd.DataFrame:
        """
        Compare this result with another (e.g., raw vs filtered).

        Parameters
        ----------
        other : PreAnalysisResult
            Another result to compare against.
        variable : str or None
            Variable to compare. If None and single variable, uses that.

        Returns
        -------
        pd.DataFrame
            Comparison table showing both results side by side with differences.
        """
        var = self._resolve_variable(variable)

        # Get summaries from both results
        df1 = self.get_ts_stats(var)
        df2 = other.get_ts_stats(var)

        # Build comparison DataFrame
        comparison = pd.DataFrame({
            'self_mean': df1['mean'],
            'other_mean': df2['mean'],
            'diff': df1['mean'] - df2['mean'],
            'pct_diff': (df1['mean'] - df2['mean']) / df2['mean'].abs() * 100,
        })

        return comparison

    @staticmethod
    def compare_results(
        raw_result: 'PreAnalysisResult',
        filtered_result: 'PreAnalysisResult',
        variable: str | None = None,
    ) -> pd.DataFrame:
        """
        Compare raw and filtered results side by side.

        This static method makes it easy to compare how filtering
        affects the distribution of characteristics.

        Parameters
        ----------
        raw_result : PreAnalysisResult
            Result from unfiltered data.
        filtered_result : PreAnalysisResult
            Result from filtered data.
        variable : str or None
            Variable to compare.

        Returns
        -------
        pd.DataFrame
            Comparison table with raw, filtered, and difference columns.

        Examples
        --------
        >>> raw = PreAnalysisStats(data, 'duration').compute()
        >>> filtered = PreAnalysisStats(data, 'duration', filter_type='trim', filter_value=[-0.5, 0.5]).compute()
        >>> comparison = PreAnalysisResult.compare_results(raw, filtered)
        >>> print(comparison)
        """
        var = raw_result._resolve_variable(variable)

        raw_ts = raw_result.get_ts_stats(var)
        filt_ts = filtered_result.get_ts_stats(var)

        comparison = pd.DataFrame({
            'raw': raw_ts['mean'],
            'filtered': filt_ts['mean'],
            'diff': raw_ts['mean'] - filt_ts['mean'],
            'pct_diff': (raw_ts['mean'] - filt_ts['mean']) / raw_ts['mean'].abs() * 100,
        })

        # Add filter info as metadata
        comparison.attrs['filter_info'] = filtered_result.filter_info

        return comparison

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_dict(self) -> dict:
        """
        Convert results to a dictionary.

        Returns
        -------
        dict
            Dictionary containing all results.
        """
        return {
            'cs_stats': {k: v.to_dict() for k, v in self.cs_stats.items()},
            'ts_stats': {k: v.to_dict() for k, v in self.ts_stats.items()},
            'ts_stats_nw': (
                {k: v.to_dict() for k, v in self.ts_stats_nw.items()}
                if self.ts_stats_nw is not None else None
            ),
            'variables': self.variables,
            'n_periods': self.n_periods,
            'date_range': (str(self.date_range[0]), str(self.date_range[1])),
            'percentiles': self.percentiles,
            'issuer_col': self.issuer_col,
            'filter_info': self.filter_info,
        }

    def __repr__(self) -> str:
        return (
            f"PreAnalysisResult("
            f"variables={self.variables}, "
            f"n_periods={self.n_periods}, "
            f"date_range=[{self.date_range[0]:%Y-%m-%d}, {self.date_range[1]:%Y-%m-%d}]"
            f")"
        )

    def __str__(self) -> str:
        """Pretty print the summary for all variables."""
        lines = []
        lines.append("=" * 60)
        lines.append("Pre-Analysis Summary Statistics")
        lines.append("=" * 60)
        lines.append(
            f"Date Range: {self.date_range[0]:%Y-%m-%d} to {self.date_range[1]:%Y-%m-%d}"
        )
        lines.append(f"Number of Periods: {self.n_periods}")
        lines.append(f"Variables: {', '.join(self.variables)}")
        lines.append(f"Percentiles: {self.percentiles}")
        if self.issuer_col:
            lines.append(f"Issuer Column: {self.issuer_col}")

        # Display subset information if present
        if self.subset_info is not None:
            lines.append("-" * 60)
            lines.append("Subsetting Applied:")
            if 'rating' in self.subset_info:
                rating_info = self.subset_info['rating']
                lines.append(
                    f"  Rating: {rating_info['type']} "
                    f"(bounds: {rating_info['bounds']})"
                )
            if 'characteristics' in self.subset_info:
                lines.append(f"  Characteristics: {self.subset_info['characteristics']}")
            lines.append(
                f"  Observations: {self.subset_info['n_after_subset']:,} / "
                f"{self.subset_info['n_original']:,} "
                f"({self.subset_info['n_excluded']:,} excluded, "
                f"{100 * self.subset_info['n_excluded'] / self.subset_info['n_original']:.1f}%)"
            )

        # Display filter information if present
        if self.filter_info is not None:
            lines.append("-" * 60)
            lines.append("Return Filter Applied:")
            lines.append(f"  Type: {self.filter_info['type']}")
            lines.append(f"  Value: {self.filter_info['value']}")
            if self.filter_info['type'] == 'wins':
                lines.append(f"  Location: {self.filter_info['location']}")
            lines.append(
                f"  Observations: {self.filter_info['n_filtered']:,} / "
                f"{self.filter_info['n_original']:,} "
                f"({self.filter_info['n_excluded']:,} excluded, "
                f"{100 * self.filter_info['n_excluded'] / self.filter_info['n_original']:.1f}%)"
            )

        lines.append("-" * 60)

        for var in self.variables:
            lines.append(f"\nVariable: {var}")
            lines.append("-" * 40)
            lines.append(self.summary(var).to_string())
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Correlation Result Container
# =============================================================================

@dataclass
class CorrelationResult:
    """
    Container for cross-sectional correlation results.

    This class stores Pearson and Spearman correlations computed for each
    period and their time-series averages as correlation matrices.

    Attributes
    ----------
    pearson_cs : pd.DataFrame
        Pearson correlations over time. Index is date, columns are pair names
        (e.g., 'duration_maturity').
    spearman_cs : pd.DataFrame
        Spearman correlations over time.
    n_obs_cs : pd.DataFrame
        Number of valid observation pairs over time.
    pearson_avg : pd.DataFrame
        Time-series average Pearson correlation matrix.
    spearman_avg : pd.DataFrame
        Time-series average Spearman correlation matrix.
    variables : list[str]
        Variables used in correlation analysis.
    n_periods : int
        Number of time periods.
    date_range : tuple
        (start_date, end_date) tuple.
    winsorize : bool
        Whether winsorization was applied for Pearson.
    winsorize_pct : float or None
        Winsorization percentile, if applied.
    subset_info : dict or None
        Information about applied subsetting.

    Examples
    --------
    >>> result = corr.compute()
    >>> print(result.summary())              # Both Pearson and Spearman
    >>> print(result.summary('pearson'))     # Pearson only
    >>> cs = result.get_cs_correlations()    # Correlations over time
    >>> latex = result.to_latex()            # Export to LaTeX
    """

    pearson_cs: pd.DataFrame
    spearman_cs: pd.DataFrame
    n_obs_cs: pd.DataFrame
    pearson_avg: pd.DataFrame
    spearman_avg: pd.DataFrame
    variables: list[str]
    n_periods: int
    date_range: tuple
    winsorize: bool = True
    winsorize_pct: float | None = 1.0
    subset_info: dict | None = None

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def summary(
        self,
        method: Literal['pearson', 'spearman', 'both'] = 'both',
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Get time-series average correlation matrix.

        Parameters
        ----------
        method : str
            'pearson', 'spearman', or 'both'. Default is 'both'.

        Returns
        -------
        pd.DataFrame or dict
            If method is 'both', returns dict with 'pearson' and 'spearman' keys.
            Otherwise returns single DataFrame.
        """
        if method == 'pearson':
            return self.pearson_avg
        elif method == 'spearman':
            return self.spearman_avg
        elif method == 'both':
            return {
                'pearson': self.pearson_avg,
                'spearman': self.spearman_avg,
            }
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'pearson', 'spearman', or 'both'.")

    def get_cs_correlations(
        self,
        method: Literal['pearson', 'spearman'] = 'pearson',
    ) -> pd.DataFrame:
        """
        Get cross-sectional correlations over time.

        Parameters
        ----------
        method : str
            'pearson' or 'spearman'. Default is 'pearson'.

        Returns
        -------
        pd.DataFrame
            Correlations over time. Index is date, columns are pair names.
        """
        if method == 'pearson':
            return self.pearson_cs
        elif method == 'spearman':
            return self.spearman_cs
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'pearson' or 'spearman'.")

    def get_pair_correlation(
        self,
        var_x: str,
        var_y: str,
        method: Literal['pearson', 'spearman'] = 'pearson',
    ) -> pd.Series:
        """
        Get correlation time series for a specific variable pair.

        Parameters
        ----------
        var_x : str
            First variable.
        var_y : str
            Second variable.
        method : str
            'pearson' or 'spearman'.

        Returns
        -------
        pd.Series
            Correlation over time for this pair.
        """
        # Try both orderings of the pair name
        pair_name = f"{var_x}_{var_y}"
        pair_name_alt = f"{var_y}_{var_x}"

        cs_df = self.get_cs_correlations(method)

        if pair_name in cs_df.columns:
            return cs_df[pair_name]
        elif pair_name_alt in cs_df.columns:
            return cs_df[pair_name_alt]
        else:
            raise ValueError(
                f"Variable pair ({var_x}, {var_y}) not found. "
                f"Available pairs: {list(cs_df.columns)}"
            )

    # =========================================================================
    # LaTeX Export
    # =========================================================================

    def to_latex(
        self,
        method: Literal['pearson', 'spearman', 'both'] = 'both',
        caption: str | None = None,
        label: str | None = None,
        precision: int = 3,
    ) -> str:
        """
        Export correlation matrix to LaTeX table format.

        Parameters
        ----------
        method : str
            'pearson', 'spearman', or 'both'.
        caption : str or None
            LaTeX table caption.
        label : str or None
            LaTeX table label.
        precision : int
            Number of decimal places.

        Returns
        -------
        str
            LaTeX table as a string.
        """
        lines = []

        if method in ('pearson', 'both'):
            lines.append(self._matrix_to_latex(
                self.pearson_avg,
                title="Pearson Correlations" + (
                    f" (Winsorized at {self.winsorize_pct}%/{100-self.winsorize_pct}%)"
                    if self.winsorize else ""
                ),
                caption=caption,
                label=label,
                precision=precision,
            ))
            lines.append("")

        if method in ('spearman', 'both'):
            lines.append(self._matrix_to_latex(
                self.spearman_avg,
                title="Spearman Rank Correlations",
                caption=caption,
                label=label,
                precision=precision,
            ))

        return "\n".join(lines)

    def _matrix_to_latex(
        self,
        matrix: pd.DataFrame,
        title: str,
        caption: str | None,
        label: str | None,
        precision: int,
    ) -> str:
        """Convert a correlation matrix to LaTeX format."""
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        if caption:
            lines.append(f"\\caption{{{caption}: {title}}}")
        else:
            lines.append(f"\\caption{{{title}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        n_cols = len(matrix.columns)
        col_spec = "l" + "r" * n_cols

        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\hline")

        # Header row
        header = " & ".join([""] + list(matrix.columns))
        lines.append(header + " \\\\")
        lines.append("\\hline")

        # Data rows
        for idx in matrix.index:
            row_values = []
            for col in matrix.columns:
                val = matrix.loc[idx, col]
                if pd.isna(val):
                    row_values.append("--")
                else:
                    row_values.append(f"{val:.{precision}f}")
            row = " & ".join([str(idx)] + row_values)
            lines.append(row + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot(
        self,
        var_x: str,
        var_y: str,
        method: Literal['pearson', 'spearman', 'both'] = 'both',
        figsize: tuple = (12, 5),
        title: str | None = None,
    ):
        """
        Plot correlation over time for a specific variable pair.

        Parameters
        ----------
        var_x : str
            First variable.
        var_y : str
            Second variable.
        method : str
            'pearson', 'spearman', or 'both'.
        figsize : tuple
            Figure size.
        title : str or None
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        if method in ('pearson', 'both'):
            pearson_series = self.get_pair_correlation(var_x, var_y, 'pearson')
            ax.plot(
                pearson_series.index,
                pearson_series.values,
                label=f'Pearson (avg={pearson_series.mean():.3f})',
                linewidth=1.5,
            )

        if method in ('spearman', 'both'):
            spearman_series = self.get_pair_correlation(var_x, var_y, 'spearman')
            ax.plot(
                spearman_series.index,
                spearman_series.values,
                label=f'Spearman (avg={spearman_series.mean():.3f})',
                linewidth=1.5,
                linestyle='--',
            )

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if title is None:
            title = f"Cross-Sectional Correlation: {var_x} vs {var_y}"
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_heatmap(
        self,
        method: Literal['pearson', 'spearman'] = 'pearson',
        figsize: tuple = (8, 6),
        cmap: str = 'RdBu_r',
        annot: bool = True,
    ):
        """
        Plot correlation matrix as a heatmap.

        Parameters
        ----------
        method : str
            'pearson' or 'spearman'.
        figsize : tuple
            Figure size.
        cmap : str
            Colormap name.
        annot : bool
            If True, annotate cells with correlation values.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        matrix = self.summary(method)

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(matrix.values, cmap=cmap, vmin=-1, vmax=1)

        # Set ticks
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(matrix.index)

        # Add annotations
        if annot:
            for i in range(len(matrix.index)):
                for j in range(len(matrix.columns)):
                    val = matrix.iloc[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        title = f"{'Pearson' if method == 'pearson' else 'Spearman'} Correlation Matrix"
        ax.set_title(title)

        plt.tight_layout()
        return fig

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_dict(self) -> dict:
        """Convert results to a dictionary."""
        return {
            'pearson_cs': self.pearson_cs.to_dict(),
            'spearman_cs': self.spearman_cs.to_dict(),
            'n_obs_cs': self.n_obs_cs.to_dict(),
            'pearson_avg': self.pearson_avg.to_dict(),
            'spearman_avg': self.spearman_avg.to_dict(),
            'variables': self.variables,
            'n_periods': self.n_periods,
            'date_range': (str(self.date_range[0]), str(self.date_range[1])),
            'winsorize': self.winsorize,
            'winsorize_pct': self.winsorize_pct,
            'subset_info': self.subset_info,
        }

    def __repr__(self) -> str:
        return (
            f"CorrelationResult("
            f"variables={self.variables}, "
            f"n_periods={self.n_periods}, "
            f"winsorize={self.winsorize}"
            f")"
        )

    def __str__(self) -> str:
        """Pretty print the correlation results."""
        lines = []
        lines.append("=" * 60)
        lines.append("Cross-Sectional Correlation Analysis")
        lines.append("=" * 60)
        lines.append(
            f"Date Range: {self.date_range[0]:%Y-%m-%d} to {self.date_range[1]:%Y-%m-%d}"
        )
        lines.append(f"Number of Periods: {self.n_periods}")
        lines.append(f"Variables: {', '.join(self.variables)}")
        if self.winsorize:
            lines.append(
                f"Winsorization: {self.winsorize_pct}% / {100-self.winsorize_pct}% "
                "(for Pearson only)"
            )
        else:
            lines.append("Winsorization: None")

        # Display subset info if present
        if self.subset_info is not None:
            lines.append("-" * 60)
            lines.append("Subsetting Applied:")
            if 'rating' in self.subset_info:
                rating_info = self.subset_info['rating']
                lines.append(
                    f"  Rating: {rating_info['type']} "
                    f"(bounds: {rating_info['bounds']})"
                )
            if 'characteristics' in self.subset_info:
                lines.append(f"  Characteristics: {self.subset_info['characteristics']}")
            lines.append(
                f"  Observations: {self.subset_info['n_after_subset']:,} / "
                f"{self.subset_info['n_original']:,}"
            )

        lines.append("-" * 60)
        lines.append("\nPearson Correlations (Time-Series Average):")
        lines.append("-" * 40)
        lines.append(self.pearson_avg.round(3).to_string())

        lines.append("\nSpearman Rank Correlations (Time-Series Average):")
        lines.append("-" * 40)
        lines.append(self.spearman_avg.round(3).to_string())

        return "\n".join(lines)
