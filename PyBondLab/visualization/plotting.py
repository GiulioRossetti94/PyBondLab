# -*- coding: utf-8 -*-
"""
Created on Wed June 18 08:27:24 2025

@authors: Giulio Rossetti 
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Sequence, Optional, Union, Tuple, List, Any
from ._latex import set_latex, _esc

class PerformancePlotter:
    """
    Class for generating standardized performance plots from a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing performance data. Must include columns like
        'weight', 'type', 'avg', 'avg_t'.
        Optional columns: 'alpha_t' for alpha t-statistics plots, and any columns used in other methods.
    palette_map : dict
        Mapping of palette names to hex color codes.
    linestyles : dict
        Mapping of plot types to line style strings.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        self.df = df.copy()
        self._validate_dataframe()
        self.palette_map = {
            'blue': '#3182bd',
            'red': '#de2d26',
            'purple': '#6a51a3',
            'green': '#238b45'
        }
        self.linestyles = {
            'LS': '-',
            'L': '--',
            'S': ':'
        }

    def _validate_dataframe(self) -> None:
        required = {'weight', 'type'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    def plot_kde(
        self,
        colname: str = 'avg',
        weight_types: Union[str, Sequence[str]] = ('EW', 'VW'),
        types: Sequence[str] = ('LS', 'L', 'S'),
        types_label: Sequence[str] = ('Long-short', 'Long leg', 'Short leg'),
        palette: str = 'blue',
        figsize: Optional[Tuple[int, int]] = None,
        font_sizes: Tuple[float, float, float] = (12, 11, 11),
        x_label: str = 'Avg (%)',
        y_label: str = 'Density',
        n_points: int = 200,
        ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None
    ) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
        """
        Plot kernel density estimates for specified weight and type combinations.

        Returns
        -------
        fig : matplotlib Figure
        ax_or_axes : Axes or list of Axes
            The figure and axes containing the KDE plots.
        """
        if isinstance(weight_types, str):
            weight_types = [weight_types]
        panels = len(weight_types)

        if ax is not None:
            # multiple panels → need list of Axes
            if panels > 1:
                if not (isinstance(ax, (list, tuple)) and len(ax) == panels):
                    raise ValueError(f"Expected {panels} Axes, got {ax!r}")
                axes = list(ax)
            else:
                # single panel: accept single Axes or first of list
                axes = [ax[0]] if isinstance(ax, (list, tuple)) else [ax]
            fig = axes[0].figure
        else:
            # no ax → create new subplots
            if figsize is None:
                figsize = (6 * panels, 5)
            if panels > 1:
                fig, axes = plt.subplots(1, panels, figsize=figsize, sharey=True)
            else:
                fig, single = plt.subplots(figsize=figsize)
                axes = [single]

        title_fs, x_fs, y_fs = font_sizes
        line_color = self.palette_map.get(palette.lower(), self.palette_map['blue'])

        all_vals = []
        for w in weight_types:
            for t in types:
                vals = self.df.loc[(self.df['weight'] == w) & (self.df['type'] == t), colname].dropna().values
                if vals.size:
                    all_vals.append(vals)
        all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0])
        xmin, xmax = all_vals.min(), all_vals.max()
        x_grid = np.linspace(xmin, xmax, n_points)

        for axis, w in zip(axes, weight_types):
            for t, lbl in zip(types, types_label):
                data = self.df.loc[(self.df['weight'] == w) & (self.df['type'] == t), colname].dropna().values
                if data.size:
                    kde = gaussian_kde(data)
                    axis.plot(
                        x_grid,
                        kde(x_grid),
                        linestyle=self.linestyles.get(t, '-'),
                        color=line_color,
                        label=_esc(lbl)
                    )
            axis.set_title(rf"\textbf{{{_esc(w)}-Weighted}}", fontsize=title_fs)
            axis.set_xlabel(rf"\textit{{{_esc(x_label)}}}", fontsize=x_fs)
            axis.legend(frameon=False, fontsize=y_fs - 2)
            axis.axhline(0, color='grey', linewidth=0.8)

        axes[0].set_ylabel(rf"\textit{{{_esc(y_label)}}}", fontsize=y_fs)
        if ax is None:
            plt.tight_layout()
            plt.show()

        return fig, axes if len(axes) > 1 else axes[0]

    def plot_bar_tstats(
        self,
        types: Optional[Sequence[str]] = None,
        types_label: Optional[Sequence[str]] = None,
        weight_types: Union[str, Sequence[str]] = ('EW', 'VW'),
        figsize: Tuple[int, int] = (12, 6),
        font_sizes: Tuple[int, int, int] = (12, 11, 11),
        threshold: float = 1.96,
        ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None
    ) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
        """
        Plot bar charts of t-statistics for average and optionally alpha if present in the DataFrame.

        Returns
        -------
        fig : matplotlib Figure
        ax_or_axes : Axes or list of Axes
            Returns two side-by-side axes when 'alpha_t' is present, otherwise a single axis.
        """
        if isinstance(weight_types, str):
            weight_types = [weight_types]

        subdf = self.df[self.df['weight'].isin(weight_types)]
        if subdf.empty or 'avg_t' not in subdf:
            raise ValueError("No data or missing 'avg_t' column.")

        if types is None:
            types = subdf['type'].unique().tolist()
        if types_label is None:
            types_label = types

        has_alpha = 'alpha_t' in subdf.columns
        t_avg = subdf['avg_t'].sort_values(ascending=False).values
        t_alpha = subdf['alpha_t'].sort_values(ascending=False).values if has_alpha else None
        if ax is not None:
            if has_alpha:
                if not (isinstance(ax, (list, tuple)) and len(ax) == 2):
                    raise ValueError("Alpha present → must pass ax=[ax_avg, ax_alpha]")
                axes = list(ax)
            else:
                axes = [ax[0]] if isinstance(ax, (list, tuple)) else [ax]
            fig = axes[0].figure
        else:
            count = 2 if has_alpha else 1
            fig, axes = plt.subplots(1, count, figsize=figsize)
            if not has_alpha:
                axes = [axes]

        y_max = 1.1 * max(t_avg.max(), t_alpha.max() if t_alpha is not None else t_avg.max())
        y_min = 1.1 * min(t_avg.min(), t_alpha.min() if t_alpha is not None else t_avg.min(), 0)

        axes[0].bar(range(len(t_avg)), t_avg)
        axes[0].axhline(threshold, linestyle='--')
        axes[0].set_ylim([y_min, y_max])
        axes[0].set_title(r'(A) Average', fontsize=font_sizes[0])
        axes[0].set_xticks([])
        axes[0].grid(True)

        if has_alpha:
            axes[1].bar(range(len(t_alpha)), t_alpha)  # type: ignore
            axes[1].axhline(threshold, linestyle='--')
            axes[1].set_ylim([y_min, y_max])
            axes[1].set_title(r'(B) Alpha', fontsize=font_sizes[0])
            axes[1].set_xticks([])
            axes[1].grid(True)

        if ax is None:
            plt.tight_layout()
            plt.show()

        return fig, axes if len(axes) > 1 else axes[0]

    def plot_boxplot(
        self,
        columns: List[str],
        types: List[str] = ['LS', 'L', 'S'],
        types_label: List[str] = ['Long-short', 'Long leg', 'Short leg'],
        palette: str = 'blue',
        figsize: Optional[Tuple[int, int]] = None,
        font_sizes: Tuple[int, int] = (11, 11),
        ylims: Tuple[Optional[float], Optional[float]] = (None, None),
        ylabel: Optional[List[str]] = None,
        x_label_orientation: str = 'vertical',
        ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot boxplots for specified columns grouped by types.

        Returns
        -------
        fig : matplotlib Figure
        axes : list of Axes
        """
        n = len(columns)
        if ax is not None:
            if n > 1:
                if not (isinstance(ax, (list, tuple)) and len(ax) == n):
                    raise ValueError(f"Expected {n} Axes, got {ax!r}")
                axes = list(ax)
            else:
                axes = [ax[0] if isinstance(ax, (list, tuple)) else ax]
            fig = axes[0].figure
        else:
            if figsize is None:
                figsize = (8, 4 * n)
            fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
            if n == 1:
                axes = [axes]

        box_color = self.palette_map.get(palette.lower(), self.palette_map['blue'])
        rot = 90 if x_label_orientation.lower().startswith('vert') else 0
        x_fs, y_fs = font_sizes

        for axis, col, lab in zip(axes, columns, types_label):
            data = [self.df.loc[self.df['type'] == t, col].dropna() for t in types]
            axis.boxplot(
                data,
                labels=[_esc(lbl) for lbl in types_label],
                patch_artist=True,
                boxprops=dict(facecolor=box_color),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black')
            )
            label = lab if ylabel is None else ylabel[columns.index(col)]
            axis.set_ylabel(_esc(label), fontsize=y_fs)
            axis.axhline(0, color='grey', linewidth=0.8)
            if any(ylims):
                axis.set_ylim(*ylims)
            for tick in axis.get_xticklabels():
                tick.set_rotation(rot)
                tick.set_fontsize(x_fs)

        if ax is None:
            plt.tight_layout()
            plt.show()

        return fig, axes

    def plot_cumulative_paths(
        self,
        date_col: str = 'date',
        palette: str = 'blue',
        figsize: Optional[Tuple[int, int]] = None,
        font_sizes: Tuple[int, int] = (11, 11),
        ylims: Tuple[Optional[float], Optional[float]] = (None, None),
        ylabel: Optional[List[str]] = None,
        x_label_orientation: str = 'vertical',
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot cumulative return paths for each scenario and the median path.

        Returns
        -------
        fig : matplotlib Figure
        ax : Axes
        """
        data = self.df.copy()
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data.set_index(date_col, inplace=True)
        elif 'Unnamed: 0' in data.columns:
            data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
            data.set_index('Unnamed: 0', inplace=True)
        else:
            data.index = pd.to_datetime(data.index)

        # Create a unique series identifier per scenario & portfolio leg
        # data['series'] = (
        #     data['ID'].astype(str)
        #     + '_' + data['weight'].astype(str)
        #     + '_' + data['type'].astype(str)
        # )

        # 2) Pivot so each strategy-ID becomes its own column of returns
        #    (instead of pivoting on 'ret' itself)
        panel = data.pivot(columns='ID', values='ret')

        cumret = (1 + panel).cumprod()
        median = cumret.median(axis=1)

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=figsize or (8, 4))

        cumret.plot(ax=ax, legend=False, color='lightgrey', linewidth=1, alpha=0.8)
        median.plot(ax=ax, color='red', linewidth=2, label='Median Cumulative Return')

        ax.set_yscale('log')
        ax.set_title('Gross returns', fontsize=font_sizes[0])
        ax.set_ylabel(_esc('Value of $1 investment'), fontsize=font_sizes[1])
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        if ax is None:
            plt.tight_layout()
            plt.show()

        return fig, ax
