"""
Correlation Analysis for Assayed Anomalies

This module provides functions to compute and analyze correlations
between different assayed anomalies (signals/strategies).

@authors: Giulio Rossetti & Alex Dickerson
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
import warnings


def compute_anomaly_correlations(
    runs: pd.DataFrame,
    weight: str = "EW",
    type_filter: str = "LS",
    min_overlap: int = 12,
) -> pd.DataFrame:
    """
    Compute correlation matrix between different assayed anomalies.

    Parameters
    ----------
    runs : pd.DataFrame
        Panel of assayed results with columns: ret, weight, type, ID, etc.
    weight : str, default "EW"
        Weight type to analyze ("EW" or "VW")
    type_filter : str, default "LS"
        Portfolio type ("LS", "L", or "S")
    min_overlap : int, default 12
        Minimum number of overlapping observations required

    Returns
    -------
    pd.DataFrame
        Correlation matrix with anomaly IDs as index and columns
    """
    # Filter to requested weight and type
    subset = runs[(runs["weight"] == weight) & (runs["type"] == type_filter)].copy()

    if subset.empty:
        warnings.warn(f"No data found for weight={weight}, type={type_filter}")
        return pd.DataFrame()

    # Pivot to wide format: dates × anomaly IDs
    try:
        wide = subset.pivot_table(
            values="ret",
            index=subset.index,  # dates
            columns="ID",
            aggfunc="first"  # in case of duplicates
        )
    except Exception as e:
        raise ValueError(f"Could not pivot data: {e}")

    # Compute correlations
    corr_matrix = wide.corr(min_periods=min_overlap)

    return corr_matrix


def get_correlation_summary(
    corr_matrix: pd.DataFrame,
    exclude_diagonal: bool = True
) -> pd.DataFrame:
    """
    Summarize correlation matrix with statistics.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix from compute_anomaly_correlations
    exclude_diagonal : bool, default True
        Exclude diagonal (self-correlations) from statistics

    Returns
    -------
    pd.DataFrame
        Summary statistics for each anomaly:
        - mean_corr: Average correlation with other anomalies
        - max_corr: Maximum correlation
        - min_corr: Minimum correlation
        - median_corr: Median correlation
        - std_corr: Standard deviation of correlations
    """
    if corr_matrix.empty:
        return pd.DataFrame()

    summary = []

    for col in corr_matrix.columns:
        corrs = corr_matrix[col].copy()

        if exclude_diagonal:
            corrs = corrs[corrs.index != col]

        corrs = corrs.dropna()

        if len(corrs) == 0:
            continue

        summary.append({
            "anomaly_id": col,
            "mean_corr": corrs.mean(),
            "median_corr": corrs.median(),
            "std_corr": corrs.std(),
            "min_corr": corrs.min(),
            "max_corr": corrs.max(),
            "n_pairs": len(corrs)
        })

    return pd.DataFrame(summary).set_index("anomaly_id")


def find_similar_anomalies(
    corr_matrix: pd.DataFrame,
    anomaly_id: str,
    threshold: float = 0.7,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Find anomalies similar to a given anomaly based on correlation.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    anomaly_id : str
        Target anomaly ID
    threshold : float, default 0.7
        Minimum correlation threshold
    top_n : int, optional
        Return only top N similar anomalies

    Returns
    -------
    pd.DataFrame
        Similar anomalies with their correlations, sorted descending
    """
    if anomaly_id not in corr_matrix.columns:
        raise ValueError(f"Anomaly ID '{anomaly_id}' not found in correlation matrix")

    # Get correlations for this anomaly
    corrs = corr_matrix[anomaly_id].copy()

    # Exclude self-correlation
    corrs = corrs[corrs.index != anomaly_id]

    # Filter by threshold
    corrs = corrs[corrs.abs() >= threshold]

    # Sort by absolute correlation
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)

    # Limit to top N if requested
    if top_n is not None:
        corrs = corrs.head(top_n)

    return corrs.to_frame("correlation")


def compute_pairwise_correlations(
    runs: pd.DataFrame,
    weight: str = "EW",
    type_filter: str = "LS",
    min_overlap: int = 12
) -> pd.DataFrame:
    """
    Compute all pairwise correlations in long format.

    Useful for filtering, sorting, and detailed analysis.

    Parameters
    ----------
    runs : pd.DataFrame
        Panel of assayed results
    weight : str, default "EW"
        Weight type
    type_filter : str, default "LS"
        Portfolio type
    min_overlap : int, default 12
        Minimum overlapping observations

    Returns
    -------
    pd.DataFrame
        Long-format pairwise correlations with columns:
        - anomaly_1: First anomaly ID
        - anomaly_2: Second anomaly ID
        - correlation: Correlation coefficient
        - n_obs: Number of overlapping observations
    """
    # Get correlation matrix
    corr_matrix = compute_anomaly_correlations(runs, weight, type_filter, min_overlap)

    if corr_matrix.empty:
        return pd.DataFrame()

    # Convert to long format
    pairs = []
    ids = corr_matrix.columns.tolist()

    for i, id1 in enumerate(ids):
        for id2 in ids[i+1:]:  # Only upper triangle (avoid duplicates)
            corr = corr_matrix.loc[id1, id2]
            if not np.isnan(corr):
                pairs.append({
                    "anomaly_1": id1,
                    "anomaly_2": id2,
                    "correlation": corr,
                })

    df = pd.DataFrame(pairs)

    # Add number of observations
    if not df.empty:
        subset = runs[(runs["weight"] == weight) & (runs["type"] == type_filter)]
        wide = subset.pivot_table(values="ret", index=subset.index, columns="ID", aggfunc="first")

        def count_overlap(row):
            id1, id2 = row["anomaly_1"], row["anomaly_2"]
            if id1 in wide.columns and id2 in wide.columns:
                return (~wide[id1].isna() & ~wide[id2].isna()).sum()
            return 0

        df["n_obs"] = df.apply(count_overlap, axis=1)

    return df.sort_values("correlation", key=abs, ascending=False)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = False,
    **kwargs
) -> "matplotlib.figure.Figure":
    """
    Plot correlation matrix as heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    figsize : tuple, default (12, 10)
        Figure size
    cmap : str, default "RdBu_r"
        Colormap
    vmin, vmax : float
        Color scale limits
    annot : bool, default False
        Annotate cells with values
    **kwargs
        Additional arguments passed to seaborn.heatmap

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False
        warnings.warn("Seaborn not available, using basic matplotlib heatmap")

    fig, ax = plt.subplots(figsize=figsize)

    if use_seaborn:
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            annot=annot,
            fmt=".2f" if annot else None,
            ax=ax,
            **kwargs
        )
    else:
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=90)
        ax.set_yticklabels(corr_matrix.index)

    ax.set_title("Anomaly Correlation Matrix")
    plt.tight_layout()

    return fig


# =============================================================================
# Convenience wrapper for AnomalyResults
# =============================================================================
def add_correlation_methods_to_results(AnomalyResultsClass):
    """
    Add correlation analysis methods to AnomalyResults class.

    This can be used to enhance the existing class without modifying it directly.
    """
    def get_correlations(self, weight="EW", type_filter="LS", min_overlap=12):
        """Compute correlation matrix between assayed anomalies."""
        return compute_anomaly_correlations(self.runs, weight, type_filter, min_overlap)

    def get_correlation_summary(self, weight="EW", type_filter="LS", min_overlap=12):
        """Get summary statistics for correlations."""
        corr_matrix = self.get_correlations(weight, type_filter, min_overlap)
        return get_correlation_summary(corr_matrix)

    def get_pairwise_correlations(self, weight="EW", type_filter="LS", min_overlap=12):
        """Get all pairwise correlations in long format."""
        return compute_pairwise_correlations(self.runs, weight, type_filter, min_overlap)

    def plot_correlation_heatmap(self, weight="EW", type_filter="LS", **kwargs):
        """Plot correlation matrix as heatmap."""
        corr_matrix = self.get_correlations(weight, type_filter)
        return plot_correlation_heatmap(corr_matrix, **kwargs)

    # Add methods to class
    AnomalyResultsClass.get_correlations = get_correlations
    AnomalyResultsClass.get_correlation_summary = get_correlation_summary
    AnomalyResultsClass.get_pairwise_correlations = get_pairwise_correlations
    AnomalyResultsClass.plot_correlation_heatmap = plot_correlation_heatmap

    return AnomalyResultsClass
