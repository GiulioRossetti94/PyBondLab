"""
06b - Slow-Path Anomaly Diagnostics with AssayAnomalyRunner
===========================================================
Small diagnostic example for the slow/full anomaly-assay path.

This example is intentionally narrow. It uses a compact specification grid to
show how to inspect:

- long-short return series
- turnover
- bond counts (`nbonds`)

Use this path when you want richer diagnostic output. If you only need
long-short returns across many specifications, use `assay_anomaly_fast`
or `BatchAssayAnomaly` instead.

PyBondLab features used:
  - `SingleSort`
  - `AssayAnomalyRunner` (advanced/internal runner)
  - slow path diagnostics via `turnover=True` and `save_idx=True`
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from _config import ASSAY_COLUMNS, load_panel

from PyBondLab import SingleSort
from PyBondLab.AnomalyAssayer import AssayAnomalyRunner


def ig_breakpoints_only(df: pd.DataFrame) -> pd.Series:
    """Compute breakpoints using IG bonds only."""
    return (df["RATING_NUM"] >= 1) & (df["RATING_NUM"] <= 10)


ig_breakpoints_only.required_columns = ["RATING_NUM"]


data = load_panel()
print(f"Panel: {len(data):,} observations")

# Use one seed strategy and let the runner expand a small grid around it.
strategy = SingleSort(
    sort_var="cs",
    holding_period=1,
    num_portfolios=5,
    breakpoint_universe_func=ig_breakpoints_only,
    verbose=False,
)

results = AssayAnomalyRunner(
    strategy=strategy,
    data=data,
    holding_periods=[1],
    nport=[3, 5],
    ratings=[None, "NIG"],
    subset_filter={
        "tmat": [
            (0, float("inf")),  # full maturity range
            (0, 5),             # short maturity subsample
        ]
    },
    breakpoint_universe_func=ig_breakpoints_only,
    dynamic_weights=True,
    turnover=True,   # force slow path and compute turnover
    save_idx=True,   # keep portfolio assignments so nbonds is available
    n_jobs=1,
    verbose=True,
    **ASSAY_COLUMNS,
).run()

runs = results.df.reset_index().rename(columns={"index": "date"})
print(f"\nRuns shape: {runs.shape}")
print(f"Columns: {runs.columns.tolist()}")

print("\nSample rows:")
sample_cols = ["date", "Holding", "Sort", "Rating", "Subset", "weight", "type", "ret", "nbonds", "TO"]
print(runs[sample_cols].head(12).to_string(index=False))

ls_runs = runs[runs["type"] == "LS"].copy()

summary = (
    ls_runs.groupby(["weight", "Sort", "Rating", "Subset"], dropna=False)
    .agg(
        mean_ret=("ret", "mean"),
        mean_nbonds=("nbonds", "mean"),
        mean_turnover=("TO", "mean"),
        n_obs=("ret", "count"),
    )
    .round(4)
    .reset_index()
    .sort_values(["weight", "Sort", "Rating", "Subset"])
)

print("\nLong-short diagnostic summary:")
print(summary.to_string(index=False))

full_results, recap = results.summary_results(nw_lag=3)
print("\nRecap by leg:")
print(recap.to_string())

print("\n[Done] Script 06b complete.")
