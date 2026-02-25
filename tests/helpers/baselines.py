"""
Baseline loading and comparison utilities.

Per AGENT_PLAN.md:
- Baselines are SOURCE OF TRUTH for numerical correctness
- Tests validate current code produces same results as baselines
- Classification: PASS (exact match), SOFT (mean matches), FAIL (both fail)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Base paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BASELINE_DIR = _PROJECT_ROOT / "PyBondLab" / "baseline_results"

# Baseline file locations (all baselines live in PyBondLab/baseline_results/)
BASELINE_PATHS = {
    "main": _BASELINE_DIR / "baseline_results.json",
    "main_pkl": _BASELINE_DIR / "baseline_results.pkl",
    "data_uncertainty": _BASELINE_DIR / "data_uncertainty_baseline.json",
    "singlesort": _BASELINE_DIR / "singlesort_baseline.json",
}


@dataclass
class BaselineEntry:
    """Container for a single baseline test result."""

    name: str
    ew_mean: float
    vw_mean: float
    config: Dict[str, Any]

    # Optional fields
    ew_std: Optional[float] = None
    vw_std: Optional[float] = None
    ew_turnover_mean: Optional[float] = None
    vw_turnover_mean: Optional[float] = None
    n_dates: Optional[int] = None


def load_main_baseline() -> Dict[str, BaselineEntry]:
    """
    Load the main baseline (12 SingleSort/DoubleSort configurations).

    Source: PyBondLab/baseline_results/baseline_results.json
    Data generator: generate_synthetic_data (seed=42, n_dates=60, n_bonds=500)

    Returns
    -------
    dict
        Mapping of test_name -> BaselineEntry
    """
    path = BASELINE_PATHS["main"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = {}
    for name, entry in data.items():
        baselines[name] = BaselineEntry(
            name=name,
            ew_mean=entry["ew_ls_mean"],
            vw_mean=entry["vw_ls_mean"],
            config=entry.get("config", {}),
            ew_turnover_mean=entry.get("ew_turnover_mean"),
            vw_turnover_mean=entry.get("vw_turnover_mean"),
            n_dates=entry.get("n_dates"),
        )

    return baselines


def load_data_uncertainty_baseline() -> Dict[str, BaselineEntry]:
    """
    Load the data uncertainty baseline (Momentum filter configurations).

    Source: examples/data_uncertainty_results/data_uncertainty_baseline.json
    Data generator: generate_synthetic_data_fast (seed=42, n_dates=60, n_bonds=500)
    Strategy: Momentum(3, 3, skip=1)

    Returns
    -------
    dict
        Mapping of filter_key -> BaselineEntry
    """
    path = BASELINE_PATHS["data_uncertainty"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = {}
    for name, entry in data.items():
        baselines[name] = BaselineEntry(
            name=name,
            ew_mean=entry["ew_ea_mean"],
            vw_mean=entry["vw_ea_mean"],
            config={
                "filter_type": entry.get("filter_type"),
                "filter_level": entry.get("filter_level"),
                "strategy_name": entry.get("strategy_name"),
            },
            ew_std=entry.get("ew_ea_std"),
            vw_std=entry.get("vw_ea_std"),
            n_dates=entry.get("n_dates"),
        )

    return baselines


def load_singlesort_baseline() -> Dict[str, BaselineEntry]:
    """
    Load the singlesort baseline (HP/dynamic_weights/filter configurations).

    Source: examples/data_uncertainty_results/singlesort_baseline.json
    Data generator: generate_synthetic_data_fast (seed=42, n_dates=60, n_bonds=500)
    Strategy: SingleSort(signal1, num_portfolios=5)

    Returns
    -------
    dict
        Mapping of test_name -> BaselineEntry
    """
    path = BASELINE_PATHS["singlesort"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = {}
    for name, entry in data.items():
        baselines[name] = BaselineEntry(
            name=name,
            ew_mean=entry["ew_ea_mean"],
            vw_mean=entry["vw_ea_mean"],
            config={
                "holding_period": entry.get("holding_period"),
                "dynamic_weights": entry.get("dynamic_weights"),
                "filter_type": entry.get("filter_type"),
                "filter_level": entry.get("filter_level"),
            },
            ew_std=entry.get("ew_ea_std"),
            vw_std=entry.get("vw_ea_std"),
            n_dates=entry.get("n_dates"),
        )

    return baselines


@dataclass
class ComparisonResult:
    """Result of comparing actual values against baseline."""

    status: str  # "PASS", "SOFT", "FAIL"
    ew_diff: float
    vw_diff: float
    message: str

    # For SOFT status
    ew_mean_diff: Optional[float] = None
    vw_mean_diff: Optional[float] = None


def compare_to_baseline(
    baseline: BaselineEntry,
    actual_ew_mean: float,
    actual_vw_mean: float,
    tolerance: float = 1e-10,
    mean_tolerance: float = 1e-6,
) -> ComparisonResult:
    """
    Compare actual results against baseline.

    Per AGENT_PLAN.md:
    - PASS: Both EW and VW match within tolerance
    - SOFT: Point-wise fails but would pass with looser tolerance (needs review)
    - FAIL: Both checks fail

    Parameters
    ----------
    baseline : BaselineEntry
        Expected baseline values
    actual_ew_mean : float
        Actual EW long-short mean
    actual_vw_mean : float
        Actual VW long-short mean
    tolerance : float
        Primary tolerance for exact match (default: 1e-10)
    mean_tolerance : float
        Secondary tolerance for SOFT classification (default: 1e-6)

    Returns
    -------
    ComparisonResult
        Comparison result with status and details
    """
    # Handle NaN baselines
    if np.isnan(baseline.ew_mean) and np.isnan(actual_ew_mean):
        ew_diff = 0.0
    elif np.isnan(baseline.ew_mean) or np.isnan(actual_ew_mean):
        ew_diff = float('inf')
    else:
        ew_diff = abs(baseline.ew_mean - actual_ew_mean)

    if np.isnan(baseline.vw_mean) and np.isnan(actual_vw_mean):
        vw_diff = 0.0
    elif np.isnan(baseline.vw_mean) or np.isnan(actual_vw_mean):
        vw_diff = float('inf')
    else:
        vw_diff = abs(baseline.vw_mean - actual_vw_mean)

    # Primary check: exact match
    ew_pass = ew_diff <= tolerance
    vw_pass = vw_diff <= tolerance

    if ew_pass and vw_pass:
        return ComparisonResult(
            status="PASS",
            ew_diff=ew_diff,
            vw_diff=vw_diff,
            message=f"Exact match: EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e}",
        )

    # Secondary check: mean tolerance (SOFT)
    ew_soft = ew_diff <= mean_tolerance
    vw_soft = vw_diff <= mean_tolerance

    if ew_soft and vw_soft:
        return ComparisonResult(
            status="SOFT",
            ew_diff=ew_diff,
            vw_diff=vw_diff,
            message=f"Mean match (needs review): EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e}",
            ew_mean_diff=ew_diff,
            vw_mean_diff=vw_diff,
        )

    # Both checks fail
    return ComparisonResult(
        status="FAIL",
        ew_diff=ew_diff,
        vw_diff=vw_diff,
        message=f"FAIL: EW diff={ew_diff:.2e}, VW diff={vw_diff:.2e}",
    )


def get_baseline_test_configs(baseline_type: str) -> List[str]:
    """
    Get list of test configuration names for a baseline type.

    Parameters
    ----------
    baseline_type : str
        "main", "data_uncertainty", or "singlesort"

    Returns
    -------
    list
        List of test configuration names
    """
    if baseline_type == "main":
        return list(load_main_baseline().keys())
    elif baseline_type == "data_uncertainty":
        return list(load_data_uncertainty_baseline().keys())
    elif baseline_type == "singlesort":
        return list(load_singlesort_baseline().keys())
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
