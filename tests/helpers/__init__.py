"""Test helper utilities for PyBondLab."""

from .diff_report import write_diff_artifact, write_diff_artifact_by_status, ARTIFACT_ROOTS
from .manifest import (
    load_manifest,
    load_datasets,
    load_yaml,
    get_expected_outputs_dir,
    get_specs_dir,
)
from .canonicalize import (
    Status,
    canonicalize_metric,
    canonicalize_metrics,
    assert_close_df,
    compare_with_mean_fallback,
    compare_from_manifest_rule,
)
from .runner_adapter import run_functionality, FunctionalityResult, build_dataset
from .contracts import (
    MetricContract,
    FunctionalityContract,
    extract_metric_contract,
    extract_functionality_contract,
    save_contract,
    load_contract,
    verify_contract_match,
)
from .baselines import (
    BaselineEntry,
    ComparisonResult,
    load_main_baseline,
    load_data_uncertainty_baseline,
    load_singlesort_baseline,
    compare_to_baseline,
    get_baseline_test_configs,
    BASELINE_PATHS,
)

__all__ = [
    # diff_report
    "write_diff_artifact",
    "write_diff_artifact_by_status",
    "ARTIFACT_ROOTS",
    # manifest
    "load_manifest",
    "load_datasets",
    "load_yaml",
    "get_expected_outputs_dir",
    "get_specs_dir",
    # canonicalize
    "Status",
    "canonicalize_metric",
    "canonicalize_metrics",
    "assert_close_df",
    "compare_with_mean_fallback",
    "compare_from_manifest_rule",
    # runner_adapter
    "run_functionality",
    "FunctionalityResult",
    "build_dataset",
    # contracts
    "MetricContract",
    "FunctionalityContract",
    "extract_metric_contract",
    "extract_functionality_contract",
    "save_contract",
    "load_contract",
    "verify_contract_match",
    # baselines
    "BaselineEntry",
    "ComparisonResult",
    "load_main_baseline",
    "load_data_uncertainty_baseline",
    "load_singlesort_baseline",
    "compare_to_baseline",
    "get_baseline_test_configs",
    "BASELINE_PATHS",
]
