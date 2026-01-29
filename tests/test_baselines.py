"""
Baseline validation tests for PyBondLab.

These tests validate that the current PyBondLab implementation produces results
that match the stored baseline (source of truth) values.

Per AGENT_PLAN.md:
- PASS: Both EW and VW match within tolerance (1e-10)
- SOFT: Point-wise fails but would pass with looser tolerance (1e-6)
- FAIL: Both checks fail

Baselines:
- main: 12 SingleSort/DoubleSort configurations (generate_synthetic_data)
- data_uncertainty: 17 Momentum filter configurations (generate_synthetic_data_fast)
- singlesort: 40 SingleSort filter configurations (generate_synthetic_data_fast)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import PyBondLab as pbl
from PyBondLab.pbl_test import (
    generate_synthetic_data,
    run_all_baseline_tests,
    RANDOM_SEED,
    N_DATES,
    N_BONDS,
)

from tests.helpers import (
    load_main_baseline,
    load_data_uncertainty_baseline,
    load_singlesort_baseline,
    compare_to_baseline,
    BaselineEntry,
)


# Tolerance levels
EXACT_TOLERANCE = 1e-10
SOFT_TOLERANCE = 1e-6


class TestMainBaseline:
    """Test suite for main baseline (12 SingleSort/DoubleSort configurations)."""

    @pytest.fixture(scope="class")
    def data(self):
        """Generate synthetic data matching baseline generation."""
        return generate_synthetic_data(
            n_dates=N_DATES,
            n_bonds=N_BONDS,
            seed=RANDOM_SEED,
        )

    @pytest.fixture(scope="class")
    def baselines(self):
        """Load main baseline entries."""
        return load_main_baseline()

    @pytest.fixture(scope="class")
    def current_results(self, data):
        """Run all baseline tests with current code."""
        return run_all_baseline_tests(data, verbose=False)

    @pytest.mark.parametrize("test_name", [
        "test_01_single_hp1_noband_noturn",
        "test_02_single_hp1_noband_turn",
        "test_03_single_hp1_band1_turn",
        "test_04_single_hp1_band2_turn",
        "test_05_single_hp3_noband_turn",
        "test_06_single_hp3_band1_turn",
        "test_07_double_uncond_hp1_noband_turn",
        "test_08_double_cond_hp1_noband_turn",
        "test_09_double_uncond_hp3_band1_turn",
        "test_10_double_cond_hp3_band1_turn",
        "test_11_single_hp1_turn_chars2",
        "test_12_single_hp3_band1_turn_chars3",
    ])
    def test_baseline_match(self, test_name, baselines, current_results):
        """Verify current code matches baseline for given test configuration."""
        assert test_name in baselines, f"Test '{test_name}' not found in baselines"
        assert test_name in current_results, f"Test '{test_name}' not found in current results"

        baseline = baselines[test_name]
        result = current_results[test_name]

        comparison = compare_to_baseline(
            baseline=baseline,
            actual_ew_mean=result.ew_ls_mean,
            actual_vw_mean=result.vw_ls_mean,
            tolerance=EXACT_TOLERANCE,
            mean_tolerance=SOFT_TOLERANCE,
        )

        # Log details for debugging
        print(f"\n{test_name}:")
        print(f"  Baseline: EW={baseline.ew_mean:.10f}, VW={baseline.vw_mean:.10f}")
        print(f"  Current:  EW={result.ew_ls_mean:.10f}, VW={result.vw_ls_mean:.10f}")
        print(f"  {comparison.message}")

        assert comparison.status in ("PASS", "SOFT"), (
            f"Test '{test_name}' failed baseline comparison: {comparison.message}"
        )

        # Strict check: prefer PASS over SOFT
        if comparison.status == "SOFT":
            pytest.skip(f"SOFT match (needs review): {comparison.message}")


class TestDataUncertaintyBaseline:
    """Test suite for data uncertainty baseline (17 Momentum filter configurations)."""

    @pytest.fixture(scope="class")
    def data(self):
        """Generate synthetic data matching baseline generation."""
        from PyBondLab.pbl_test import generate_synthetic_data_fast
        return generate_synthetic_data_fast(
            n_dates=N_DATES,
            n_bonds=N_BONDS,
            seed=RANDOM_SEED,
        )

    @pytest.fixture(scope="class")
    def baselines(self):
        """Load data uncertainty baseline entries."""
        return load_data_uncertainty_baseline()

    def _run_momentum_with_filter(self, data, filter_type, filter_level, filter_key):
        """Run Momentum strategy with specified filter."""
        import ast

        filters = None
        if filter_type and filter_type not in ("none", "baseline"):
            # Parse filter level - can be a number, string, or list string like "[-0.3, 0.3]"
            level = filter_level
            if isinstance(level, str):
                # Try to parse as a list first
                if level.startswith("["):
                    try:
                        level = ast.literal_eval(level)
                    except (ValueError, SyntaxError):
                        pass
                else:
                    try:
                        level = float(level)
                    except ValueError:
                        pass

            if filter_type == "trim":
                filters = {"adj": "trim", "level": level}
            elif filter_type == "price":
                filters = {"adj": "price", "level": level}
            elif filter_type == "bounce":
                filters = {"adj": "bounce", "level": level}
            elif filter_type == "wins":
                # Wins filter uses percentile level and location
                # Parse location from filter_key (e.g., wins_99_both -> both)
                parts = filter_key.split("_")
                loc = parts[-1] if len(parts) >= 3 else "both"
                filters = {"adj": "wins", "level": level, "loc": loc}

        mom = pbl.Momentum(holding_period=3, lookback_period=3, skip=1, num_portfolios=5)
        sf = pbl.StrategyFormation(
            data,
            strategy=mom,
            filters=filters,
            turnover=False,
            verbose=False,
        )
        result = sf.fit()
        ew_ls, vw_ls = result.get_long_short()
        return ew_ls.mean(), vw_ls.mean()

    @pytest.mark.parametrize("filter_key", [
        "no_filter",
        "trim_0.2",
        "trim_0.5",
        "trim_-0.3",
        "trim_-0.3_0.3",
        "price_50",
        "price_200",
        "price_500",
        "price_20_500",
        "bounce_0.05",
        "bounce_-0.05",
        "bounce_0.1",
        "bounce_-0.05_0.05",
        "wins_99_both",
        "wins_99_right",
        "wins_95_both",
        "wins_95_left",
    ])
    def test_momentum_filter_baseline(self, filter_key, data, baselines):
        """Verify Momentum strategy matches baseline for given filter configuration."""
        if filter_key not in baselines:
            pytest.skip(f"Filter '{filter_key}' not found in baselines")

        baseline = baselines[filter_key]
        config = baseline.config

        # Run current code
        ew_mean, vw_mean = self._run_momentum_with_filter(
            data,
            filter_type=config.get("filter_type", "baseline"),
            filter_level=config.get("filter_level"),
            filter_key=filter_key,
        )

        comparison = compare_to_baseline(
            baseline=baseline,
            actual_ew_mean=ew_mean,
            actual_vw_mean=vw_mean,
            tolerance=EXACT_TOLERANCE,
            mean_tolerance=SOFT_TOLERANCE,
        )

        print(f"\n{filter_key}:")
        print(f"  Baseline: EW={baseline.ew_mean:.10f}, VW={baseline.vw_mean:.10f}")
        print(f"  Current:  EW={ew_mean:.10f}, VW={vw_mean:.10f}")
        print(f"  {comparison.message}")

        assert comparison.status in ("PASS", "SOFT"), (
            f"Filter '{filter_key}' failed baseline comparison: {comparison.message}"
        )


class TestSingleSortBaseline:
    """Test suite for singlesort baseline (40 HP/dynamic_weights/filter configurations)."""

    @pytest.fixture(scope="class")
    def data(self):
        """Generate synthetic data matching baseline generation."""
        from PyBondLab.pbl_test import generate_synthetic_data_fast
        return generate_synthetic_data_fast(
            n_dates=N_DATES,
            n_bonds=N_BONDS,
            seed=RANDOM_SEED,
        )

    @pytest.fixture(scope="class")
    def baselines(self):
        """Load singlesort baseline entries."""
        return load_singlesort_baseline()

    def _run_singlesort_with_config(self, data, hp, dynamic_weights, filter_type, filter_level):
        """Run SingleSort strategy with specified configuration."""
        from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig

        filters = None
        if filter_type and filter_type not in ("none", "no_filter"):
            # Parse filter level - can be a number or string
            level = filter_level
            if isinstance(level, str) and level != "None":
                try:
                    level = float(level)
                except ValueError:
                    pass

            if filter_type == "trim":
                filters = {"adj": "trim", "level": level}
            elif filter_type == "price":
                filters = {"adj": "price", "level": level}
            elif filter_type == "bounce":
                filters = {"adj": "bounce", "level": level}
            elif filter_type == "wins":
                filters = {"adj": "wins", "level": level}

        config = StrategyFormationConfig(
            data=DataConfig(),
            formation=FormationConfig(dynamic_weights=dynamic_weights),
        )

        strategy = pbl.SingleSort(
            holding_period=hp,
            sort_var="signal1",
            num_portfolios=5,
        )

        sf = pbl.StrategyFormation(
            data,
            strategy=strategy,
            filters=filters,
            turnover=False,
            config=config,
            verbose=False,
        )
        result = sf.fit()
        ew_ls, vw_ls = result.get_long_short()
        return ew_ls.mean(), vw_ls.mean()

    @pytest.mark.parametrize("test_name", [
        # HP=1, dynamic_weights=True
        "hp1_dw_true_trim_0.2",
        "hp1_dw_true_trim_-0.3",
        "hp1_dw_true_price_50",
        "hp1_dw_true_price_200",
        "hp1_dw_true_bounce_0.05",
        "hp1_dw_true_bounce_-0.05",
        "hp1_dw_true_wins_99_both",
        "hp1_dw_true_wins_99_right",
        "hp1_dw_true_wins_95_both",
        "hp1_dw_true_no_filter",
        # HP=1, dynamic_weights=False
        "hp1_dw_false_trim_0.2",
        "hp1_dw_false_trim_-0.3",
        "hp1_dw_false_price_50",
        "hp1_dw_false_price_200",
        "hp1_dw_false_bounce_0.05",
        "hp1_dw_false_bounce_-0.05",
        "hp1_dw_false_wins_99_both",
        "hp1_dw_false_wins_99_right",
        "hp1_dw_false_wins_95_both",
        "hp1_dw_false_no_filter",
        # HP=3, dynamic_weights=True
        "hp3_dw_true_trim_0.2",
        "hp3_dw_true_trim_-0.3",
        "hp3_dw_true_price_50",
        "hp3_dw_true_price_200",
        "hp3_dw_true_bounce_0.05",
        "hp3_dw_true_bounce_-0.05",
        "hp3_dw_true_wins_99_both",
        "hp3_dw_true_wins_99_right",
        "hp3_dw_true_wins_95_both",
        "hp3_dw_true_no_filter",
        # HP=3, dynamic_weights=False
        "hp3_dw_false_trim_0.2",
        "hp3_dw_false_trim_-0.3",
        "hp3_dw_false_price_50",
        "hp3_dw_false_price_200",
        "hp3_dw_false_bounce_0.05",
        "hp3_dw_false_bounce_-0.05",
        "hp3_dw_false_wins_99_both",
        "hp3_dw_false_wins_99_right",
        "hp3_dw_false_wins_95_both",
        "hp3_dw_false_no_filter",
    ])
    def test_singlesort_baseline(self, test_name, data, baselines):
        """Verify SingleSort strategy matches baseline for given configuration."""
        if test_name not in baselines:
            pytest.skip(f"Test '{test_name}' not found in baselines")

        baseline = baselines[test_name]
        config = baseline.config

        # Parse config
        hp = config.get("holding_period", 1)
        dw = config.get("dynamic_weights", True)
        filter_type = config.get("filter_type")
        filter_level = config.get("filter_level")

        # Run current code
        ew_mean, vw_mean = self._run_singlesort_with_config(
            data, hp, dw, filter_type, filter_level
        )

        comparison = compare_to_baseline(
            baseline=baseline,
            actual_ew_mean=ew_mean,
            actual_vw_mean=vw_mean,
            tolerance=EXACT_TOLERANCE,
            mean_tolerance=SOFT_TOLERANCE,
        )

        print(f"\n{test_name}:")
        print(f"  Baseline: EW={baseline.ew_mean:.10f}, VW={baseline.vw_mean:.10f}")
        print(f"  Current:  EW={ew_mean:.10f}, VW={vw_mean:.10f}")
        print(f"  {comparison.message}")

        assert comparison.status in ("PASS", "SOFT"), (
            f"Test '{test_name}' failed baseline comparison: {comparison.message}"
        )


class TestBaselineConsistency:
    """Meta-tests to verify baseline files are loadable and consistent."""

    def test_main_baseline_loads(self):
        """Verify main baseline file can be loaded."""
        baselines = load_main_baseline()
        assert len(baselines) == 12, f"Expected 12 main baselines, got {len(baselines)}"

    def test_data_uncertainty_baseline_loads(self):
        """Verify data uncertainty baseline file can be loaded."""
        baselines = load_data_uncertainty_baseline()
        assert len(baselines) > 0, "Data uncertainty baseline is empty"

    def test_singlesort_baseline_loads(self):
        """Verify singlesort baseline file can be loaded."""
        baselines = load_singlesort_baseline()
        assert len(baselines) > 0, "SingleSort baseline is empty"

    def test_baseline_entries_have_required_fields(self):
        """Verify all baseline entries have required fields."""
        for loader_name, loader in [
            ("main", load_main_baseline),
            ("data_uncertainty", load_data_uncertainty_baseline),
            ("singlesort", load_singlesort_baseline),
        ]:
            baselines = loader()
            for name, entry in baselines.items():
                assert isinstance(entry, BaselineEntry), (
                    f"{loader_name}/{name}: Expected BaselineEntry"
                )
                assert entry.name == name, (
                    f"{loader_name}/{name}: Name mismatch"
                )
                assert isinstance(entry.ew_mean, float), (
                    f"{loader_name}/{name}: ew_mean should be float"
                )
                assert isinstance(entry.vw_mean, float), (
                    f"{loader_name}/{name}: vw_mean should be float"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
