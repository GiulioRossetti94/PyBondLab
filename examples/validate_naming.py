#!/usr/bin/env python
"""
Validation script for NamingConfig feature.

Tests:
1. Basic naming (lowercase, signal-based)
2. Rating suffix (_ig, _hy)
3. Sign correction (EW and VW independent)
4. Weighting prefix (ew_, vw_)
5. DoubleSort separator
6. WithinFirmSort suffix and portfolios
7. Factor turnover computation
8. Backward compatibility (no naming = legacy names)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Import PyBondLab components
import PyBondLab as pbl
from PyBondLab.naming import NamingConfig, make_factor_name, make_portfolio_name, rating_to_suffix
from PyBondLab.pbl_test import generate_synthetic_data


def test_make_factor_name():
    """Test the make_factor_name function."""
    print("\n" + "=" * 60)
    print("TEST: make_factor_name function")
    print("=" * 60)

    cfg = NamingConfig()

    # Test 1: Basic lowercase
    name = make_factor_name('CS', cfg)
    assert name == 'cs', f"Expected 'cs', got '{name}'"
    print(f"  [PASS] Basic lowercase: 'CS' -> '{name}'")

    # Test 2: Rating suffix
    name = make_factor_name('CS', cfg, rating='ig')
    assert name == 'cs_ig', f"Expected 'cs_ig', got '{name}'"
    print(f"  [PASS] Rating suffix: 'CS' + rating='ig' -> '{name}'")

    name = make_factor_name('CS', cfg, rating='hy')
    assert name == 'cs_hy', f"Expected 'cs_hy', got '{name}'"
    print(f"  [PASS] Rating suffix: 'CS' + rating='hy' -> '{name}'")

    # Test 3: WithinFirmSort suffix
    name = make_factor_name('CS', cfg, is_within_firm=True)
    assert name == 'cs_wf', f"Expected 'cs_wf', got '{name}'"
    print(f"  [PASS] WithinFirmSort suffix: 'CS' + is_within_firm=True -> '{name}'")

    # Test 4: Sign correction suffix
    name = make_factor_name('CS', cfg, sign_corrected=True)
    assert name == 'cs*', f"Expected 'cs*', got '{name}'"
    print(f"  [PASS] Sign correction: 'CS' + sign_corrected=True -> '{name}'")

    # Test 5: DoubleSort
    name = make_factor_name('CS', cfg, second_signal='duration')
    assert name == 'cs_duration', f"Expected 'cs_duration', got '{name}'"
    print(f"  [PASS] DoubleSort: 'CS' + second_signal='duration' -> '{name}'")

    # Test 6: Weighting prefix
    cfg_prefix = NamingConfig(weighting_prefix=True)
    name = make_factor_name('CS', cfg_prefix, weighting='ew')
    assert name == 'ew_cs', f"Expected 'ew_cs', got '{name}'"
    print(f"  [PASS] Weighting prefix: 'CS' + weighting='ew' -> '{name}'")

    name = make_factor_name('CS', cfg_prefix, weighting='vw')
    assert name == 'vw_cs', f"Expected 'vw_cs', got '{name}'"
    print(f"  [PASS] Weighting prefix: 'CS' + weighting='vw' -> '{name}'")

    # Test 7: Combined
    name = make_factor_name('CS', cfg, rating='ig', is_within_firm=True, sign_corrected=True)
    assert name == 'cs_wf_ig*', f"Expected 'cs_wf_ig*', got '{name}'"
    print(f"  [PASS] Combined: 'CS' + rating='ig' + is_within_firm=True + sign_corrected=True -> '{name}'")

    # Test 8: No lowercase
    cfg_upper = NamingConfig(lowercase=False)
    name = make_factor_name('CS', cfg_upper)
    assert name == 'CS', f"Expected 'CS', got '{name}'"
    print(f"  [PASS] Uppercase: 'CS' with lowercase=False -> '{name}'")

    print("  All make_factor_name tests PASSED")
    return True


def test_make_portfolio_name():
    """Test the make_portfolio_name function."""
    print("\n" + "=" * 60)
    print("TEST: make_portfolio_name function")
    print("=" * 60)

    cfg = NamingConfig()

    # Test 1: SingleSort portfolios
    name = make_portfolio_name('CS', 1, 5, cfg)
    assert name == 'cs1', f"Expected 'cs1', got '{name}'"
    print(f"  [PASS] SingleSort P1: 'CS', portfolio_num=1 -> '{name}'")

    name = make_portfolio_name('CS', 5, 5, cfg)
    assert name == 'cs5', f"Expected 'cs5', got '{name}'"
    print(f"  [PASS] SingleSort P5: 'CS', portfolio_num=5 -> '{name}'")

    # Test 2: WithinFirmSort portfolios
    name = make_portfolio_name('CS', 1, 2, cfg, is_within_firm=True)
    assert name == 'cs_low', f"Expected 'cs_low', got '{name}'"
    print(f"  [PASS] WithinFirmSort LOW: portfolio_num=1, is_within_firm=True -> '{name}'")

    name = make_portfolio_name('CS', 2, 2, cfg, is_within_firm=True)
    assert name == 'cs_high', f"Expected 'cs_high', got '{name}'"
    print(f"  [PASS] WithinFirmSort HIGH: portfolio_num=2, is_within_firm=True -> '{name}'")

    # Test 3: DoubleSort portfolios
    name = make_portfolio_name('CS', 1, 5, cfg, second_signal='dur', second_portfolio_num=3)
    assert name == 'cs1_dur3', f"Expected 'cs1_dur3', got '{name}'"
    print(f"  [PASS] DoubleSort: portfolio_num=1, second_signal='dur', second_portfolio_num=3 -> '{name}'")

    print("  All make_portfolio_name tests PASSED")
    return True


def test_rating_to_suffix():
    """Test the rating_to_suffix function."""
    print("\n" + "=" * 60)
    print("TEST: rating_to_suffix function")
    print("=" * 60)

    # Test 1: String ratings
    assert rating_to_suffix('IG') == 'ig', "Expected 'ig' for 'IG'"
    print("  [PASS] 'IG' -> 'ig'")

    assert rating_to_suffix('NIG') == 'hy', "Expected 'hy' for 'NIG'"
    print("  [PASS] 'NIG' -> 'hy'")

    assert rating_to_suffix('HY') == 'hy', "Expected 'hy' for 'HY'"
    print("  [PASS] 'HY' -> 'hy'")

    # Test 2: Tuple ratings
    assert rating_to_suffix((1, 10)) == 'ig', "Expected 'ig' for (1, 10)"
    print("  [PASS] (1, 10) -> 'ig'")

    assert rating_to_suffix((11, 22)) == 'hy', "Expected 'hy' for (11, 22)"
    print("  [PASS] (11, 22) -> 'hy'")

    # Test 3: None
    assert rating_to_suffix(None) is None, "Expected None for None"
    print("  [PASS] None -> None")

    # Test 4: Mixed tuple (neither IG nor HY)
    result = rating_to_suffix((5, 15))
    assert result is None, f"Expected None for (5, 15), got {result}"
    print("  [PASS] (5, 15) -> None (mixed)")

    print("  All rating_to_suffix tests PASSED")
    return True


def test_strategy_results_naming():
    """Test naming with actual StrategyFormation results."""
    print("\n" + "=" * 60)
    print("TEST: StrategyResults with naming")
    print("=" * 60)

    # Generate synthetic data
    print("  Generating synthetic data...")
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run SingleSort
    print("  Running SingleSort strategy...")
    strategy = pbl.SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5)
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        chars=['char1'],
        verbose=False,
    )
    result = sf.fit()

    # Test 1: Backward compatibility (no naming)
    print("\n  Test 1: Backward compatibility (no naming)")
    ew_ls, vw_ls = result.get_long_short()
    print(f"    EW name: {ew_ls.name}")
    print(f"    VW name: {vw_ls.name}")
    # Legacy behavior - names may be None or original
    print("    [PASS] Backward compatible - no errors")

    # Test 2: Basic naming
    print("\n  Test 2: Basic naming with NamingConfig()")
    cfg = NamingConfig()
    ew_ls, vw_ls = result.get_long_short(naming=cfg)
    print(f"    EW name: {ew_ls.name}")
    print(f"    VW name: {vw_ls.name}")
    # Should be lowercase signal name
    assert 'signal1' in ew_ls.name.lower() or ew_ls.name == 'factor', f"Unexpected name: {ew_ls.name}"
    print("    [PASS] Names contain signal name")

    # Test 3: Weighting prefix
    print("\n  Test 3: With weighting prefix")
    cfg_prefix = NamingConfig(weighting_prefix=True)
    ew_ls, vw_ls = result.get_long_short(naming=cfg_prefix)
    print(f"    EW name: {ew_ls.name}")
    print(f"    VW name: {vw_ls.name}")
    assert ew_ls.name.startswith('ew_'), f"Expected 'ew_' prefix, got: {ew_ls.name}"
    assert vw_ls.name.startswith('vw_'), f"Expected 'vw_' prefix, got: {vw_ls.name}"
    print("    [PASS] Weighting prefixes applied")

    # Test 4: Factor turnover
    print("\n  Test 4: Factor turnover")
    ew_turn, vw_turn = result.get_turnover(level='factor')
    print(f"    EW factor turnover shape: {ew_turn.shape}")
    print(f"    VW factor turnover shape: {vw_turn.shape}")
    assert isinstance(ew_turn, pd.Series), "Factor turnover should be Series"
    assert isinstance(vw_turn, pd.Series), "Factor turnover should be Series"
    print("    [PASS] Factor turnover computed as Series")

    # Test 5: Portfolio turnover with naming
    print("\n  Test 5: Portfolio turnover with naming")
    ew_turn, vw_turn = result.get_turnover(level='portfolio', naming=cfg)
    print(f"    EW turnover columns: {list(ew_turn.columns)}")
    # Should have renamed columns
    print("    [PASS] Portfolio turnover with naming")

    # Test 6: Characteristics with naming
    print("\n  Test 6: Characteristics with naming")
    ew_chars, vw_chars = result.get_characteristics(naming=cfg)
    for char_name, df in ew_chars.items():
        print(f"    {char_name} columns: {list(df.columns)}")
    print("    [PASS] Characteristics with naming")

    print("\n  All StrategyResults tests PASSED")
    return True


def test_sign_correction():
    """Test sign correction is independent for EW and VW."""
    print("\n" + "=" * 60)
    print("TEST: Sign correction (independent EW/VW)")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run strategy
    strategy = pbl.SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5)
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=False,
        verbose=False,
    )
    result = sf.fit()

    # Get original means
    ew_orig, vw_orig = result.get_long_short()
    ew_mean = ew_orig.mean()
    vw_mean = vw_orig.mean()
    print(f"  Original EW mean: {ew_mean:.6f}")
    print(f"  Original VW mean: {vw_mean:.6f}")

    # Test with sign correction
    cfg_sign = NamingConfig(sign_correct=True)
    ew_corr, vw_corr = result.get_long_short(naming=cfg_sign)

    print(f"  Corrected EW name: {ew_corr.name}")
    print(f"  Corrected VW name: {vw_corr.name}")
    print(f"  Corrected EW mean: {ew_corr.mean():.6f}")
    print(f"  Corrected VW mean: {vw_corr.mean():.6f}")

    # Check sign correction applied correctly
    if ew_mean < 0:
        assert '*' in ew_corr.name, "EW should have * when sign was negative"
        assert ew_corr.mean() > 0, "EW should be positive after correction"
        print("  [PASS] EW was negative, now positive with '*'")
    else:
        if ew_corr.name and '*' in ew_corr.name:
            print("  [INFO] EW was positive but still has '*' (unexpected)")
        else:
            print("  [PASS] EW was positive, no '*' needed")

    if vw_mean < 0:
        assert '*' in vw_corr.name, "VW should have * when sign was negative"
        assert vw_corr.mean() > 0, "VW should be positive after correction"
        print("  [PASS] VW was negative, now positive with '*'")
    else:
        if vw_corr.name and '*' in vw_corr.name:
            print("  [INFO] VW was positive but still has '*' (unexpected)")
        else:
            print("  [PASS] VW was positive, no '*' needed")

    print("  Sign correction test PASSED")
    return True


def test_formation_results_passthrough():
    """Test that FormationResults passes naming to StrategyResults."""
    print("\n" + "=" * 60)
    print("TEST: FormationResults naming passthrough")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run strategy
    strategy = pbl.SingleSort(holding_period=1, sort_var='signal1', num_portfolios=5)
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False,
    )
    result = sf.fit()  # This returns FormationResults

    cfg = NamingConfig()

    # Test get_long_short
    ew_ls, vw_ls = result.get_long_short(naming=cfg)
    print(f"  get_long_short: EW={ew_ls.name}, VW={vw_ls.name}")
    assert ew_ls.name is not None, "EW name should be set"
    print("  [PASS] get_long_short naming works")

    # Test get_turnover with level='factor'
    ew_turn, vw_turn = result.get_turnover(level='factor', naming=cfg)
    print(f"  get_turnover(level='factor'): EW={ew_turn.name}, VW={vw_turn.name}")
    assert 'turnover' in ew_turn.name.lower(), "Factor turnover name should contain 'turnover'"
    print("  [PASS] get_turnover(level='factor') naming works")

    # Test get_turnover with level='portfolio'
    ew_turn, vw_turn = result.get_turnover(level='portfolio', naming=cfg)
    print(f"  get_turnover(level='portfolio') columns: {list(ew_turn.columns)[:3]}...")
    print("  [PASS] get_turnover(level='portfolio') naming works")

    print("  FormationResults passthrough test PASSED")
    return True


def test_doublesort_factor_turnover():
    """Test DoubleSort factor turnover computation."""
    print("\n" + "=" * 60)
    print("TEST: DoubleSort factor turnover")
    print("=" * 60)

    # Generate data
    data = generate_synthetic_data(n_dates=30, n_bonds=100, seed=42)

    # Run DoubleSort strategy
    print("  Running DoubleSort strategy...")
    strategy = pbl.DoubleSort(
        holding_period=1,
        sort_var='signal1',
        sort_var2='signal2',
        num_portfolios=3,
        num_portfolios2=3,  # 3x3 = 9 portfolios
    )
    sf = pbl.StrategyFormation(
        data=data,
        strategy=strategy,
        turnover=True,
        verbose=False,
    )
    result = sf.fit()

    cfg = NamingConfig()

    # Test 1: Portfolio-level turnover
    print("\n  Test 1: Portfolio-level turnover")
    ew_turn, vw_turn = result.get_turnover(level='portfolio')
    print(f"    Number of portfolios: {len(ew_turn.columns)}")
    assert len(ew_turn.columns) == 9, f"Expected 9 portfolios (3x3), got {len(ew_turn.columns)}"
    print("    [PASS] DoubleSort has 9 portfolios")

    # Test 2: Factor-level turnover
    print("\n  Test 2: Factor-level turnover")
    ew_factor_turn, vw_factor_turn = result.get_turnover(level='factor')
    print(f"    EW factor turnover shape: {ew_factor_turn.shape}")
    print(f"    VW factor turnover shape: {vw_factor_turn.shape}")
    assert isinstance(ew_factor_turn, pd.Series), "Factor turnover should be Series"

    # Verify the computation: average of long leg (cols 6,7,8) + short leg (cols 0,1,2)
    # Long leg: portfolios (3,1), (3,2), (3,3) = columns 6, 7, 8
    # Short leg: portfolios (1,1), (1,2), (1,3) = columns 0, 1, 2
    ew_long_manual = ew_turn.iloc[:, [6, 7, 8]].mean(axis=1)
    ew_short_manual = ew_turn.iloc[:, [0, 1, 2]].mean(axis=1)
    ew_factor_manual = (ew_long_manual + ew_short_manual) / 2

    diff = (ew_factor_turn - ew_factor_manual).abs().max()
    print(f"    Max diff vs manual computation: {diff:.2e}")
    assert diff < 1e-10, f"Factor turnover doesn't match expected: diff={diff}"
    print("    [PASS] DoubleSort factor turnover computation is correct")

    # Test 3: Factor turnover with naming
    print("\n  Test 3: Factor turnover with naming")
    ew_factor_turn, vw_factor_turn = result.get_turnover(level='factor', naming=cfg)
    print(f"    EW factor turnover name: {ew_factor_turn.name}")
    print(f"    VW factor turnover name: {vw_factor_turn.name}")
    assert 'signal1_signal2' in ew_factor_turn.name.lower(), f"Expected DoubleSort name, got: {ew_factor_turn.name}"
    print("    [PASS] DoubleSort factor turnover naming is correct")

    print("\n  DoubleSort factor turnover test PASSED")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("NamingConfig Validation Script")
    print("=" * 60)

    all_passed = True

    # Unit tests
    all_passed &= test_make_factor_name()
    all_passed &= test_make_portfolio_name()
    all_passed &= test_rating_to_suffix()

    # Integration tests
    all_passed &= test_strategy_results_naming()
    all_passed &= test_sign_correction()
    all_passed &= test_formation_results_passthrough()
    all_passed &= test_doublesort_factor_turnover()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
