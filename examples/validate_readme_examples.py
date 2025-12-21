#!/usr/bin/env python
"""
Validate all code examples in BatchStrategyFormation_README.md and
BatchWithinFirmSortFormation_README.md to ensure they actually work.
"""

import sys
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

# Generate test data once
def generate_test_data(n_dates=60, n_bonds=200, n_firms=40, seed=42):
    """Generate synthetic data matching README requirements."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='M')

    rows = []
    for d, date in enumerate(dates):
        for bond_id in range(n_bonds):
            firm_id = bond_id % n_firms
            rows.append({
                'date': date,
                'ID': f'BOND_{bond_id:04d}',
                'ret': np.random.randn() * 0.02,
                'VW': np.random.uniform(100, 1000),
                'RATING_NUM': np.random.randint(1, 23),
                'PERMNO': f'FIRM_{firm_id:03d}',
                'MATURITY': np.random.uniform(1, 10),
                'DURATION': np.random.uniform(0.5, 12),
                # Signals
                'momentum': np.random.randn(),
                'momentum_3m': np.random.randn(),
                'momentum_6m': np.random.randn(),
                'momentum_12m': np.random.randn(),
                'reversal': np.random.randn(),
                'reversal_1m': np.random.randn(),
                'reversal_3m': np.random.randn(),
                'value': np.random.randn(),
                'quality': np.random.randn(),
                'size': np.random.randn(),
                'low_vol': np.random.randn(),
                'sig1': np.random.randn(),
                'sig2': np.random.randn(),
                'sig3': np.random.randn(),
                'sig4': np.random.randn(),
                'sig5': np.random.randn(),
                'signal1': np.random.randn(),
                'signal2': np.random.randn(),
                'credit_spread': np.random.randn(),
                'yield_spread': np.random.randn(),
                'duration_spread': np.random.randn(),
                'spread': np.random.randn(),
                'duration': np.random.uniform(1, 10),
                'rating': np.random.randint(1, 23),
                # Custom column name versions
                'cusip_id': f'BOND_{bond_id:04d}',
                'ret_vw': np.random.randn() * 0.02,
                'ret_vw_bgn': np.random.randn() * 0.02,
                'mcap_e': np.random.uniform(100, 1000),
                'spc_rat': np.random.randint(1, 23),
                'firm_identifier': f'FIRM_{firm_id:03d}',
                'mom3': np.random.randn(),
                'mom6': np.random.randn(),
                'val': np.random.randn(),
                'my_signal': np.random.randn(),
                'other_signal': np.random.randn(),
                'spread_1': np.random.randn(),
                'spread_2': np.random.randn(),
                'oas_spread': np.random.randn(),
                'liquidity': np.random.randn(),
            })

    data = pd.DataFrame(rows)
    return data


def test_example(name, code_func):
    """Test a single example and report success/failure."""
    try:
        code_func()
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}")
        print(f"         Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("Validating README Code Examples")
    print("=" * 70)

    data = generate_test_data()
    results = []

    # =========================================================================
    # BatchStrategyFormation_README.md Examples
    # =========================================================================
    print("\n" + "=" * 70)
    print("BatchStrategyFormation_README.md")
    print("=" * 70)

    # Example 1: Quick Start
    def test_bsf_quick_start():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value', 'size'],
            holding_period=1,
            num_portfolios=5,
            turnover=False,
        )
        results = batch.fit()

        ew_ls, vw_ls = results['momentum'].get_long_short()
        sharpe = ew_ls.mean() / ew_ls.std() * 12**0.5
        assert not np.isnan(sharpe), "Sharpe is NaN"

    results.append(test_example("Quick Start", test_bsf_quick_start))

    # Example 2: Basic Usage with summary_df
    def test_bsf_basic_usage():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum_3m', 'momentum_6m', 'reversal', 'value'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            verbose=False,
        )
        results = batch.fit()

        # summary_df should exist
        summary = results.summary_df
        assert summary is not None, "summary_df is None"
        assert len(summary) == 4, f"Expected 4 rows, got {len(summary)}"

    results.append(test_example("Basic Usage with summary_df", test_bsf_basic_usage))

    # Example 3: Custom Column Names
    def test_bsf_custom_columns():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['mom3', 'mom6', 'val'],
            columns={
                'ID': 'cusip_id',
                'ret': 'ret_vw',
                'VW': 'mcap_e',
                'RATING_NUM': 'spc_rat',
            },
            holding_period=1,
            num_portfolios=5,
            turnover=False,
        )
        results = batch.fit()
        ew_ls, _ = results['mom3'].get_long_short()
        assert len(ew_ls) > 0, "No results"

    results.append(test_example("Custom Column Names", test_bsf_custom_columns))

    # Example 4: Fast Path with filters
    def test_bsf_fast_path():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['sig1', 'sig2', 'sig3', 'sig4', 'sig5'],
            holding_period=1,
            num_portfolios=5,
            turnover=False,
            chars=None,
            banding=None,
            rating='IG',
            subset_filter={'MATURITY': (1, 5)},
            verbose=False,
        )
        results = batch.fit()
        assert 'sig1' in results, "sig1 not in results"

    results.append(test_example("Fast Path with Filters", test_bsf_fast_path))

    # Example 5: With Turnover and Characteristics
    def test_bsf_turnover_chars():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            chars=['duration', 'spread'],
            banding=1,
            verbose=False,
        )
        results = batch.fit()

        ew_turn, vw_turn = results['momentum'].get_turnover()
        assert ew_turn is not None, "turnover is None"

        ew_chars, vw_chars = results['momentum'].get_characteristics()
        assert 'duration' in ew_chars, "duration not in chars"

    results.append(test_example("Turnover and Characteristics", test_bsf_turnover_chars))

    # Example 6: Dictionary-like Access
    def test_bsf_dict_access():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value', 'size'],
            holding_period=1,
            num_portfolios=5,
            turnover=False,
            verbose=False,
        )
        results = batch.fit()

        # Get results for specific signal
        momentum_result = results['momentum']

        # Check available signals
        keys = list(results.keys())
        assert 'momentum' in keys
        assert 'value' in keys
        assert 'size' in keys

        # Iterate over all results
        for signal, result in results.items():
            ew_ls, vw_ls = result.get_long_short()
            sharpe = ew_ls.mean() / ew_ls.std() * 12**0.5
            # Just check it doesn't crash

    results.append(test_example("Dictionary-like Access", test_bsf_dict_access))

    # Example 7: Long-Short Returns
    def test_bsf_long_short():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum'],
            holding_period=1,
            num_portfolios=5,
            turnover=False,
            verbose=False,
        )
        results = batch.fit()

        ew_ls, vw_ls = results['momentum'].get_long_short()

        # ew_ls is a pandas Series with DatetimeIndex
        assert isinstance(ew_ls, pd.Series), "ew_ls is not a Series"
        assert isinstance(ew_ls.index, pd.DatetimeIndex), "Index is not DatetimeIndex"

        # Compute statistics
        mean_ann = ew_ls.mean() * 12
        std_ann = ew_ls.std() * 12**0.5
        sharpe = ew_ls.mean() / ew_ls.std() * 12**0.5
        assert not np.isnan(sharpe), "Sharpe is NaN"

    results.append(test_example("Long-Short Returns", test_bsf_long_short))

    # Example 8: Factor Returns DataFrame
    def test_bsf_factor_returns():
        from PyBondLab import BatchStrategyFormation

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value', 'size'],
            holding_period=1,
            num_portfolios=5,
            turnover=False,
            verbose=False,
        )
        results = batch.fit()

        # Get all factor returns as a DataFrame
        factor_returns = results.get_factor_returns(weight_type='ew')

        # Should have signals as columns, dates as index
        assert 'momentum' in factor_returns.columns
        assert 'value' in factor_returns.columns
        assert 'size' in factor_returns.columns

        # Compute correlation matrix
        corr = factor_returns.corr()
        assert corr is not None

    results.append(test_example("Factor Returns DataFrame", test_bsf_factor_returns))

    # Example 9: extract_panel
    def test_bsf_extract_panel():
        from PyBondLab import BatchStrategyFormation, extract_panel

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value', 'quality'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            chars=['duration', 'spread'],
            verbose=False,
        )
        results = batch.fit()

        # Extract unified panel
        panel = extract_panel(results)

        # Check panel structure
        assert 'date' in panel.columns
        assert 'factor' in panel.columns
        assert 'leg' in panel.columns
        assert 'weighting' in panel.columns
        assert 'return' in panel.columns

    results.append(test_example("extract_panel basic", test_bsf_extract_panel))

    # Example 10: extract_panel with sign correction
    def test_bsf_extract_panel_sign_correct():
        from PyBondLab import BatchStrategyFormation, extract_panel, NamingConfig

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            verbose=False,
        )
        results = batch.fit()

        panel = extract_panel(results, naming=NamingConfig(sign_correct=True))
        assert panel is not None
        assert len(panel) > 0

    results.append(test_example("extract_panel with sign correction", test_bsf_extract_panel_sign_correct))

    # Example 11: Pivot to wide format - THE PROBLEMATIC ONE
    def test_bsf_pivot_table():
        from PyBondLab import BatchStrategyFormation, extract_panel

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value', 'quality'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            verbose=False,
        )
        results = batch.fit()

        panel = extract_panel(results)

        # Get L-S returns in wide format (dates × factors)
        ls_returns = panel[panel['leg'] == 'ls'].pivot_table(
            index='date',
            columns=['factor', 'weighting'],
            values='return'
        )

        # Check the result
        assert ls_returns is not None, "pivot_table returned None"
        assert len(ls_returns) > 0, "pivot_table is empty"

        # Check that we can access data - the columns are MultiIndex
        assert isinstance(ls_returns.columns, pd.MultiIndex), "Columns should be MultiIndex"

        # Verify we can compute correlations between factors
        # Note: ls_returns columns are ('momentum', 'ew'), ('momentum', 'vw'), etc.
        # Access with ls_returns[('momentum', 'ew')] not ls_returns['momentum']['ew']

        # Check that the factors exist as first level
        factor_names = ls_returns.columns.get_level_values(0).unique()
        assert 'momentum' in factor_names, "momentum not in columns"
        assert 'value' in factor_names, "value not in columns"

    results.append(test_example("Pivot to wide format", test_bsf_pivot_table))

    # Example 12: Pivot table correlation - THE PROBLEMATIC SYNTAX
    def test_bsf_pivot_correlation():
        from PyBondLab import BatchStrategyFormation, extract_panel

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            verbose=False,
        )
        results = batch.fit()

        panel = extract_panel(results)
        ls_returns = panel[panel['leg'] == 'ls'].pivot_table(
            index='date',
            columns=['factor', 'weighting'],
            values='return'
        )

        # README example uses: ls_returns['momentum']['ew'].corr(ls_returns['value']['ew'])
        # But with MultiIndex columns, the correct syntax is:
        corr = ls_returns[('momentum', 'ew')].corr(ls_returns[('value', 'ew')])
        assert not np.isnan(corr), "Correlation is NaN"

        # Old syntax DOES NOT WORK with MultiIndex
        try:
            # This should fail
            _ = ls_returns['momentum']['ew']
            print("         WARNING: ls_returns['momentum']['ew'] should have failed but didn't")
        except (KeyError, TypeError):
            # Expected - this syntax doesn't work with MultiIndex
            pass

    results.append(test_example("Pivot correlation (MultiIndex access)", test_bsf_pivot_correlation))

    # Example 13: Filter panel by leg/weighting
    def test_bsf_filter_panel():
        from PyBondLab import BatchStrategyFormation, extract_panel

        batch = BatchStrategyFormation(
            data=data,
            signals=['momentum', 'value'],
            holding_period=1,
            num_portfolios=5,
            turnover=True,
            verbose=False,
        )
        results = batch.fit()

        panel = extract_panel(results)

        # Get only long-short EW returns
        ew_ls = panel[(panel['leg'] == 'ls') & (panel['weighting'] == 'ew')]
        assert len(ew_ls) > 0

        # Get only long leg data
        long_leg = panel[panel['leg'] == 'l']
        assert len(long_leg) > 0

        # Group by factor
        factor_means = panel[panel['leg'] == 'ls'].groupby(['factor', 'weighting'])['return'].mean()
        assert len(factor_means) > 0

    results.append(test_example("Filter panel by leg/weighting", test_bsf_filter_panel))

    # =========================================================================
    # BatchWithinFirmSortFormation_README.md Examples
    # =========================================================================
    print("\n" + "=" * 70)
    print("BatchWithinFirmSortFormation_README.md")
    print("=" * 70)

    # Example 1: Quick Start
    def test_bwf_quick_start():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['credit_spread', 'yield_spread', 'duration'],
            firm_id_col='PERMNO',
            turnover=False,
            verbose=False,
        )
        results = batch.fit()

        ew_ls, vw_ls = results['credit_spread'].get_long_short()
        sharpe = vw_ls.mean() / vw_ls.std() * 12**0.5
        assert not np.isnan(sharpe), "Sharpe is NaN"

    results.append(test_example("BWFSF Quick Start", test_bwf_quick_start))

    # Example 2: Basic Usage
    def test_bwf_basic_usage():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['spread_1', 'spread_2', 'duration_spread', 'yield_spread'],
            firm_id_col='PERMNO',
            min_bonds_per_firm=2,
            turnover=False,
            verbose=False,
        )
        results = batch.fit()

        for signal in batch.signals:
            ew_ls, vw_ls = results[signal].get_long_short()
            sharpe = vw_ls.mean() / vw_ls.std() * 12**0.5
            # Just check it doesn't crash

    results.append(test_example("BWFSF Basic Usage", test_bwf_basic_usage))

    # Example 3: Custom Column Names
    def test_bwf_custom_columns():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['my_signal', 'other_signal'],
            firm_id_col='firm_identifier',
            columns={
                'ID': 'cusip_id',
                'ret': 'ret_vw_bgn',
                'VW': 'mcap_e',
                'RATING_NUM': 'spc_rat',
            },
            turnover=False,
            verbose=False,
        )
        results = batch.fit()
        ew_ls, _ = results['my_signal'].get_long_short()
        assert len(ew_ls) > 0, "No results"

    results.append(test_example("BWFSF Custom Column Names", test_bwf_custom_columns))

    # Example 4: Fast Path with filters
    def test_bwf_fast_path():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['sig1', 'sig2', 'sig3'],
            firm_id_col='PERMNO',
            turnover=False,
            chars=None,
            rating='IG',
            subset_filter={'MATURITY': (1, 5)},
            verbose=False,
        )
        results = batch.fit()
        assert 'sig1' in results, "sig1 not in results"

    results.append(test_example("BWFSF Fast Path with Filters", test_bwf_fast_path))

    # Example 5: With Turnover and Characteristics
    def test_bwf_turnover_chars():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['credit_spread', 'yield_spread'],
            firm_id_col='PERMNO',
            turnover=True,
            chars=['duration', 'spread'],
            n_jobs=1,
            verbose=False,
        )
        results = batch.fit()

        ew_turn, vw_turn = results['credit_spread'].get_turnover()
        assert ew_turn is not None, "turnover is None"

        ew_chars, vw_chars = results['credit_spread'].get_characteristics()
        assert 'duration' in ew_chars, "duration not in chars"

    results.append(test_example("BWFSF Turnover and Characteristics", test_bwf_turnover_chars))

    # Example 6: Dictionary-like Access
    def test_bwf_dict_access():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['credit_spread', 'yield_spread', 'duration_spread'],
            firm_id_col='PERMNO',
            turnover=False,
            verbose=False,
        )
        results = batch.fit()

        cs_result = results['credit_spread']

        keys = list(results.keys())
        assert 'credit_spread' in keys
        assert 'yield_spread' in keys

        for signal, result in results.items():
            ew_ls, vw_ls = result.get_long_short()
            # Just check it doesn't crash

    results.append(test_example("BWFSF Dictionary-like Access", test_bwf_dict_access))

    # Example 7: Rating filter examples
    def test_bwf_rating_filters():
        from PyBondLab import BatchWithinFirmSortFormation

        # Investment Grade bonds only
        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['signal1', 'signal2'],
            firm_id_col='PERMNO',
            rating='IG',
            turnover=False,
            verbose=False,
        )
        results_ig = batch.fit()

        # Custom rating range
        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['signal1', 'signal2'],
            firm_id_col='PERMNO',
            rating=(7, 10),
            turnover=False,
            verbose=False,
        )
        results_bbb = batch.fit()

        assert 'signal1' in results_ig
        assert 'signal1' in results_bbb

    results.append(test_example("BWFSF Rating Filters", test_bwf_rating_filters))

    # Example 8: Custom Rating Bins
    def test_bwf_custom_rating_bins():
        from PyBondLab import BatchWithinFirmSortFormation

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['signal1', 'signal2'],
            firm_id_col='PERMNO',
            rating_bins=[-np.inf, 4, 10, np.inf],
            min_bonds_per_firm=2,
            turnover=False,
            verbose=False,
        )
        results = batch.fit()
        assert 'signal1' in results

    results.append(test_example("BWFSF Custom Rating Bins", test_bwf_custom_rating_bins))

    # Example 9: extract_panel
    def test_bwf_extract_panel():
        from PyBondLab import BatchWithinFirmSortFormation, extract_panel

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['credit_spread', 'yield_spread'],
            firm_id_col='PERMNO',
            turnover=True,
            chars=['duration', 'spread'],
            n_jobs=1,
            verbose=False,
        )
        results = batch.fit()

        panel = extract_panel(results)

        assert 'date' in panel.columns
        assert 'factor' in panel.columns
        assert 'leg' in panel.columns
        assert 'weighting' in panel.columns
        assert 'return' in panel.columns

    results.append(test_example("BWFSF extract_panel", test_bwf_extract_panel))

    # Example 10: Pivot to wide format
    def test_bwf_pivot_table():
        from PyBondLab import BatchWithinFirmSortFormation, extract_panel

        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['credit_spread', 'yield_spread'],
            firm_id_col='PERMNO',
            turnover=True,
            n_jobs=1,
            verbose=False,
        )
        results = batch.fit()

        panel = extract_panel(results)

        ls_returns = panel[panel['leg'] == 'ls'].pivot_table(
            index='date',
            columns=['factor', 'weighting'],
            values='return'
        )

        assert ls_returns is not None, "pivot_table returned None"
        assert len(ls_returns) > 0, "pivot_table is empty"
        assert isinstance(ls_returns.columns, pd.MultiIndex), "Columns should be MultiIndex"

    results.append(test_example("BWFSF Pivot to wide format", test_bwf_pivot_table))

    # Example 11: Result Consistency check
    def test_bwf_consistency():
        import PyBondLab as pbl
        from PyBondLab import BatchWithinFirmSortFormation

        # Batch approach
        batch = BatchWithinFirmSortFormation(
            data=data,
            signals=['signal1'],
            firm_id_col='PERMNO',
            turnover=False,
            verbose=False,
        )
        batch_results = batch.fit()
        batch_ew, batch_vw = batch_results['signal1'].get_long_short()

        # Individual approach
        strategy = pbl.WithinFirmSort(holding_period=1, sort_var='signal1', firm_id_col='PERMNO')
        sf = pbl.StrategyFormation(data, strategy, turnover=False, verbose=False)
        sf_results = sf.fit()
        sf_ew, sf_vw = sf_results.get_long_short()

        # Compare
        diff = (batch_vw - sf_vw).abs().max()
        assert diff < 1e-10, f"Results differ: {diff}"

    results.append(test_example("BWFSF Result Consistency", test_bwf_consistency))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nALL TESTS PASSED!")
        return True
    else:
        print(f"\n{total - passed} TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
