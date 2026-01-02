"""
Diagnostic tests for signals with low cross-sectional variation.

This script investigates what happens when we try to sort on signals
that don't vary much in the cross-section at each date.

Scenarios tested:
1. Constant signal - all bonds have identical values
2. Binary signal - only 2 unique values
3. Low-cardinality signal - 3-4 unique values for 5 portfolios
4. Clustered signal - values clustered around few points with small noise
5. Mostly constant with outliers - 95% same value, 5% different

For each scenario, we test:
- SingleSort (5 portfolios)
- DoubleSort (conditional, 2x5 portfolios)
- WithinFirmSort (HIGH/LOW)
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/user/PyBondLab-Dev')

import PyBondLab as pbl
from PyBondLab import StrategyFormation, SingleSort, DoubleSort, WithinFirmSort


def generate_base_data(n_dates=12, n_bonds=100, seed=42):
    """Generate base panel data without signal column."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='M')
    bond_ids = [f'BOND_{i:03d}' for i in range(n_bonds)]

    # Create balanced panel
    rows = []
    for d in dates:
        for b in bond_ids:
            rows.append({
                'date': d,
                'ID': b,
                'ret': np.random.normal(0.005, 0.02),
                'VW': np.random.uniform(100, 1000),
                'RATING_NUM': np.random.randint(1, 23),
                'PERMNO': f'FIRM_{int(b.split("_")[1]) % 20:03d}',  # 20 firms
            })

    data = pd.DataFrame(rows)
    return data


def add_constant_signal(data, value=5.0):
    """All bonds have identical signal value."""
    data = data.copy()
    data['signal'] = value
    return data


def add_binary_signal(data, seed=42):
    """Only 2 unique values (0 and 1)."""
    np.random.seed(seed)
    data = data.copy()
    data['signal'] = np.random.choice([0.0, 1.0], size=len(data))
    return data


def add_low_cardinality_signal(data, n_unique=3, seed=42):
    """Only n_unique distinct values."""
    np.random.seed(seed)
    data = data.copy()
    values = np.linspace(1, 10, n_unique)
    data['signal'] = np.random.choice(values, size=len(data))
    return data


def add_clustered_signal(data, n_clusters=3, noise=0.001, seed=42):
    """Values clustered around n_clusters points with tiny noise."""
    np.random.seed(seed)
    data = data.copy()
    cluster_centers = np.linspace(1, 10, n_clusters)
    clusters = np.random.choice(cluster_centers, size=len(data))
    data['signal'] = clusters + np.random.normal(0, noise, size=len(data))
    return data


def add_mostly_constant_signal(data, pct_same=0.95, seed=42):
    """Most bonds have same value, few outliers."""
    np.random.seed(seed)
    data = data.copy()
    n = len(data)
    n_same = int(n * pct_same)

    signals = np.full(n, 5.0)  # Default value
    outlier_idx = np.random.choice(n, size=n - n_same, replace=False)
    signals[outlier_idx] = np.random.uniform(1, 10, size=len(outlier_idx))

    data['signal'] = signals
    return data


def test_singlesort(data, scenario_name, num_portfolios=5):
    """Test SingleSort behavior."""
    print(f"\n  SingleSort (num_portfolios={num_portfolios}):")

    try:
        strategy = SingleSort(
            holding_period=1,
            sort_var='signal',
            num_portfolios=num_portfolios
        )

        sf = StrategyFormation(
            data=data,
            strategy=strategy,
            turnover=False,
            chars=['signal'],  # Track signal as char to verify
            verbose=False
        )

        result = sf.fit()

        # Get portfolio counts
        ew_ls, vw_ls = result.get_long_short()

        # Check bond counts per portfolio
        ew_chars, vw_chars = result.get_characteristics()
        signal_chars = ew_chars['signal']

        print(f"    LS returns computed: {len(ew_ls)} dates")
        print(f"    Mean LS return: {ew_ls.mean():.6f}")

        # Get bond counts if available
        try:
            counts = result.get_bond_count()
            avg_counts = counts.mean()
            print(f"    Avg bonds per portfolio: {avg_counts.values}")

            # Check for empty portfolios
            empty_ptfs = (avg_counts == 0).sum()
            if empty_ptfs > 0:
                print(f"    WARNING: {empty_ptfs} empty portfolios on average!")
        except Exception as e:
            print(f"    Could not get bond counts: {e}")

        # Check signal char distribution
        print(f"    Signal char means by portfolio: {signal_chars.mean().values}")

        return True, None

    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return False, str(e)


def test_doublesort(data, scenario_name):
    """Test DoubleSort behavior."""
    print(f"\n  DoubleSort (conditional, 2x5):")

    # Add a second sort variable with good variation
    data = data.copy()
    np.random.seed(123)
    data['signal2'] = np.random.uniform(1, 10, size=len(data))

    try:
        strategy = DoubleSort(
            holding_period=1,
            sort_var='signal',  # Low variation signal
            sort_var2='signal2',  # Good variation signal
            num_portfolios=5,
            num_portfolios2=2,
            how='conditional'
        )

        sf = StrategyFormation(
            data=data,
            strategy=strategy,
            turnover=False,
            verbose=False
        )

        result = sf.fit()

        ew_ls, vw_ls = result.get_long_short()

        print(f"    LS returns computed: {len(ew_ls)} dates")
        print(f"    Mean LS return: {ew_ls.mean():.6f}")

        # Get bond counts
        try:
            counts = result.get_bond_count()
            avg_counts = counts.mean()
            print(f"    Avg bonds per portfolio: {avg_counts.values[:5]}... (showing first 5)")

            # Check for empty portfolios
            empty_ptfs = (avg_counts == 0).sum()
            if empty_ptfs > 0:
                print(f"    WARNING: {empty_ptfs} empty portfolios on average!")
        except Exception as e:
            print(f"    Could not get bond counts: {e}")

        return True, None

    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return False, str(e)


def test_withinfirmsort(data, scenario_name):
    """Test WithinFirmSort behavior."""
    print(f"\n  WithinFirmSort (HIGH/LOW):")

    try:
        strategy = WithinFirmSort(
            holding_period=1,
            sort_var='signal',
            firm_id_col='PERMNO',
            min_bonds_per_firm=2
        )

        sf = StrategyFormation(
            data=data,
            strategy=strategy,
            turnover=False,
            verbose=False
        )

        result = sf.fit()

        ew_ls, vw_ls = result.get_long_short()

        # Count non-NaN returns
        valid_returns = ew_ls.notna().sum()

        print(f"    LS returns computed: {valid_returns}/{len(ew_ls)} dates with valid returns")

        if valid_returns > 0:
            print(f"    Mean LS return: {ew_ls.mean():.6f}")
        else:
            print(f"    WARNING: All returns are NaN!")

        return True, None

    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return False, str(e)


def run_scenario(scenario_name, data_generator, base_data):
    """Run all tests for a scenario."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print('='*60)

    data = data_generator(base_data)

    # Show signal distribution
    signal = data.groupby('date')['signal'].agg(['nunique', 'std', 'min', 'max'])
    print(f"\nSignal stats per date:")
    print(f"  Unique values: {signal['nunique'].mean():.1f} (avg)")
    print(f"  Std dev: {signal['std'].mean():.6f} (avg)")
    print(f"  Range: [{signal['min'].mean():.3f}, {signal['max'].mean():.3f}] (avg)")

    results = {}

    # Test each sort type
    results['SingleSort'] = test_singlesort(data, scenario_name)
    results['DoubleSort'] = test_doublesort(data, scenario_name)
    results['WithinFirmSort'] = test_withinfirmsort(data, scenario_name)

    return results


def main():
    print("="*60)
    print("DIAGNOSTIC: Low Cross-Sectional Variation Signals")
    print("="*60)

    # Generate base data
    base_data = generate_base_data(n_dates=12, n_bonds=100)
    print(f"\nBase data: {len(base_data)} rows, {base_data['date'].nunique()} dates, "
          f"{base_data['ID'].nunique()} bonds, {base_data['PERMNO'].nunique()} firms")

    all_results = {}

    # Run scenarios
    scenarios = [
        ("1. Constant signal (all=5.0)", lambda d: add_constant_signal(d, 5.0)),
        ("2. Binary signal (0 or 1)", add_binary_signal),
        ("3. Low cardinality (3 unique values)", lambda d: add_low_cardinality_signal(d, 3)),
        ("4. Low cardinality (4 unique values)", lambda d: add_low_cardinality_signal(d, 4)),
        ("5. Clustered (3 clusters, noise=0.001)", lambda d: add_clustered_signal(d, 3, 0.001)),
        ("6. Clustered (3 clusters, noise=0.1)", lambda d: add_clustered_signal(d, 3, 0.1)),
        ("7. Mostly constant (95% same)", lambda d: add_mostly_constant_signal(d, 0.95)),
        ("8. Mostly constant (80% same)", lambda d: add_mostly_constant_signal(d, 0.80)),
    ]

    for name, generator in scenarios:
        all_results[name] = run_scenario(name, generator, base_data)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for scenario, results in all_results.items():
        print(f"\n{scenario}:")
        for sort_type, (success, error) in results.items():
            status = "OK" if success else f"FAILED: {error[:50]}..."
            print(f"  {sort_type}: {status}")


if __name__ == '__main__':
    main()
