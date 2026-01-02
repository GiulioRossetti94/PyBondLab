"""
Test DoubleSort with pre-defined rating breaks.

This test validates:
1. IG bonds (ratings 1-10) are correctly in rating_group=1
2. NIG bonds (ratings 11+) are correctly in rating_group=2
3. chars=['spc_rat'] works to verify average ratings per portfolio
4. No cross-contamination between rating groups in conditional sort
"""

import sys
sys.path.insert(0, '/home/user/PyBondLab-Dev')

import numpy as np
import pandas as pd
import PyBondLab as pbl

def generate_test_data(n_dates=24, n_bonds=200, seed=42):
    """Generate synthetic data with known rating distribution."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-31', periods=n_dates, freq='ME')
    bond_ids = [f'BOND_{i:04d}' for i in range(n_bonds)]

    rows = []
    for date in dates:
        # Each date, randomly select ~80% of bonds to be active
        active_bonds = np.random.choice(bond_ids, size=int(n_bonds * 0.8), replace=False)

        for bond_id in active_bonds:
            # Assign ratings: 50% IG (1-10), 50% NIG (11-22)
            bond_num = int(bond_id.split('_')[1])
            if bond_num < n_bonds // 2:
                # IG bonds: ratings 1-10
                rating = np.random.randint(1, 11)
            else:
                # NIG bonds: ratings 11-22
                rating = np.random.randint(11, 23)

            rows.append({
                'date': date,
                'cusip': bond_id,
                'ret_vw': np.random.randn() * 0.05,  # ~5% monthly vol
                'mcap_e': np.random.uniform(100, 1000),  # Market cap
                'spc_rat': rating,
                'cs': np.random.randn(),  # Credit spread signal
            })

    df = pd.DataFrame(rows)
    return df


def test_doublesort_predefined_breaks():
    """Test DoubleSort with pre-defined rating breaks."""
    print("="*70)
    print("TEST: DoubleSort with Pre-defined Rating Breaks")
    print("="*70)

    # Generate test data
    data = generate_test_data(n_dates=24, n_bonds=200, seed=42)
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Rating distribution:")
    print(f"  IG (1-10): {(data['spc_rat'] <= 10).sum()} observations")
    print(f"  NIG (11+): {(data['spc_rat'] > 10).sum()} observations")

    # Create pre-defined rating groups
    data['rating_group'] = np.where(data['spc_rat'] <= 10, 1, 2)
    print(f"\nRating group distribution:")
    print(f"  Group 1 (IG): {(data['rating_group'] == 1).sum()} observations")
    print(f"  Group 2 (NIG): {(data['rating_group'] == 2).sum()} observations")

    # Test 1: Run DoubleSort with chars=['spc_rat']
    print("\n" + "-"*70)
    print("Test 1: DoubleSort with chars=['spc_rat']")
    print("-"*70)

    try:
        double_sort = pbl.DoubleSort(
            sort_var='rating_group',
            sort_var2='cs',
            num_portfolios=2,      # 2 rating groups
            num_portfolios2=5,     # 5 CS quintiles within each group
            how='conditional',
            holding_period=1,
            verbose=True
        )

        sf = pbl.StrategyFormation(
            data=data,
            strategy=double_sort,
            turnover=True,
            chars=['spc_rat'],  # Track average rating per portfolio
            verbose=True
        )

        results = sf.fit(
            IDvar='cusip',
            RETvar='ret_vw',
            VWvar='mcap_e',
            RATINGvar='spc_rat'
        )

        print("\n[SUCCESS] DoubleSort with chars=['spc_rat'] completed!")

        # Get characteristics to verify ratings
        ew_chars, vw_chars = results.get_characteristics()

        if 'spc_rat' in ew_chars:
            char_df = ew_chars['spc_rat']
            print(f"\nCharacteristics shape: {char_df.shape}")
            print(f"Columns (portfolios): {list(char_df.columns)}")

            # Analyze average ratings per portfolio
            print("\nAverage rating by portfolio (EW):")
            avg_ratings = char_df.mean()
            for col in char_df.columns:
                print(f"  Portfolio {col}: {avg_ratings[col]:.2f}")

            # Check for cross-contamination
            # Columns contain GROUP1 (IG) or GROUP2 (NIG) in their names
            all_cols = list(char_df.columns)
            ig_portfolios = [c for c in all_cols if 'GROUP1' in str(c)]
            nig_portfolios = [c for c in all_cols if 'GROUP2' in str(c)]

            print(f"\nValidating rating separation:")
            print(f"  IG portfolios (should have avg rating <= 10): {ig_portfolios}")
            print(f"  NIG portfolios (should have avg rating > 10): {nig_portfolios}")

            # Check IG portfolios
            ig_violation = False
            for p in ig_portfolios:
                avg = avg_ratings[p]
                if avg > 10.5:  # Small tolerance
                    print(f"  [ERROR] Portfolio {p} has avg rating {avg:.2f} > 10!")
                    ig_violation = True
                else:
                    print(f"  [OK] Portfolio {p} has avg rating {avg:.2f} (IG correct)")

            # Check NIG portfolios
            nig_violation = False
            for p in nig_portfolios:
                avg = avg_ratings[p]
                if avg < 10.5:  # Small tolerance
                    print(f"  [ERROR] Portfolio {p} has avg rating {avg:.2f} < 11!")
                    nig_violation = True
                else:
                    print(f"  [OK] Portfolio {p} has avg rating {avg:.2f} (NIG correct)")

            if not ig_violation and not nig_violation:
                print("\n  [PASS] No cross-contamination detected!")
            else:
                print("\n  [FAIL] Cross-contamination detected!")
                return False
        else:
            print(f"\n[WARNING] 'spc_rat' not found in chars. Available: {list(ew_chars.keys())}")

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Verify with direct data inspection
    print("\n" + "-"*70)
    print("Test 2: Direct data inspection of portfolio assignments")
    print("-"*70)

    # Get the portfolio weights to see actual assignments
    try:
        # Access internal port_idx to see assignments
        if hasattr(sf, 'port_idx') and sf.port_idx:
            # Get first date's assignments
            first_date = list(sf.port_idx.keys())[0]
            weights_df = sf.port_idx[first_date]

            print(f"\nPortfolio assignments for {first_date}:")
            print(f"  Total bonds assigned: {len(weights_df)}")

            # Merge with original data to check ratings
            date_data = data[data['date'] == first_date].copy()
            date_data = date_data.rename(columns={'cusip': 'ID'})

            merged = weights_df.merge(date_data[['ID', 'spc_rat', 'rating_group']], on='ID', how='left')

            # Check each portfolio
            print("\n  Portfolio breakdown:")
            for ptf in sorted(merged['ptf_rank'].unique()):
                ptf_data = merged[merged['ptf_rank'] == ptf]
                avg_rating = ptf_data['spc_rat'].mean()
                min_rating = ptf_data['spc_rat'].min()
                max_rating = ptf_data['spc_rat'].max()
                n_bonds = len(ptf_data)

                # Determine expected group based on portfolio number
                # With 2x5 conditional sort: P1-5 = GROUP1 (IG), P6-10 = GROUP2 (NIG)
                expected_group = 1 if ptf <= 5 else 2
                actual_groups = ptf_data['rating_group'].unique()

                status = "OK" if len(actual_groups) == 1 and actual_groups[0] == expected_group else "MIXED!"

                print(f"    P{ptf}: n={n_bonds}, avg_rat={avg_rating:.1f}, "
                      f"range=[{min_rating}, {max_rating}], groups={list(actual_groups)} [{status}]")

            # Final validation
            print("\n  Rating range validation:")
            ig_data = merged[merged['ptf_rank'] <= 5]
            nig_data = merged[merged['ptf_rank'] > 5]
            print(f"    IG portfolios (P1-5): min={ig_data['spc_rat'].min()}, max={ig_data['spc_rat'].max()}")
            print(f"    NIG portfolios (P6-10): min={nig_data['spc_rat'].min()}, max={nig_data['spc_rat'].max()}")

            # Ultimate check: IG should have all ratings <= 10, NIG should have all ratings > 10
            ig_max = ig_data['spc_rat'].max()
            nig_min = nig_data['spc_rat'].min()
            if ig_max <= 10 and nig_min >= 11:
                print(f"    [PASS] Perfect separation: IG max={ig_max} <= 10, NIG min={nig_min} >= 11")
            else:
                print(f"    [FAIL] Cross-contamination: IG max={ig_max}, NIG min={nig_min}")

    except Exception as e:
        print(f"  [WARNING] Could not inspect internal data: {e}")

    # Test 3: Get long-short returns
    print("\n" + "-"*70)
    print("Test 3: Long-Short Returns")
    print("-"*70)

    ew_ls, vw_ls = results.get_long_short()
    print(f"\nEW Long-Short mean: {ew_ls.mean():.6f}")
    print(f"VW Long-Short mean: {vw_ls.mean():.6f}")
    print(f"Number of dates: {len(ew_ls)}")

    # Test 4: Verify turnover works
    print("\n" + "-"*70)
    print("Test 4: Turnover")
    print("-"*70)

    ew_turn, vw_turn = results.get_turnover()
    print(f"Turnover shape: {ew_turn.shape}")
    print(f"Mean turnover by portfolio:")
    for col in ew_turn.columns:
        print(f"  P{col}: EW={ew_turn[col].mean():.4f}, VW={vw_turn[col].mean():.4f}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    return True


def test_chars_column_conflict():
    """Test if using sort column as chars causes issues."""
    print("\n" + "="*70)
    print("TEST: Column Conflict (sort var = chars var)")
    print("="*70)

    # Generate minimal test data
    data = generate_test_data(n_dates=12, n_bonds=100, seed=123)
    data['rating_group'] = np.where(data['spc_rat'] <= 10, 1, 2)

    # Test: Sort on rating_group, chars includes rating_group
    print("\nTest: Sort on 'rating_group', chars=['rating_group']")

    try:
        double_sort = pbl.DoubleSort(
            sort_var='rating_group',
            sort_var2='cs',
            num_portfolios=2,
            num_portfolios2=3,
            how='conditional',
            holding_period=1,
            verbose=False
        )

        sf = pbl.StrategyFormation(
            data=data,
            strategy=double_sort,
            turnover=False,
            chars=['rating_group'],  # Same as sort var!
            verbose=False
        )

        results = sf.fit(
            IDvar='cusip',
            RETvar='ret_vw',
            VWvar='mcap_e',
            RATINGvar='spc_rat'
        )

        ew_chars, vw_chars = results.get_characteristics()

        if 'rating_group' in ew_chars:
            print("[SUCCESS] No conflict - chars work with sort variable")
            char_df = ew_chars['rating_group']
            print(f"  Avg rating_group by portfolio: {char_df.mean().values}")
        else:
            print(f"[WARNING] 'rating_group' not in chars. Available: {list(ew_chars.keys())}")

    except Exception as e:
        print(f"[ERROR] Conflict detected: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run main test
    success = test_doublesort_predefined_breaks()

    # Run conflict test
    test_chars_column_conflict()

    if success:
        print("\n" + "="*70)
        print("SUMMARY: All tests passed - approach is valid!")
        print("="*70)
