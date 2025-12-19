#!/usr/bin/env python
"""
Debug script to verify char_lookup is built correctly.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from PyBondLab.pbl_test import generate_synthetic_data_fast


def main():
    # Generate the exact same data as debug_chars_phase15b.py
    data = generate_synthetic_data_fast(
        n_dates=24,
        n_bonds=50,
        seed=42,
        balanced_panel=True,
        n_chars=2,
    )

    print(f"Data shape: {data.shape}")
    print(f"Data sorted by: {data.columns.tolist()}")

    # Check data ordering
    print("\nFirst 10 rows of data:")
    print(data[['date', 'ID', 'char1']].head(10).to_string())

    # Create date mapping (same as _fit_nonstaggered_fast)
    dates = sorted(data['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    date_idx = data['date'].map(date_to_idx).values.astype(np.int64)

    # Create ID mapping (same as _fit_nonstaggered_fast)
    all_ids = data['ID'].unique()  # Note: NOT sorted
    id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}
    id_idx = data['ID'].map(id_to_idx).values.astype(np.int64)

    n_dates = len(dates)
    n_ids = len(all_ids)
    n_chars = 1

    print(f"\nNumber of dates: {n_dates}")
    print(f"Number of IDs: {n_ids}")

    # Build char_lookup (same as kernel)
    char_values = data['char1'].values.astype(np.float64)
    char_lookup = np.full((n_dates, n_ids), np.nan, dtype=np.float64)

    for i in range(len(data)):
        d = date_idx[i]
        bond = id_idx[i]
        char_lookup[d, bond] = char_values[i]

    # Verify: What's in char_lookup[5, 0] (June, first bond)?
    june_idx = 5
    first_bond_idx = 0
    first_bond_id = [k for k, v in id_to_idx.items() if v == 0][0]

    print(f"\nFirst bond ID: {first_bond_id}")
    print(f"char_lookup[5 (June), 0 (first bond)] = {char_lookup[5, 0]}")
    print(f"char_lookup[6 (July), 0 (first bond)] = {char_lookup[6, 0]}")

    # Compare with actual data
    june_date = dates[5]
    july_date = dates[6]

    char_at_june = data[(data['date'] == june_date) & (data['ID'] == first_bond_id)]['char1'].values
    char_at_july = data[(data['date'] == july_date) & (data['ID'] == first_bond_id)]['char1'].values

    print(f"\nActual char1 at June for {first_bond_id}: {char_at_june}")
    print(f"Actual char1 at July for {first_bond_id}: {char_at_july}")

    # Check if they match
    if len(char_at_june) > 0 and abs(char_lookup[5, 0] - char_at_june[0]) < 1e-10:
        print("\n✓ char_lookup[5, 0] matches June's data")
    else:
        print("\n✗ char_lookup[5, 0] DOES NOT match June's data!")

    if len(char_at_july) > 0 and abs(char_lookup[6, 0] - char_at_july[0]) < 1e-10:
        print("✓ char_lookup[6, 0] matches July's data")
    else:
        print("✗ char_lookup[6, 0] DOES NOT match July's data!")

    # Now check: what if we use d-1 in the loop?
    print("\n" + "=" * 50)
    print("Simulating kernel char lookup...")
    print("=" * 50)

    # Simulate d=6 (July return date)
    d = 6
    char_date = d - 1  # = 5 (June)

    print(f"\nFor return date d={d} (July):")
    print(f"  char_date = d-1 = {char_date} (June)")
    print(f"  Lookup: char_lookup[{char_date}, 0] = {char_lookup[char_date, 0]}")
    print(f"  This should be June's char = {char_at_june[0] if len(char_at_june) > 0 else 'N/A'}")

    # Simulate d=7 (August return date)
    d = 7
    char_date = d - 1  # = 6 (July)
    aug_date = dates[7]
    char_at_aug = data[(data['date'] == aug_date) & (data['ID'] == first_bond_id)]['char1'].values

    print(f"\nFor return date d={d} (August):")
    print(f"  char_date = d-1 = {char_date} (July)")
    print(f"  Lookup: char_lookup[{char_date}, 0] = {char_lookup[char_date, 0]}")
    print(f"  This should be July's char = {char_at_july[0] if len(char_at_july) > 0 else 'N/A'}")


if __name__ == '__main__':
    main()
