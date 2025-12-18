"""Debug script for combined filter failure with HP=3."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from PyBondLab import BatchStrategyFormation, StrategyFormation, SingleSort
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig
from PyBondLab.pbl_test import generate_synthetic_data


def main():
    print("Generating data...")
    data = generate_synthetic_data(n_dates=60, n_bonds=500, seed=42)

    signal = 'signal1'
    hp = 3
    rating = 'IG'
    subset_filter = {'char1': (-0.5, 0.5)}

    print(f"\n{'='*60}")
    print(f"Testing: HP={hp}, rating={rating}, subset_filter={subset_filter}")
    print(f"{'='*60}")

    # Count filtered bonds
    rating_mask = data['RATING_NUM'] <= 10
    char_mask = (data['char1'] >= -0.5) & (data['char1'] <= 0.5)
    combined_mask = rating_mask & char_mask

    n_ig = rating_mask.sum()
    n_char = char_mask.sum()
    n_combined = combined_mask.sum()
    n_total = len(data)

    print(f"\nData: {n_total} obs")
    print(f"  IG only: {n_ig} ({100*n_ig/n_total:.1f}%)")
    print(f"  char1 filter only: {n_char} ({100*n_char/n_total:.1f}%)")
    print(f"  Combined: {n_combined} ({100*n_combined/n_total:.1f}%)")

    # ===== FAST PATH =====
    print("\n--- Fast Batch Path ---")
    batch = BatchStrategyFormation(
        data=data,
        signals=[signal],
        holding_period=hp,
        num_portfolios=5,
        turnover=False,
        rating=rating,
        subset_filter=subset_filter,
        verbose=True
    )
    batch_results = batch.fit()
    fast_ew, fast_vw = batch_results[signal].get_long_short()

    print(f"\nFast EW mean: {fast_ew.mean():.6f}")
    print(f"Fast EW first 5:\n{fast_ew.head()}")

    # ===== SLOW PATH =====
    print("\n--- Slow SingleSort Path ---")
    strategy = SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=5,
        verbose=False
    )

    sf_config = StrategyFormationConfig(
        data=DataConfig(rating=rating, subset_filter=subset_filter),
        formation=FormationConfig(
            dynamic_weights=True,
            compute_turnover=False,
            verbose=True,
        )
    )

    sf = StrategyFormation(data=data, strategy=strategy, config=sf_config)
    result = sf.fit()
    slow_ew, slow_vw = result.get_long_short()

    print(f"\nSlow EW mean: {slow_ew.mean():.6f}")
    print(f"Slow EW first 5:\n{slow_ew.head()}")

    # ===== COMPARISON =====
    print("\n--- Comparison ---")
    common_idx = fast_ew.index.intersection(slow_ew.index)

    fast_ew_aligned = fast_ew.loc[common_idx]
    slow_ew_aligned = slow_ew.loc[common_idx]

    diff = fast_ew_aligned - slow_ew_aligned
    print(f"EW diff max: {diff.abs().max():.6e}")
    print(f"\nDiff values (non-zero):")
    non_zero = diff[diff != 0]
    print(non_zero)

    # Check if there are NaN mismatches
    fast_nan = fast_ew_aligned.isna().sum()
    slow_nan = slow_ew_aligned.isna().sum()
    print(f"\nNaN counts: fast={fast_nan}, slow={slow_nan}")


if __name__ == '__main__':
    main()
