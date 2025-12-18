"""Debug script to investigate fast vs slow path differences with rating filter."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from PyBondLab import BatchStrategyFormation, StrategyFormation, SingleSort
from PyBondLab.config import StrategyFormationConfig, FormationConfig, DataConfig
from PyBondLab.pbl_test import generate_synthetic_data


def main():
    # Generate small dataset for debugging
    print("Generating data...")
    data = generate_synthetic_data(n_dates=20, n_bonds=100, seed=42)

    signal = 'signal1'
    hp = 1
    rating = 'IG'

    print(f"\n{'='*60}")
    print(f"Testing signal={signal}, HP={hp}, rating={rating}")
    print(f"{'='*60}")

    # Count how many IG bonds
    n_ig = (data['RATING_NUM'] <= 10).sum()
    n_total = len(data)
    print(f"\nData: {n_total} obs, {n_ig} IG ({100*n_ig/n_total:.1f}%)")

    # ===== FAST PATH =====
    print("\n--- Fast Batch Path ---")
    batch = BatchStrategyFormation(
        data=data,
        signals=[signal],
        holding_period=hp,
        num_portfolios=5,
        turnover=False,
        rating=rating,
        verbose=True
    )
    batch_results = batch.fit()
    fast_ew, fast_vw = batch_results[signal].get_long_short()

    print(f"\nFast EW mean: {fast_ew.mean():.6f}")
    print(f"Fast VW mean: {fast_vw.mean():.6f}")
    print(f"Fast EW first 5:\n{fast_ew.head()}")

    # ===== SLOW PATH =====
    print("\n--- Slow SingleSort Path ---")
    strategy = SingleSort(
        holding_period=hp,
        sort_var=signal,
        num_portfolios=5,
        verbose=True
    )

    sf_config = StrategyFormationConfig(
        data=DataConfig(rating=rating),
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
    print(f"Slow VW mean: {slow_vw.mean():.6f}")
    print(f"Slow EW first 5:\n{slow_ew.head()}")

    # ===== COMPARISON =====
    print("\n--- Comparison ---")

    # Align dates
    common_idx = fast_ew.index.intersection(slow_ew.index)
    print(f"Common dates: {len(common_idx)}")

    fast_ew_aligned = fast_ew.loc[common_idx]
    slow_ew_aligned = slow_ew.loc[common_idx]

    diff = fast_ew_aligned - slow_ew_aligned
    print(f"\nEW diff stats:")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Max:  {diff.abs().max():.6f}")
    print(f"  Diff by date:\n{diff}")

    # Check specific date
    if len(common_idx) > 2:
        test_date = common_idx[2]
        print(f"\n--- Detailed check for date {test_date} ---")

        # Get date indices
        date_mask = data['date'] == test_date
        prev_date = common_idx[1]
        prev_mask = data['date'] == prev_date

        # Formation date data
        form_data = data[prev_mask].copy()
        ig_form = form_data[form_data['RATING_NUM'] <= 10]

        print(f"Formation date {prev_date}: {len(form_data)} bonds, {len(ig_form)} IG")

        # Check signal distribution for IG bonds
        ig_signal = ig_form[signal].dropna()
        print(f"IG signal: min={ig_signal.min():.4f}, max={ig_signal.max():.4f}")

        # Check ranking
        thresholds = np.percentile(ig_signal, [0, 20, 40, 60, 80, 100])
        print(f"Percentile thresholds: {thresholds}")


if __name__ == '__main__':
    main()
