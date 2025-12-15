# -*- coding: utf-8 -*-
"""
PyBondLab Test Script for Spyder
================================

Run this script directly in Spyder to test portfolio formation with various
configurations. It generates synthetic data and compares fast vs slow paths.

Created: 2024
"""

import time
import numpy as np
import pandas as pd
import warnings
import PyBondLab as pbl

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Set random seed for reproducibility
RANDOM_SEED = 42

# Data generation parameters
N_DATES = 192       # Number of monthly periods (16 years)
N_BONDS = 2500      # Total number of unique bonds
MIN_BONDS = 500     # Minimum active bonds per period
MAX_BONDS = 2000    # Maximum active bonds per period

# Test tolerance (for comparing fast vs slow paths)
TOLERANCE = 1e-6

# =============================================================================
# Data Generation
# =============================================================================

def generate_test_data(n_dates=N_DATES, n_bonds=N_BONDS, seed=RANDOM_SEED):
    """
    Generate synthetic bond data for testing.

    Parameters
    ----------
    n_dates : int
        Number of monthly periods
    n_bonds : int
        Total number of unique bonds
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic bond panel data
    """
    np.random.seed(seed)

    dates = pd.date_range('2010-01-31', periods=n_dates, freq='ME')
    rows = []

    for i, date in enumerate(dates):
        # Vary number of active bonds over time
        n_active = max(MIN_BONDS, min(MAX_BONDS, 1000 + int(500 * np.sin(i / 12))))
        active_bonds = np.random.choice(n_bonds, n_active, replace=False)

        for bond_id in active_bonds:
            rows.append({
                'date': date,
                'ID': f'BOND_{bond_id:04d}',
                'ret': np.random.randn() * 0.02 + 0.003,  # ~0.3% mean, 2% std
                'RATING_NUM': np.random.randint(1, 22),
                'VW': np.random.uniform(0.1, 10),
                'PRICE': np.random.uniform(80, 120),
                'signal1': np.random.randn(),
                'signal2': np.random.randn(),
                'char1': np.random.uniform(0, 100),  # Characteristic 1
                'char2': np.random.uniform(-50, 50),  # Characteristic 2
            })

    return pd.DataFrame(rows)
