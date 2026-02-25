"""
PyBondLab Test Configuration (public, synthetic-data-only).

Fixtures use synthetic data generators from pbl_test.py.
No real data, no manifest, no parquet loading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyBondLab.pbl_test import (
    generate_synthetic_data,
    generate_synthetic_data_fast,
    RANDOM_SEED,
    N_DATES,
    N_BONDS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def synthetic_data():
    """Generate synthetic data matching main baseline generation (slow generator)."""
    return generate_synthetic_data(
        n_dates=N_DATES,
        n_bonds=N_BONDS,
        seed=RANDOM_SEED,
    )


@pytest.fixture(scope="session")
def synthetic_data_fast():
    """Generate synthetic data matching DUA/singlesort baseline generation (fast generator)."""
    return generate_synthetic_data_fast(
        n_dates=N_DATES,
        n_bonds=N_BONDS,
        seed=RANDOM_SEED,
    )


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "consistency: marks consistency tests between functionalities"
    )
    config.addinivalue_line(
        "markers", "contract: marks tests that extract and save code contracts"
    )
    config.addinivalue_line(
        "markers", "smoke: marks quick smoke tests for basic functionality"
    )
