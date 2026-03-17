> **DEPRECATED:** This file is a legacy artifact and may not reflect the current API.
> For up-to-date documentation, see [README.md](README.md) and the files in [docs/](docs/).

# PyBondLab_v2 Package Summary

## Package Structure

```
PyBondLab_v2/
├── LICENSE                         # MIT License
├── MANIFEST.in                     # Package manifest for distribution
├── README.md                       # Package documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup configuration
└── PyBondLab/                      # Main package directory
    ├── __init__.py                 # Package initialization
    ├── PyBondLab.py               # Main module with StrategyFormation class
    ├── AnomalyAssayer.py          # Enhanced anomaly analysis
    ├── StrategyClass.py           # Strategy classes (SingleSort, DoubleSort, etc.)
    ├── FilterClass.py             # Filter utilities
    ├── config.py                  # Configuration classes
    ├── constants.py               # Package constants
    ├── precompute.py              # Precomputation utilities
    ├── results.py                 # Results handling
    ├── anomaly_correlation.py     # Correlation analysis
    ├── utils_portfolio.py         # Portfolio utilities
    ├── utils_turnover.py          # Turnover analysis
    ├── data/                      # Data loading utilities
    │   ├── __init__.py
    │   ├── data_loading.py
    │   └── WRDS/
    │       ├── __init__.py
    │       └── breakpoints_wrds.csv
    ├── iotools/                   # I/O utilities
    │   ├── __init__.py
    │   ├── PyBondLabResults.py
    │   └── table.py
    ├── optm/                      # Optimization helpers (reserved for future use)
    │   └── __init__.py
    └── visualization/             # Plotting and visualization
        ├── __init__.py
        ├── _latex.py
        └── plotting.py
```

## Installation

### Development Mode (Recommended)

```bash
cd PyBondLab_v2
pip install -e .
```

This installs the package in "editable" mode, meaning changes to the source code will immediately affect the installed package without needing to reinstall.

### Regular Installation

```bash
cd PyBondLab_v2
pip install .
```

## Package Details

- **Version**: 0.2.0
- **Python Requirements**: >= 3.11
- **Total Python Files**: 27
- **Subpackages**: 7 (PyBondLab, PyBondLab.data, PyBondLab.data.WRDS, PyBondLab.dataset_builder, PyBondLab.iotools, PyBondLab.optm, PyBondLab.visualization)

## Dependencies

- numpy < 2
- pandas >= 1.5
- statsmodels >= 0.14
- matplotlib >= 3.5
- pyarrow
- wrds

## Key Features

1. **StrategyFormation**: Main class for portfolio sorting and strategy evaluation
2. **AssayAnomaly**: Enhanced anomaly analysis with correlation visualization
3. **Strategy Classes**: SingleSort, DoubleSort, Momentum, LTreversal
4. **Data Management**: WRDS data loading and dataset building utilities
5. **Visualization**: Plotting and LaTeX table generation
6. **I/O Tools**: Results handling and export functionality

## Verification

The package has been verified to:
- ✅ Have correct directory structure
- ✅ Include all necessary Python modules (27 files)
- ✅ Have proper __init__.py files in all packages
- ✅ Include setup.py with correct configuration
- ✅ Include requirements.txt with all dependencies
- ✅ Include README.md with documentation
- ✅ Include LICENSE file (MIT)
- ✅ Be installable via pip (tested with --dry-run)
- ✅ Discover all 7 subpackages correctly

## Usage Example

```python
from PyBondLab import StrategyFormation, AssayAnomaly
from PyBondLab.config import StrategyFormationConfig

# Create configuration
config = StrategyFormationConfig()

# Run strategy formation
strategy = StrategyFormation(data, config)

# Assay anomalies
assayer = AssayAnomaly(data)
results = assayer.run()
```

## Next Steps

1. Navigate to the PyBondLab_v2 directory
2. Run `pip install -e .` to install in development mode
3. Test the package by importing and using key functions
4. Verify all functionality works as expected

## Notes

- The `optm/` folder is reserved for future optimization helpers
- All subpackages have proper __init__.py files for correct imports
- The package includes WRDS breakpoints data file
- Version number has been updated to 0.2.0 to reflect enhancements
