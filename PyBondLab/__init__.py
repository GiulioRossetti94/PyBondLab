# Core strategy formation
try:
    from .PyBondLab import StrategyFormation, load_breakpoints_WRDS
except ImportError as e:
    print(f"Warning: Could not import StrategyFormation: {e}")
    StrategyFormation = None
    load_breakpoints_WRDS = None

# Batch processing
try:
    from .batch import BatchStrategyFormation, BatchResults, batch_single_sort
except ImportError as e:
    print(f"Warning: Could not import batch processing: {e}")
    BatchStrategyFormation = BatchResults = batch_single_sort = None

# Batch WithinFirmSort processing
try:
    from .batch_withinfirm import BatchWithinFirmSortFormation
except ImportError as e:
    print(f"Warning: Could not import BatchWithinFirmSortFormation: {e}")
    BatchWithinFirmSortFormation = None

# Strategy classes
try:
    from .StrategyClass import SingleSort, DoubleSort, Momentum, LTreversal, WithinFirmSort
except ImportError as e:
    print(f"Warning: Could not import Strategy classes: {e}")
    SingleSort = DoubleSort = Momentum = LTreversal = WithinFirmSort = None


# Anomaly assayer
try:
    from .AnomalyAssayer import AssayAnomaly
except ImportError:
    AssayAnomaly = None

# Constants and config
try:
    from .constants import RatingBounds, Defaults, ColumnNames
except ImportError:
    RatingBounds = Defaults = ColumnNames = None

try:
    from .config import StrategyFormationConfig
except ImportError:
    StrategyFormationConfig = None

# Precompute
try:
    from .precompute import build_precomputed_data
except ImportError:
    build_precomputed_data = None

# Rolling Beta estimation
try:
    from .rolling_beta import RollingBeta
except ImportError as e:
    print(f"Warning: Could not import RollingBeta: {e}")
    RollingBeta = None

# Describe module (summary statistics)
try:
    from .describe import PreAnalysisStats, PreAnalysisResult
except ImportError:
    PreAnalysisStats = PreAnalysisResult = None

# Panel validation utilities
try:
    from .utils import validate_panel, check_duplicates
except ImportError:
    validate_panel = check_duplicates = None

# Data Uncertainty Analysis
try:
    from .data_uncertainty import DataUncertaintyAnalysis, DataUncertaintyResults
except ImportError as e:
    print(f"Warning: Could not import DataUncertaintyAnalysis: {e}")
    DataUncertaintyAnalysis = DataUncertaintyResults = None

# Naming configuration
try:
    from .naming import NamingConfig
except ImportError as e:
    print(f"Warning: Could not import NamingConfig: {e}")
    NamingConfig = None

# Panel extraction
try:
    from .extract import extract_panel
except ImportError as e:
    print(f"Warning: Could not import extract_panel: {e}")
    extract_panel = None

# Specification validator
try:
    from .spec_validator import (
        validate_specs,
        SpecificationValidator,
        ValidationResult,
        generate_spec_list,
        get_valid_spec_list,
        filter_spec_list,
    )
except ImportError as e:
    print(f"Warning: Could not import spec_validator: {e}")
    validate_specs = SpecificationValidator = ValidationResult = None
    generate_spec_list = get_valid_spec_list = filter_spec_list = None

# Fast anomaly assayer (numba-optimized)
try:
    from .anomaly_assay_fast import assay_anomaly_fast, AnomalyAssayResult
except ImportError as e:
    print(f"Warning: Could not import anomaly_assay_fast: {e}")
    assay_anomaly_fast = AnomalyAssayResult = None

# Batch anomaly assayer
try:
    from .batch_assay import BatchAssayAnomaly, BatchAssayResults, batch_assay_anomaly
except ImportError as e:
    print(f"Warning: Could not import batch_assay: {e}")
    BatchAssayAnomaly = BatchAssayResults = batch_assay_anomaly = None

__version__ = '0.2.0'
__all__ = [
    'StrategyFormation',
    'load_breakpoints_WRDS',
    'BatchStrategyFormation',
    'BatchWithinFirmSortFormation',
    'BatchResults',
    'batch_single_sort',
    'SingleSort',
    'DoubleSort',
    'Momentum',
    'LTreversal',
    'WithinFirmSort',
    'AssayAnomaly',
    'RollingBeta',
    'PreAnalysisStats',
    'PreAnalysisResult',
    'validate_panel',
    'check_duplicates',
    'DataUncertaintyAnalysis',
    'DataUncertaintyResults',
    'NamingConfig',
    'extract_panel',
    'validate_specs',
    'SpecificationValidator',
    'ValidationResult',
    'generate_spec_list',
    'get_valid_spec_list',
    'filter_spec_list',
    'assay_anomaly_fast',
    'AnomalyAssayResult',
    'BatchAssayAnomaly',
    'BatchAssayResults',
    'batch_assay_anomaly',
]
name = 'PyBondLab'
