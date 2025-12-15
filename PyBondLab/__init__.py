# Core strategy formation
try:
    from .PyBondLab import StrategyFormation, load_breakpoints_WRDS
except ImportError as e:
    print(f"Warning: Could not import StrategyFormation: {e}")
    StrategyFormation = None
    load_breakpoints_WRDS = None

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

__version__ = '0.2.0'
__all__ = [
    'StrategyFormation',
    'load_breakpoints_WRDS',
    'SingleSort',
    'DoubleSort',
    'Momentum',
    'LTreversal',
    'WithinFirmSort',
    'CreateDailyEnhancedTRACE',
    'AssayAnomaly',
    'RollingBeta',
    'PreAnalysisStats',
    'PreAnalysisResult',
    'validate_panel',
    'check_duplicates',
]
name = 'PyBondLab'
