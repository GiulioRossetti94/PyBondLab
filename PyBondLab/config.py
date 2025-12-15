# -*- coding: utf-8 -*-
"""
Created: 03-11-2025
Last modified: 03-11-2025

@authors: Giulio Rossetti

Configuration classes for PyBondLab.

This module provides configuration dataclasses that encapsulate
related parameters and provide validation.

"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Callable, Tuple
import warnings

from .constants import (
    Defaults,
    FilterType,
    ValidationMessages,
    RatingBounds,
)

Number = Union[int, float]
SubsetFilter = Dict[str, Tuple[Number, Number]]

# =============================================================================
# Data Configuration
# =============================================================================
@dataclass
class DataConfig:
    """
    Configuration for data filtering and characteristics.
    
    This handles data filtering options for StrategyFormation that are
    independent of the Strategy class.
    
    Attributes
    ----------
    rating : str, tuple, or None, optional
        Rating filter:
        - 'IG': Investment grade (RATING_NUM 1-10)
        - 'NIG': Non-investment grade (RATING_NUM 11-22)
        - (min, max): Custom rating range
        - None: No rating filter
    subset_filter : dict, optional
        Characteristic-based filters: {column: (min, max)}
        Example: {'duration': (1, 10), 'size': (0, 1e9)}
    chars : list of str, optional
        Characteristics to compute for portfolios
        Example: ['duration', 'size', 'rating']
    
    Examples
    --------
    >>> # Filter to investment grade bonds
    >>> data_config = DataConfig(rating='IG')
    >>> 
    >>> # Filter by rating and duration
    >>> data_config = DataConfig(
    ...     rating='IG',
    ...     subset_filter={'duration': (1, 10)}
    ... )
    >>> 
    >>> # Compute characteristics
    >>> data_config = DataConfig(
    ...     rating='IG',
    ...     chars=['duration', 'size', 'rating']
    ... )
    """
    
    rating: Optional[Union[str, Tuple[float, float]]] = None
    subset_filter: Optional[SubsetFilter] = None
    chars: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_rating()
        if self.subset_filter is not None:
            self._validate_subset_filter()
    
    def _validate_rating(self):
        """Validate rating parameter."""
        if self.rating is None:
            return
        
        if isinstance(self.rating, str):
            if self.rating not in RatingBounds.VALID_CATEGORIES:
                raise ValueError(
                    ValidationMessages.INVALID_RATING.format(rating=self.rating)
                )
        elif isinstance(self.rating, (tuple, list)):
            if len(self.rating) != 2:
                raise ValueError("Rating tuple must have exactly 2 elements")
            if not all(isinstance(x, (int, float)) for x in self.rating):
                raise TypeError("Rating bounds must be numeric")
            low, high = self.rating
            if low > high:
                raise ValueError(f"Rating min ({low}) must be <= max ({high})")
            self.rating = tuple(self.rating)  # Ensure tuple
        else:
            raise TypeError(
                "Rating must be None, str ('IG'/'NIG'), or (min, max) tuple"
            )
    
    def _validate_subset_filter(self):
        """Validate subset filter."""
        if not isinstance(self.subset_filter, dict):
            raise TypeError("subset_filter must be a dictionary")
        
        for col, bounds in self.subset_filter.items():
            if not isinstance(col, str):
                raise TypeError(f"Column name must be str, got {type(col)}")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    f"Bounds for '{col}' must be a (min, max) tuple/list"
                )
            low, high = bounds
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise TypeError(f"Bounds for '{col}' must be numeric")
            if low > high:
                raise ValueError(
                    ValidationMessages.INVALID_BOUNDS.format(
                        col=col, low=low, high=high
                    )
                )
    
    @property
    def has_rating_filter(self) -> bool:
        """Check if rating filter is applied."""
        return self.rating is not None
    
    @property
    def has_subset_filter(self) -> bool:
        """Check if subset filter is applied."""
        return self.subset_filter is not None and len(self.subset_filter) > 0
    
    @property
    def has_chars(self) -> bool:
        """Check if characteristics are requested."""
        return self.chars is not None and len(self.chars) > 0
    

# =============================================================================
# Filter Configuration (Ex-Post Adjustments)
# =============================================================================
@dataclass
class FilterConfig:
    """
    Configuration for ex-post filters and adjustments.
    
    These filters modify returns after portfolio formation for robustness checks.
    
    Attributes
    ----------
    adj : str, optional
        Filter type:
        - 'trim': Trim extreme returns
        - 'wins': Winsorize returns
        - 'price': Filter by bond price
        - 'bounce': Filter by return bounce
        - None: No adjustment
    level : float or tuple, optional
        Filter level (threshold or bounds)
        - For trim/wins: single value or (lower, upper) bounds
        - For price: price threshold
        - For bounce: bounce threshold
    location : str, optional
        Filter location parameter (depends on filter type)
    df_breakpoints : pd.DataFrame, optional
        Breakpoints DataFrame for winsorization
    price_threshold : float, default 25
        Price threshold for price filters
    
    Examples
    --------
    >>> # Trim extreme returns
    >>> filter_config = FilterConfig(adj='trim', level=0.01)
    >>> 
    >>> # Winsorize returns
    >>> filter_config = FilterConfig(adj='wins', level=(0.01, 0.99))
    >>> 
    >>> # Filter by price
    >>> filter_config = FilterConfig(adj='price', level=50, price_threshold=25)
    """
    
    adj: Optional[str] = None
    level: Optional[Union[float, Tuple[float, float]]] = None
    location: Optional[str] = None
    df_breakpoints: Optional[object] = None  # pd.DataFrame
    price_threshold: float = Defaults.PRICE_THRESHOLD
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.adj is not None:
            self._validate_adj()
    
    def _validate_adj(self):
        """Validate adjustment type."""
        if not FilterType.is_valid(self.adj):
            raise ValueError(
                ValidationMessages.INVALID_FILTER.format(
                    adj=self.adj,
                    valid_options=FilterType.VALID_TYPES
                )
            )
    
    @property
    def has_adjustment(self) -> bool:
        """Check if any adjustment is applied."""
        return self.adj is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        result = {}
        if self.adj is not None:
            result['adj'] = self.adj
        if self.level is not None:
            result['level'] = self.level
        if self.location is not None:
            result['location'] = self.location
        if self.df_breakpoints is not None:
            result['df_breakpoints'] = self.df_breakpoints
        if self.price_threshold != Defaults.PRICE_THRESHOLD:
            result['price_threshold'] = self.price_threshold
        return result


# =============================================================================
# Formation Options (StrategyFormation-specific)
# =============================================================================
@dataclass
class FormationConfig:
    """
    Configuration for portfolio formation options.
    
    These are StrategyFormation-specific options that control how
    portfolios are formed and what outputs are generated.
    
    Attributes
    ----------
    dynamic_weights : bool, default False
        Use dynamic weighting (weights at t+h-1 instead of t)
    compute_turnover : bool, default False
        Whether to compute turnover statistics
    save_idx : bool, default False
        Whether to save portfolio membership indices
    banding_threshold : float, optional
        Rank-based threshold to limit cross-period changes
        Example: 0.1 means bonds must move 10% of portfolios to switch
    turnover_nanmean : bool, default True
        For turnover computation with staggered rebalancing:
        - True: Use np.nanmean for cohort averaging (ignores NaN, new behavior)
        - False: Use np.mean (propagates NaN, matches old package for backward compatibility)
    verbose : bool, default True
        Enable verbose output
    
    Examples
    --------
    >>> # Basic formation
    >>> formation_config = FormationConfig()
    >>> 
    >>> # With turnover and banding
    >>> formation_config = FormationConfig(
    ...     compute_turnover=True,
    ...     banding_threshold=0.1
    ... )
    >>> 
    >>> # Dynamic weights and save indices
    >>> formation_config = FormationConfig(
    ...     dynamic_weights=True,
    ...     save_idx=True
    ... )
    """
    
    dynamic_weights: bool = False
    compute_turnover: bool = False
    save_idx: bool = False
    banding_threshold: Optional[float] = None
    turnover_nanmean: bool = True  # Use nanmean (True, new) or mean (False, old) for cohort averaging
    verbose: bool = Defaults.VERBOSE
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.banding_threshold is not None:
            self._validate_banding_threshold()
    
    def _validate_banding_threshold(self):
        """Validate banding threshold."""
        if not isinstance(self.banding_threshold, (int, float)):
            raise TypeError("banding_threshold must be numeric")
        if not 0 < self.banding_threshold < 1:
            raise ValueError("banding_threshold must be between 0 and 1")


# =============================================================================
# Complete Configuration
# =============================================================================
@dataclass
class StrategyFormationConfig:
    """
    Complete configuration for StrategyFormation.
    
    This combines all StrategyFormation-specific configuration options.
    Strategy-specific configuration (holding_period, num_portfolios, etc.)
    is handled by the Strategy class itself.
    
    Attributes
    ----------
    data : DataConfig
        Data filtering configuration
    formation : FormationConfig
        Formation options
    filters : FilterConfig, optional
        Ex-post filter configuration
    
    Examples
    --------
    >>> # Basic configuration
    >>> config = StrategyFormationConfig(
    ...     data=DataConfig(rating='IG'),
    ...     formation=FormationConfig()
    ... )
    >>> 
    >>> # Complete configuration
    >>> config = StrategyFormationConfig(
    ...     data=DataConfig(
    ...         rating='IG',
    ...         subset_filter={'duration': (1, 10)},
    ...         chars=['duration', 'size']
    ...     ),
    ...     formation=FormationConfig(
    ...         compute_turnover=True,
    ...         banding_threshold=0.1
    ...     ),
    ...     filters=FilterConfig(
    ...         adj='trim',
    ...         level=0.01
    ...     )
    ... )
    >>> 
    >>> # Use with StrategyFormation
    >>> from PyBondLab import StrategyFormation, Momentum
    >>> strategy = Momentum(holding_period=6, num_portfolios=10, 
    ...                     lookback_period=6, skip=1)
    >>> sf = StrategyFormation(data=data, strategy=strategy, config=config)
    >>> sf.fit()
    """
    
    data: DataConfig
    formation: FormationConfig
    filters: Optional[FilterConfig] = None
    
    def __post_init__(self):
        """Validate complete configuration."""
        # Ensure filters is a FilterConfig object
        if self.filters is None:
            self.filters = FilterConfig()
    
    @property
    def has_filters(self) -> bool:
        """Check if ex-post filters are applied."""
        return self.filters is not None and self.filters.has_adjustment
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        
        Returns
        -------
        dict
            Dictionary representation of configuration
        """
        result = {
            'rating': self.data.rating,
            'subset_filter': self.data.subset_filter,
            'chars': self.data.chars,
            'dynamic_weights': self.formation.dynamic_weights,
            'turnover': self.formation.compute_turnover,
            'save_idx': self.formation.save_idx,
            'banding_threshold': self.formation.banding_threshold,
            'verbose': self.formation.verbose,
        }
        
        if self.has_filters:
            result['filters'] = self.filters.to_dict()
        
        return result
    
    @classmethod
    def from_legacy_params(cls, **kwargs):
        """
        Create configuration from legacy parameter dict.
        
        This allows backward compatibility with old code that passes
        individual parameters to StrategyFormation.
        
        Parameters
        ----------
        **kwargs : dict
            Legacy parameters
        
        Returns
        -------
        StrategyFormationConfig
            Configuration object
        
        Examples
        --------
        >>> # Convert old-style parameters
        >>> config = StrategyFormationConfig.from_legacy_params(
        ...     rating='IG',
        ...     turnover=True,
        ...     chars=['duration'],
        ...     filters={'adj': 'trim', 'level': 0.01}
        ... )
        """
        # Data config
        data = DataConfig(
            rating=kwargs.get('rating'),
            subset_filter=kwargs.get('subset_filter'),
            chars=kwargs.get('chars'),
        )
        
        # Formation config
        formation = FormationConfig(
            dynamic_weights=kwargs.get('dynamic_weights', False),
            compute_turnover=kwargs.get('turnover', False),
            save_idx=kwargs.get('save_idx', False),
            banding_threshold=kwargs.get('banding_threshold'),
            verbose=kwargs.get('verbose', Defaults.VERBOSE),
        )
        
        # Filter config
        filters_dict = kwargs.get('filters', {})
        if filters_dict:
            filters = FilterConfig(
                adj=filters_dict.get('adj'),
                level=filters_dict.get('level'),
                location=filters_dict.get('location'),
                df_breakpoints=filters_dict.get('df_breakpoints'),
                price_threshold=filters_dict.get('price_threshold', Defaults.PRICE_THRESHOLD),
            )
        else:
            filters = None
        
        return cls(
            data=data,
            formation=formation,
            filters=filters,
        )


# =============================================================================
# Convenience Functions
# =============================================================================
def create_default_config(**kwargs) -> StrategyFormationConfig:
    """
    Create a default configuration with optional overrides.
    
    Parameters
    ----------
    **kwargs : dict
        Override default values
    
    Returns
    -------
    StrategyFormationConfig
        Configuration with defaults and overrides
    
    Examples
    --------
    >>> # All defaults
    >>> config = create_default_config()
    >>> 
    >>> # Override rating
    >>> config = create_default_config(rating='IG')
    >>> 
    >>> # Override multiple
    >>> config = create_default_config(
    ...     rating='IG',
    ...     turnover=True,
    ...     chars=['duration']
    ... )
    """
    return StrategyFormationConfig.from_legacy_params(**kwargs)