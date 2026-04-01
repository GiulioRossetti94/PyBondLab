"""
Specification Grid Validator for AnomalyAssayer

Validates specification combinations before running to catch:
1. Incompatible breakpoint universe / rating filter combinations
2. Logically impossible specifications
3. Specifications likely to produce empty or degenerate portfolios
4. Breakpoint scheme inconsistencies

@author: Giulio Rossetti, Alex Dickerson
"""

from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

from .constants import RatingBounds, get_rating_bounds as core_get_rating_bounds


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "ERROR"      # Specification is invalid, will fail or produce garbage
    WARNING = "WARNING"  # Specification is suspicious, may produce poor results
    INFO = "INFO"        # Informational note about specification


@dataclass
class ValidationIssue:
    """A single validation issue found in specification grid."""
    severity: ValidationSeverity
    spec_id: str
    message: str
    details: Optional[str] = None

    def __str__(self):
        base = f"[{self.severity.value}] {self.spec_id}: {self.message}"
        if self.details:
            base += f"\n    Details: {self.details}"
        return base


@dataclass
class ValidationResult:
    """Result of specification validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    total_specs: int = 0
    valid_specs: int = 0
    error_specs: int = 0
    warning_specs: int = 0

    def __str__(self):
        lines = [
            f"Validation Result: {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Total specs: {self.total_specs}",
            f"  Valid: {self.valid_specs}",
            f"  Errors: {self.error_specs}",
            f"  Warnings: {self.warning_specs}",
        ]
        if self.issues:
            lines.append("\nIssues found:")
            for issue in self.issues:
                lines.append(f"  {issue}")
        return "\n".join(lines)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class SpecificationValidator:
    """
    Validates specification grids for AnomalyAssayer.

    Catches problematic combinations like:
    - IG-only breakpoints with NIG-only rating filter (disjoint populations)
    - NIG-only breakpoints with IG-only rating filter
    - Custom breakpoints that don't match portfolio count
    - Overly restrictive filter combinations

    Usage
    -----
    >>> validator = SpecificationValidator()
    >>> result = validator.validate(specs)
    >>> if not result.is_valid:
    ...     print(result)
    ...     # Fix specs or proceed with caution

    >>> # Or validate with data to check for empty universes
    >>> result = validator.validate(specs, data=bond_data)
    """

    # Rating bounds for common categories
    RATING_BOUNDS = {
        'ig': (RatingBounds.IG_MIN, RatingBounds.IG_MAX),
        'IG': (RatingBounds.IG_MIN, RatingBounds.IG_MAX),
        'nig': (RatingBounds.NIG_MIN, RatingBounds.NIG_MAX),
        'NIG': (RatingBounds.NIG_MIN, RatingBounds.NIG_MAX),
        'all': (RatingBounds.IG_MIN, RatingBounds.NIG_MAX),
        'ALL': (RatingBounds.IG_MIN, RatingBounds.NIG_MAX),
        None: (RatingBounds.IG_MIN, RatingBounds.NIG_MAX),
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def validate(
        self,
        specs: Dict[str, Any],
        data: Optional['pd.DataFrame'] = None,
        rating_col: str = 'RATING_NUM',
    ) -> ValidationResult:
        """
        Validate specification grid.

        Parameters
        ----------
        specs : dict
            Specification grid with keys:
            - 'weighting': list of ['EW', 'VW']
            - 'portfolio_structures': list of (n_ports, name, breakpoints)
              Use breakpoints=None for equal percentiles (recommended)
            - 'rating_filters': dict of {name: filter_value}
            - 'bp_universes': dict of {name: callable or None}
            - 'maturity_filters': dict of {name: (min, max) or None}
        data : pd.DataFrame, optional
            If provided, validates against actual data for empty universe checks
        rating_col : str
            Name of rating column in data

        Returns
        -------
        ValidationResult
            Contains is_valid flag and list of issues found
        """
        issues = []
        spec_count = 0
        error_count = 0
        warning_count = 0

        # Extract components with defaults
        weightings = specs.get('weighting', ['EW', 'VW'])
        portfolio_structures = specs.get('portfolio_structures', [])
        rating_filters = specs.get('rating_filters', {'all': None})
        bp_universes = specs.get('bp_universes', {'all': None})
        maturity_filters = specs.get('maturity_filters', {'all': None})

        # Validate portfolio structures
        structure_issues = self._validate_portfolio_structures(portfolio_structures)
        issues.extend(structure_issues)

        # Generate all combinations and validate each
        for weight in weightings:
            for n_ports, bp_scheme, breakpoints in portfolio_structures:
                for bp_name, bp_func in bp_universes.items():
                    for rat_name, rat_filter in rating_filters.items():
                        for mat_name, mat_filter in maturity_filters.items():
                            spec_count += 1
                            spec_id = f"{weight}_{n_ports}p_{bp_scheme}_{bp_name}_{rat_name}_{mat_name}"

                            # Check each validation rule
                            spec_issues = self._validate_single_spec(
                                spec_id=spec_id,
                                n_ports=n_ports,
                                breakpoints=breakpoints,
                                bp_name=bp_name,
                                bp_func=bp_func,
                                rat_name=rat_name,
                                rat_filter=rat_filter,
                                mat_name=mat_name,
                                mat_filter=mat_filter,
                                data=data,
                                rating_col=rating_col,
                            )

                            issues.extend(spec_issues)

                            for issue in spec_issues:
                                if issue.severity == ValidationSeverity.ERROR:
                                    error_count += 1
                                elif issue.severity == ValidationSeverity.WARNING:
                                    warning_count += 1

        valid_specs = spec_count - error_count
        is_valid = error_count == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            total_specs=spec_count,
            valid_specs=valid_specs,
            error_specs=error_count,
            warning_specs=warning_count,
        )

    def _validate_portfolio_structures(
        self,
        structures: List[Tuple]
    ) -> List[ValidationIssue]:
        """Validate portfolio structure definitions."""
        issues = []
        valid_structures: list[tuple[int, str, Optional[List[float]]]] = []

        for i, struct in enumerate(structures):
            structure_valid = True
            if len(struct) != 3:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    spec_id=f"structure_{i}",
                    message="Portfolio structure must be (n_ports, name, breakpoints) tuple",
                    details=f"Got: {struct}"
                ))
                continue

            n_ports, name, breakpoints = struct

            # Check n_ports is valid
            if not isinstance(n_ports, int) or n_ports < 2:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    spec_id=f"structure_{name}",
                    message=f"n_ports must be integer >= 2, got {n_ports}",
                ))
                structure_valid = False

            # If breakpoints is None, PyBondLab will compute equal percentiles - this is FINE
            if breakpoints is None:
                if structure_valid:
                    valid_structures.append((n_ports, name, breakpoints))
                continue

            # Check custom breakpoints match n_ports
            expected_breakpoints = n_ports - 1
            if len(breakpoints) != expected_breakpoints:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    spec_id=f"structure_{name}",
                    message=f"Breakpoints count mismatch: {n_ports} portfolios requires {expected_breakpoints} breakpoints",
                    details=f"Got {len(breakpoints)} breakpoints: {breakpoints}"
                ))
                structure_valid = False

            # Check breakpoints are sorted and in valid range
            if not all(0 < bp < 100 for bp in breakpoints):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    spec_id=f"structure_{name}",
                    message="Breakpoints must be between 0 and 100 (exclusive)",
                    details=f"Got: {breakpoints}"
                ))
                structure_valid = False

            if breakpoints != sorted(breakpoints):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    spec_id=f"structure_{name}",
                    message="Breakpoints must be in ascending order",
                    details=f"Got: {breakpoints}"
                ))
                structure_valid = False

            if structure_valid:
                valid_structures.append((n_ports, name, breakpoints))

        # Long-short redundancy warnings:
        # anomaly assay stores only the extreme long-short spread, so structures
        # with identical lower/upper cutoffs are behaviorally redundant even if
        # their interior portfolios differ.
        seen_extremes: dict[tuple[float, float], tuple[int, str, Optional[List[float]]]] = {}
        for n_ports, name, breakpoints in valid_structures:
            extreme_key = self._get_extreme_bucket_signature(n_ports, breakpoints)
            if extreme_key is None:
                continue

            prior = seen_extremes.get(extreme_key)
            if prior is None:
                seen_extremes[extreme_key] = (n_ports, name, breakpoints)
                continue

            prior_n_ports, prior_name, prior_breakpoints = prior
            if (prior_n_ports, prior_breakpoints) == (n_ports, breakpoints):
                continue

            low_bp, high_bp = extreme_key
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                spec_id=f"structure_{name}",
                message=(
                    f"Long-short equivalent structure: '{name}' shares the same extreme "
                    f"cutoffs ({low_bp:g}, {high_bp:g}) as '{prior_name}'"
                ),
                details=(
                    "Anomaly assay keeps only the extreme long-short spread. "
                    "These structures differ in the number of interior portfolios but "
                    "should produce the same long-short return series."
                ),
            ))

        return issues

    @staticmethod
    def _get_extreme_bucket_signature(
        n_ports: int,
        breakpoints: Optional[List[float]],
    ) -> Optional[Tuple[float, float]]:
        """Return the lower/upper extreme cutoffs used by a portfolio structure."""
        if not isinstance(n_ports, int) or n_ports < 2:
            return None

        if breakpoints is None:
            step = 100.0 / n_ports
            return (step, 100.0 - step)

        if len(breakpoints) == 0:
            return None

        return (float(breakpoints[0]), float(breakpoints[-1]))

    def _validate_single_spec(
        self,
        spec_id: str,
        n_ports: int,
        breakpoints: Optional[List[float]],
        bp_name: str,
        bp_func: Optional[Callable],
        rat_name: str,
        rat_filter: Optional[Union[str, Tuple[int, int]]],
        mat_name: str,
        mat_filter: Optional[Tuple[float, float]],
        data: Optional['pd.DataFrame'],
        rating_col: str,
    ) -> List[ValidationIssue]:
        """Validate a single specification."""
        issues = []

        # Rule 1: Check breakpoint universe vs rating filter compatibility
        bp_rating_issue = self._check_bp_rating_compatibility(
            spec_id, bp_name, bp_func, rat_name, rat_filter
        )
        if bp_rating_issue:
            issues.append(bp_rating_issue)

        # Rule 2: Check for overly restrictive combinations
        restrictive_issue = self._check_restrictive_combination(
            spec_id, rat_name, rat_filter, mat_name, mat_filter
        )
        if restrictive_issue:
            issues.append(restrictive_issue)

        # Rule 3: If data provided, check for empty universes
        if data is not None:
            empty_issue = self._check_empty_universe(
                spec_id, rat_filter, mat_filter, data, rating_col
            )
            if empty_issue:
                issues.append(empty_issue)

        # Rule 4: Check bp_func required columns exist in data
        if data is not None:
            col_issue = self._check_bp_func_columns(
                spec_id, bp_name, bp_func, data
            )
            if col_issue:
                issues.append(col_issue)

        return issues

    def _check_bp_rating_compatibility(
        self,
        spec_id: str,
        bp_name: str,
        bp_func: Optional[Callable],
        rat_name: str,
        rat_filter: Optional[Union[str, Tuple[int, int]]],
    ) -> Optional[ValidationIssue]:
        """
        Check if breakpoint universe and rating filter are compatible.

        ERROR cases (disjoint populations):
        - bp_universe='ig_only' + rating_filter='nig'
        - bp_universe='nig_only' + rating_filter='ig'

        WARNING cases (partial overlap):
        - bp_universe='ig_only' + rating_filter='all' (valid but worth noting)
        """
        # Parse rating filter bounds
        rat_bounds = self._get_rating_bounds(rat_filter)

        # Infer breakpoint universe bounds from name
        bp_bounds = self._infer_bp_universe_bounds(bp_name, bp_func)

        if bp_bounds is None or rat_bounds is None:
            return None  # Can't determine, skip check

        # Check for disjoint populations (no overlap)
        bp_min, bp_max = bp_bounds
        rat_min, rat_max = rat_bounds

        # Populations are disjoint if: bp_max < rat_min OR rat_max < bp_min
        if bp_max < rat_min or rat_max < bp_min:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                spec_id=spec_id,
                message=f"DISJOINT populations: breakpoint universe '{bp_name}' ({bp_bounds}) "
                        f"does not overlap with rating filter '{rat_name}' ({rat_bounds})",
                details="Breakpoints computed on one population cannot meaningfully sort a completely different population. "
                        "All bonds may fall into extreme portfolios or portfolios may be empty."
            )

        # Check for partial overlap (subset relationship is OK, but note it)
        # bp_universe is subset of rating_filter: OK (conservative breakpoints)
        # rating_filter is subset of bp_universe: WARNING (extrapolating breakpoints)
        if rat_min < bp_min or rat_max > bp_max:
            # Rating filter includes bonds outside breakpoint universe
            return ValidationIssue(
                severity=ValidationSeverity.WARNING,
                spec_id=spec_id,
                message=f"Rating filter '{rat_name}' ({rat_bounds}) extends beyond "
                        f"breakpoint universe '{bp_name}' ({bp_bounds})",
                details="Some bonds being sorted are outside the population used to compute breakpoints. "
                        "This may cause uneven portfolio sizes."
            )

        return None

    def _get_rating_bounds(
        self,
        rat_filter: Optional[Union[str, Tuple[int, int]]]
    ) -> Optional[Tuple[int, int]]:
        """Get (min, max) rating bounds from filter specification."""
        if rat_filter is None:
            return (RatingBounds.IG_MIN, RatingBounds.NIG_MAX)

        if isinstance(rat_filter, str):
            rat_upper = rat_filter.upper()
            if rat_upper in ('IG', 'NIG'):
                return core_get_rating_bounds(rat_upper)
            return self.RATING_BOUNDS.get(rat_upper)

        if isinstance(rat_filter, tuple) and len(rat_filter) == 2:
            return rat_filter

        return None

    def _infer_bp_universe_bounds(
        self,
        bp_name: str,
        bp_func: Optional[Callable]
    ) -> Optional[Tuple[int, int]]:
        """
        Infer rating bounds of breakpoint universe from name/function.

        Heuristic based on common naming patterns:
        - 'all', None, 'full' -> (1, 22)
        - 'ig_only', 'ig' -> (1, 10)
        - 'nig_only', 'nig' -> (11, 22)
        """
        if bp_func is None:
            return (RatingBounds.IG_MIN, RatingBounds.NIG_MAX)  # No filter = all bonds

        bp_name_lower = bp_name.lower()

        if 'nig_only' in bp_name_lower or bp_name_lower == 'nig':
            return (RatingBounds.NIG_MIN, RatingBounds.NIG_MAX)

        if 'ig_only' in bp_name_lower or bp_name_lower == 'ig':
            return (RatingBounds.IG_MIN, RatingBounds.IG_MAX)

        if bp_name_lower in ('all', 'full', 'none'):
            return (RatingBounds.IG_MIN, RatingBounds.NIG_MAX)

        # Can't infer from name, return None (skip check)
        return None

    def _check_restrictive_combination(
        self,
        spec_id: str,
        rat_name: str,
        rat_filter: Optional[Union[str, Tuple[int, int]]],
        mat_name: str,
        mat_filter: Optional[Tuple[float, float]],
    ) -> Optional[ValidationIssue]:
        """Check for overly restrictive filter combinations."""
        restrictive_ratings = {
            (1, 3),   # AAA-AA only
            (1, 4),   # AAA-AA+ only
            (19, 21), # CCC and below
            (20, 21), # CC-D only
        }

        restrictive_maturities = {
            (0, 1),     # Very short (<1 year)
            (30, 100),  # Very long (>30 years)
            (50, 100),  # Ultra long (>50 years)
        }

        rat_bounds = self._get_rating_bounds(rat_filter)

        # Check if both rating and maturity are restrictive
        is_restrictive_rating = rat_bounds in restrictive_ratings if rat_bounds else False
        is_restrictive_maturity = mat_filter in restrictive_maturities if mat_filter else False

        if is_restrictive_rating and is_restrictive_maturity:
            return ValidationIssue(
                severity=ValidationSeverity.WARNING,
                spec_id=spec_id,
                message=f"Very restrictive filters: rating={rat_name}, maturity={mat_name}",
                details="This combination may result in very few bonds, potentially "
                        "causing unstable portfolio sorts or empty portfolios."
            )

        return None

    def _check_empty_universe(
        self,
        spec_id: str,
        rat_filter: Optional[Union[str, Tuple[int, int]]],
        mat_filter: Optional[Tuple[float, float]],
        data: 'pd.DataFrame',
        rating_col: str,
    ) -> Optional[ValidationIssue]:
        """Check if filter combination results in empty or near-empty universe."""
        import pandas as pd

        mask = pd.Series(True, index=data.index)

        # Apply rating filter
        if rat_filter is not None:
            rat_bounds = self._get_rating_bounds(rat_filter)
            if rat_bounds and rating_col in data.columns:
                min_r, max_r = rat_bounds
                mask &= (data[rating_col] >= min_r) & (data[rating_col] <= max_r)

        # Apply maturity filter
        if mat_filter is not None:
            mat_col = 'tmat'  # Common maturity column name
            if mat_col in data.columns:
                min_m, max_m = mat_filter
                mask &= (data[mat_col] >= min_m) & (data[mat_col] <= max_m)

        n_obs = mask.sum()
        n_dates = data.loc[mask, 'date'].nunique() if 'date' in data.columns else 0

        if n_obs == 0:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                spec_id=spec_id,
                message="Empty universe: no observations pass filters",
                details=f"Rating filter: {rat_filter}, Maturity filter: {mat_filter}"
            )

        # Check average bonds per date
        if n_dates > 0:
            avg_bonds_per_date = n_obs / n_dates
            if avg_bonds_per_date < 10:
                return ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    spec_id=spec_id,
                    message=f"Sparse universe: average {avg_bonds_per_date:.1f} bonds per date",
                    details="May result in unstable portfolio sorts. Consider relaxing filters."
                )

        return None

    def _check_bp_func_columns(
        self,
        spec_id: str,
        bp_name: str,
        bp_func: Optional[Callable],
        data: Optional['pd.DataFrame'],
    ) -> Optional[ValidationIssue]:
        """
        Check that bp_func required columns exist in data.

        bp_func can declare required columns via:
        - bp_func.required_columns = ['col1', 'col2']
        """
        if bp_func is None or data is None:
            return None

        if not hasattr(bp_func, 'required_columns'):
            return None

        missing = [col for col in bp_func.required_columns if col not in data.columns]

        if missing:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                spec_id=spec_id,
                message=f"bp_func '{bp_name}' requires missing columns: {missing}",
                details=f"Required: {list(bp_func.required_columns)}"
            )

        return None


def validate_specs(
    specs: Dict[str, Any],
    data: Optional['pd.DataFrame'] = None,
    rating_col: str = 'RATING_NUM',
    raise_on_error: bool = False,
    verbose: bool = True,
) -> ValidationResult:
    """
    Convenience function to validate specification grid.

    Parameters
    ----------
    specs : dict
        Specification grid
    data : pd.DataFrame, optional
        Bond data for universe size checks
    rating_col : str
        Rating column name
    raise_on_error : bool
        If True, raise ValueError on validation errors
    verbose : bool
        Print validation result

    Returns
    -------
    ValidationResult

    Raises
    ------
    ValueError
        If raise_on_error=True and validation fails

    Examples
    --------
    >>> from PyBondLab.spec_validator import validate_specs
    >>>
    >>> specs = {
    ...     'weighting': ['EW', 'VW'],
    ...     'portfolio_structures': [
    ...         (5, 'quintiles', None),      # None = PyBondLab computes equal quintiles
    ...         (3, 'extreme', [10, 90]),    # Custom: 10/80/10 split
    ...     ],
    ...     'rating_filters': {'all': None, 'ig': (1, 10), 'nig': (11, 22)},
    ...     'bp_universes': {'all': None, 'ig_only': lambda df: df['RATING_NUM'] <= 10},
    ...     'maturity_filters': {'all': None, 'short': (0, 5)},
    ... }
    >>>
    >>> result = validate_specs(specs)
    >>> # Will flag: ig_only breakpoints + nig rating filter as ERROR
    """
    validator = SpecificationValidator(verbose=verbose)
    result = validator.validate(specs, data=data, rating_col=rating_col)

    if verbose:
        print(result)

    if raise_on_error and not result.is_valid:
        error_msgs = [str(e) for e in result.errors]
        raise ValueError(f"Specification validation failed:\n" + "\n".join(error_msgs))

    return result


def generate_spec_list(specs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate list of all specification combinations from a spec grid.

    Parameters
    ----------
    specs : dict
        Specification grid with keys:
        - 'weighting': list of ['EW', 'VW']
        - 'portfolio_structures': list of (n_ports, name, breakpoints)
        - 'rating_filters': dict of {name: filter_value}
        - 'bp_universes': dict of {name: callable or None}
        - 'maturity_filters': dict of {name: (min, max) or None}

    Returns
    -------
    list of dict
        Each dict contains:
        - spec_id: str
        - weighting: str
        - n_ports: int
        - bp_scheme: str
        - breakpoints: list or None
        - rating_name: str
        - rating_filter: tuple or None
        - bp_name: str
        - bp_func: callable or None
        - maturity_name: str
        - maturity_filter: tuple or None
    """
    spec_list = []

    weightings = specs.get('weighting', ['EW', 'VW'])
    portfolio_structures = specs.get('portfolio_structures', [])
    rating_filters = specs.get('rating_filters', {'all': None})
    bp_universes = specs.get('bp_universes', {'all': None})
    maturity_filters = specs.get('maturity_filters', {'all': None})

    for weight in weightings:
        for n_ports, bp_scheme, breakpoints in portfolio_structures:
            for bp_name, bp_func in bp_universes.items():
                for rat_name, rat_filter in rating_filters.items():
                    for mat_name, mat_filter in maturity_filters.items():
                        spec_id = f"{weight}_{n_ports}p_{bp_scheme}_{bp_name}_{rat_name}_{mat_name}"

                        spec_list.append({
                            'spec_id': spec_id,
                            'weighting': weight,
                            'n_ports': n_ports,
                            'bp_scheme': bp_scheme,
                            'breakpoints': breakpoints,
                            'rating_name': rat_name,
                            'rating_filter': rat_filter,
                            'bp_name': bp_name,
                            'bp_func': bp_func,
                            'maturity_name': mat_name,
                            'maturity_filter': mat_filter,
                        })

    return spec_list


def get_valid_spec_list(
    specs: Dict[str, Any],
    data: Optional['pd.DataFrame'] = None,
    rating_col: str = 'RATING_NUM',
    exclude_warnings: bool = False,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], ValidationResult]:
    """
    Generate list of valid specifications, filtering out invalid ones.

    Parameters
    ----------
    specs : dict
        Specification grid
    data : pd.DataFrame, optional
        Bond data for universe size checks
    rating_col : str
        Rating column name
    exclude_warnings : bool
        If True, also exclude specs with warnings
    verbose : bool
        Print summary of filtering

    Returns
    -------
    tuple of (list, ValidationResult)
        - List of valid spec dicts (ready to iterate over)
        - ValidationResult with details of what was filtered

    Examples
    --------
    >>> specs = {
    ...     'weighting': ['EW', 'VW'],
    ...     'portfolio_structures': [(5, 'quintiles', None)],
    ...     'rating_filters': {'all': None, 'ig': (1, 10), 'nig': (11, 22)},
    ...     'bp_universes': {'all': None, 'ig_only': lambda df: df['RATING_NUM'] <= 10},
    ...     'maturity_filters': {'all': None},
    ... }
    >>> valid_specs, result = get_valid_spec_list(specs)
    >>> print(f"Running {len(valid_specs)} valid specs (skipped {result.error_specs} errors)")
    >>> for spec in valid_specs:
    ...     # Run each spec
    ...     print(f"Running {spec['spec_id']}")
    """
    # Validate
    validator = SpecificationValidator(verbose=False)
    result = validator.validate(specs, data=data, rating_col=rating_col)

    # Get spec_ids to exclude
    exclude_ids = {issue.spec_id for issue in result.errors}
    if exclude_warnings:
        exclude_ids.update(issue.spec_id for issue in result.warnings)

    # Generate all specs and filter
    all_specs = generate_spec_list(specs)
    valid_specs = [s for s in all_specs if s['spec_id'] not in exclude_ids]

    if verbose:
        n_total = len(all_specs)
        n_valid = len(valid_specs)
        n_skipped = n_total - n_valid
        if n_skipped > 0:
            print(f"Spec validation: {n_valid}/{n_total} valid ({n_skipped} skipped)")
            if result.error_specs > 0:
                print(f"  Errors: {result.error_specs} (invalid bp_universe/rating combinations)")
            if exclude_warnings and result.warning_specs > 0:
                print(f"  Warnings: {result.warning_specs} (excluded)")
        else:
            print(f"Spec validation: all {n_total} specs valid")

    return valid_specs, result


def filter_spec_list(
    spec_list: List[Dict[str, Any]],
    validation_result: ValidationResult,
    exclude_warnings: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filter an existing spec list based on validation results.

    Parameters
    ----------
    spec_list : list of dict
        List of spec dictionaries (from generate_spec_list)
    validation_result : ValidationResult
        Result from validate_specs()
    exclude_warnings : bool
        If True, also exclude specs with warnings

    Returns
    -------
    list of dict
        Filtered list with invalid specs removed
    """
    exclude_ids = {issue.spec_id for issue in validation_result.errors}
    if exclude_warnings:
        exclude_ids.update(issue.spec_id for issue in validation_result.warnings)

    return [s for s in spec_list if s['spec_id'] not in exclude_ids]
