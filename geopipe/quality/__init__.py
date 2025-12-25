"""Data quality intelligence for geopipe fusion schemas."""

from geopipe.quality.checks import (
    IssueSeverity,
    IssueCategory,
    QualityIssue,
    QualityCheck,
    TemporalOverlapCheck,
    CRSAlignmentCheck,
    BoundsOverlapCheck,
    TemporalGapCheck,
    SpatialCoverageCheck,
    MissingValueCheck,
    GeocodingPrecisionCheck,
    DEFAULT_CHECKS,
)
from geopipe.quality.preflight import (
    PreflightStatus,
    PreflightIssue,
    PreflightResult,
    PreflightCheck,
    PathAccessibilityCheck,
    RequiredConfigCheck,
    RasterFormatCheck,
    TabularFormatCheck,
    CoordinateRangeCheck,
    AuthenticationCheck,
    DEFAULT_PREFLIGHT_CHECKS,
    run_preflight_checks,
    check_data_quality,
)
from geopipe.quality.report import QualityReport

__all__ = [
    # Core quality types
    "IssueSeverity",
    "IssueCategory",
    "QualityIssue",
    "QualityCheck",
    "QualityReport",
    # Post-load quality checks
    "TemporalOverlapCheck",
    "CRSAlignmentCheck",
    "BoundsOverlapCheck",
    "TemporalGapCheck",
    "SpatialCoverageCheck",
    "MissingValueCheck",
    "GeocodingPrecisionCheck",
    "DEFAULT_CHECKS",
    # Preflight checks
    "PreflightStatus",
    "PreflightIssue",
    "PreflightResult",
    "PreflightCheck",
    "PathAccessibilityCheck",
    "RequiredConfigCheck",
    "RasterFormatCheck",
    "TabularFormatCheck",
    "CoordinateRangeCheck",
    "AuthenticationCheck",
    "DEFAULT_PREFLIGHT_CHECKS",
    "run_preflight_checks",
    "check_data_quality",
]
