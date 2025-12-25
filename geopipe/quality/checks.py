"""Quality check implementations for geopipe data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from geopipe.sources.base import DataSource
    from geopipe.fusion.schema import FusionSchema


class IssueSeverity(str, Enum):
    """Severity level for quality issues."""

    ERROR = "error"  # Blocks execution
    WARNING = "warning"  # May affect results
    INFO = "info"  # Informational only


class IssueCategory(str, Enum):
    """Category of quality issue."""

    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SCHEMA = "schema"
    ALIGNMENT = "alignment"
    DATA_QUALITY = "data_quality"


class QualityIssue(BaseModel):
    """A single quality issue detected in a data source."""

    source_name: str
    category: IssueCategory
    severity: IssueSeverity
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    auto_fixable: bool = False
    fix_description: str | None = None

    def __str__(self) -> str:
        severity_icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}
        icon = severity_icon.get(self.severity.value, "•")
        return f"{icon} [{self.source_name}] {self.message}"


class QualityCheck(ABC):
    """Base class for quality checks."""

    name: str
    category: IssueCategory
    description: str = ""

    @abstractmethod
    def check(
        self,
        source: "DataSource",
        schema: "FusionSchema",
        sample_size: int = 1000,
    ) -> list[QualityIssue]:
        """
        Run quality check on a data source.

        Args:
            source: Data source to check
            schema: Fusion schema context
            sample_size: Sample size for data-level checks

        Returns:
            List of quality issues found
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CrossSourceCheck(QualityCheck):
    """Base class for checks that operate across multiple sources."""

    @abstractmethod
    def check_all(
        self,
        sources: list["DataSource"],
        schema: "FusionSchema",
    ) -> list[QualityIssue]:
        """Run check across all sources."""
        ...

    def check(
        self,
        source: "DataSource",
        schema: "FusionSchema",
        sample_size: int = 1000,
    ) -> list[QualityIssue]:
        """Single-source check delegates to check_all."""
        return []


class TemporalOverlapCheck(CrossSourceCheck):
    """Check that all sources have overlapping temporal ranges."""

    name = "temporal_overlap"
    category = IssueCategory.ALIGNMENT
    description = "Verify temporal ranges overlap across sources"

    def check_all(
        self,
        sources: list["DataSource"],
        schema: "FusionSchema",
    ) -> list[QualityIssue]:
        issues = []

        # Collect temporal ranges from sources
        ranges: dict[str, tuple[str, str] | None] = {}
        for source in sources:
            if hasattr(source, "temporal_range") and source.temporal_range:
                ranges[source.name] = source.temporal_range
            elif hasattr(source, "config") and source.config.get("temporal_range"):
                ranges[source.name] = tuple(source.config["temporal_range"])

        if len(ranges) < 2:
            return issues  # Need at least 2 sources to check overlap

        # Check pairwise overlap
        source_names = list(ranges.keys())
        for i, name1 in enumerate(source_names):
            for name2 in source_names[i + 1 :]:
                range1 = ranges[name1]
                range2 = ranges[name2]

                if range1 and range2:
                    start1, end1 = range1
                    start2, end2 = range2

                    # Check if ranges overlap
                    if end1 < start2 or end2 < start1:
                        issues.append(
                            QualityIssue(
                                source_name=f"{name1}, {name2}",
                                category=IssueCategory.ALIGNMENT,
                                severity=IssueSeverity.ERROR,
                                message=f"Temporal ranges do not overlap: {name1} ({start1} to {end1}) vs {name2} ({start2} to {end2})",
                                details={
                                    "source1": name1,
                                    "range1": range1,
                                    "source2": name2,
                                    "range2": range2,
                                },
                            )
                        )

        return issues


class CRSAlignmentCheck(CrossSourceCheck):
    """Check that all sources use compatible coordinate reference systems."""

    name = "crs_alignment"
    category = IssueCategory.ALIGNMENT
    description = "Verify CRS compatibility across sources"

    def check_all(
        self,
        sources: list["DataSource"],
        schema: "FusionSchema",
    ) -> list[QualityIssue]:
        issues = []

        # Collect CRS from sources
        crs_map: dict[str, str | None] = {}
        for source in sources:
            crs = None
            if hasattr(source, "crs"):
                crs = source.crs
            elif hasattr(source, "config"):
                crs = source.config.get("crs") or source.config.get("target_crs")

            crs_map[source.name] = crs

        # Filter to sources with known CRS
        known_crs = {k: v for k, v in crs_map.items() if v is not None}

        if len(known_crs) < 2:
            return issues

        # Check for mismatches
        target_crs = schema.target_crs if hasattr(schema, "target_crs") else "EPSG:4326"
        unique_crs = set(known_crs.values())

        if len(unique_crs) > 1:
            issues.append(
                QualityIssue(
                    source_name="(cross-source)",
                    category=IssueCategory.ALIGNMENT,
                    severity=IssueSeverity.WARNING,
                    message=f"Multiple CRS detected: {unique_crs}. Will be transformed to {target_crs}",
                    details={"crs_by_source": known_crs, "target_crs": target_crs},
                    auto_fixable=True,
                    fix_description=f"Transform all sources to {target_crs}",
                )
            )

        return issues


class BoundsOverlapCheck(CrossSourceCheck):
    """Check that source spatial extents overlap."""

    name = "bounds_overlap"
    category = IssueCategory.SPATIAL
    description = "Verify spatial extents overlap"

    def check_all(
        self,
        sources: list["DataSource"],
        schema: "FusionSchema",
    ) -> list[QualityIssue]:
        issues = []

        # Collect bounds from sources
        bounds_map: dict[str, tuple[float, float, float, float] | None] = {}
        for source in sources:
            bounds = None
            if hasattr(source, "_bounds") and source._bounds:
                bounds = source._bounds
            elif hasattr(source, "config") and source.config.get("bounds"):
                bounds = tuple(source.config["bounds"])

            bounds_map[source.name] = bounds

        # Filter to sources with known bounds
        known_bounds = {k: v for k, v in bounds_map.items() if v is not None}

        if len(known_bounds) < 2:
            return issues

        # Check pairwise overlap
        source_names = list(known_bounds.keys())
        for i, name1 in enumerate(source_names):
            for name2 in source_names[i + 1 :]:
                b1 = known_bounds[name1]
                b2 = known_bounds[name2]

                if b1 and b2:
                    # Check if bounding boxes overlap
                    if b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1]:
                        issues.append(
                            QualityIssue(
                                source_name=f"{name1}, {name2}",
                                category=IssueCategory.SPATIAL,
                                severity=IssueSeverity.ERROR,
                                message=f"Spatial extents do not overlap: {name1} vs {name2}",
                                details={
                                    "source1": name1,
                                    "bounds1": b1,
                                    "source2": name2,
                                    "bounds2": b2,
                                },
                            )
                        )

        return issues


class TemporalGapCheck(QualityCheck):
    """Check for temporal gaps in time series data."""

    name = "temporal_gaps"
    category = IssueCategory.TEMPORAL
    description = "Detect missing dates in time series"

    def check(
        self,
        source: "DataSource",
        schema: "FusionSchema",
        sample_size: int = 1000,
    ) -> list[QualityIssue]:
        issues = []

        # This check is primarily for raster time series
        # Try to detect expected frequency and gaps
        try:
            files = source.list_files()
            if len(files) < 2:
                return issues

            # Try to extract dates from filenames
            import re
            from datetime import datetime

            dates = []
            date_patterns = [
                r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})",  # YYYY-MM-DD or YYYYMMDD
                r"(\d{4})[-_]?(\d{2})",  # YYYY-MM
            ]

            for f in files:
                fname = f.name
                for pattern in date_patterns:
                    match = re.search(pattern, fname)
                    if match:
                        groups = match.groups()
                        if len(groups) == 3:
                            try:
                                dates.append(datetime(int(groups[0]), int(groups[1]), int(groups[2])))
                            except ValueError:
                                pass
                        elif len(groups) == 2:
                            try:
                                dates.append(datetime(int(groups[0]), int(groups[1]), 1))
                            except ValueError:
                                pass
                        break

            if len(dates) < 2:
                return issues

            dates.sort()

            # Detect typical interval
            intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            if not intervals:
                return issues

            # Most common interval (mode)
            from collections import Counter

            interval_counts = Counter(intervals)
            typical_interval = interval_counts.most_common(1)[0][0]

            # Find gaps (intervals significantly larger than typical)
            gap_threshold = typical_interval * 1.5
            gaps = []
            for i, interval in enumerate(intervals):
                if interval > gap_threshold:
                    gaps.append((dates[i], dates[i + 1], interval))

            if gaps:
                issues.append(
                    QualityIssue(
                        source_name=source.name,
                        category=IssueCategory.TEMPORAL,
                        severity=IssueSeverity.WARNING,
                        message=f"Found {len(gaps)} temporal gap(s) in time series",
                        details={
                            "typical_interval_days": typical_interval,
                            "gaps": [
                                {
                                    "from": g[0].isoformat(),
                                    "to": g[1].isoformat(),
                                    "days": g[2],
                                }
                                for g in gaps[:5]  # Limit to 5
                            ],
                        },
                    )
                )

        except Exception:
            pass  # Skip if we can't analyze temporal structure

        return issues


class SpatialCoverageCheck(QualityCheck):
    """Check spatial coverage and NoData percentage."""

    name = "spatial_coverage"
    category = IssueCategory.SPATIAL
    description = "Check NoData percentage in raster data"

    def check(
        self,
        source: "DataSource",
        schema: "FusionSchema",
        sample_size: int = 1000,
    ) -> list[QualityIssue]:
        issues = []

        # This check is primarily for raster sources
        from geopipe.sources.raster import RasterSource

        if not isinstance(source, RasterSource):
            return issues

        try:
            import numpy as np

            files = source.list_files()
            if not files:
                return issues

            # Sample first file
            import rasterio

            with rasterio.open(files[0]) as src:
                data = src.read(1)
                nodata = src.nodata

                if nodata is not None:
                    nodata_count = np.sum(data == nodata)
                else:
                    nodata_count = np.sum(np.isnan(data))

                total = data.size
                nodata_pct = (nodata_count / total) * 100

                if nodata_pct > 50:
                    issues.append(
                        QualityIssue(
                            source_name=source.name,
                            category=IssueCategory.SPATIAL,
                            severity=IssueSeverity.WARNING,
                            message=f"High NoData percentage: {nodata_pct:.1f}%",
                            details={
                                "nodata_percent": nodata_pct,
                                "file_sampled": str(files[0]),
                            },
                        )
                    )
                elif nodata_pct > 20:
                    issues.append(
                        QualityIssue(
                            source_name=source.name,
                            category=IssueCategory.SPATIAL,
                            severity=IssueSeverity.INFO,
                            message=f"Moderate NoData percentage: {nodata_pct:.1f}%",
                            details={
                                "nodata_percent": nodata_pct,
                                "file_sampled": str(files[0]),
                            },
                        )
                    )

        except Exception:
            pass  # Skip if we can't read raster

        return issues


class MissingValueCheck(QualityCheck):
    """Check for missing values in tabular data."""

    name = "missing_values"
    category = IssueCategory.DATA_QUALITY
    description = "Detect missing values in key columns"

    def check(
        self,
        source: "DataSource",
        schema: "FusionSchema",
        sample_size: int = 1000,
    ) -> list[QualityIssue]:
        issues = []

        # This check is primarily for tabular sources
        from geopipe.sources.tabular import TabularSource

        if not isinstance(source, TabularSource):
            return issues

        try:
            import pandas as pd

            # Load sample
            bounds = None
            if hasattr(schema, "_bounds"):
                bounds = schema._bounds

            # Read file directly for efficiency
            from pathlib import Path

            files = source.list_files()
            if not files:
                return issues

            file_path = files[0]
            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                df = pd.read_csv(file_path, nrows=sample_size)
            elif suffix == ".parquet":
                df = pd.read_parquet(file_path)
                df = df.head(sample_size)
            else:
                return issues

            # Check key columns
            key_cols = []
            if hasattr(source, "lat_col") and source.lat_col:
                key_cols.append(source.lat_col)
            if hasattr(source, "lon_col") and source.lon_col:
                key_cols.append(source.lon_col)
            if hasattr(source, "time_col") and source.time_col:
                key_cols.append(source.time_col)

            # Also check commonly important columns
            for col in df.columns:
                if col.lower() in ["id", "geometry", "date", "timestamp", "value"]:
                    if col not in key_cols:
                        key_cols.append(col)

            # Analyze missing values
            high_missing = []
            for col in key_cols:
                if col in df.columns:
                    missing_pct = (df[col].isna().sum() / len(df)) * 100
                    if missing_pct > 5:
                        high_missing.append((col, missing_pct))

            if high_missing:
                issues.append(
                    QualityIssue(
                        source_name=source.name,
                        category=IssueCategory.DATA_QUALITY,
                        severity=IssueSeverity.WARNING,
                        message=f"Missing values in {len(high_missing)} key column(s)",
                        details={
                            "columns": {col: f"{pct:.1f}%" for col, pct in high_missing},
                            "sample_size": len(df),
                        },
                    )
                )

        except Exception:
            pass

        return issues


class GeocodingPrecisionCheck(QualityCheck):
    """Check geocoding precision for tabular data with coordinates."""

    name = "geocoding_precision"
    category = IssueCategory.SPATIAL
    description = "Analyze coordinate precision distribution"

    def check(
        self,
        source: "DataSource",
        schema: "FusionSchema",
        sample_size: int = 1000,
    ) -> list[QualityIssue]:
        issues = []

        from geopipe.sources.tabular import TabularSource

        if not isinstance(source, TabularSource):
            return issues

        try:
            import pandas as pd
            from pathlib import Path

            files = source.list_files()
            if not files:
                return issues

            file_path = files[0]
            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                df = pd.read_csv(file_path, nrows=sample_size)
            elif suffix == ".parquet":
                df = pd.read_parquet(file_path)
                df = df.head(sample_size)
            else:
                return issues

            # Check for precision column (common in geocoded data)
            precision_cols = [c for c in df.columns if "precision" in c.lower() or "geo_precision" in c.lower()]

            if precision_cols:
                col = precision_cols[0]
                precision_dist = df[col].value_counts(normalize=True)

                # High precision codes (1-2) are usually better
                low_precision_pct = 0
                for code in precision_dist.index:
                    if isinstance(code, (int, float)) and code > 2:
                        low_precision_pct += precision_dist[code] * 100

                if low_precision_pct > 30:
                    issues.append(
                        QualityIssue(
                            source_name=source.name,
                            category=IssueCategory.SPATIAL,
                            severity=IssueSeverity.WARNING,
                            message=f"{low_precision_pct:.1f}% of records have low geocoding precision",
                            details={
                                "precision_distribution": precision_dist.to_dict(),
                                "precision_column": col,
                            },
                        )
                    )

            # Also check coordinate decimal precision
            lat_col = getattr(source, "lat_col", "latitude")
            lon_col = getattr(source, "lon_col", "longitude")

            if lat_col in df.columns and lon_col in df.columns:
                # Count decimal places
                def decimal_places(x):
                    if pd.isna(x):
                        return 0
                    s = str(float(x))
                    if "." in s:
                        return len(s.split(".")[1])
                    return 0

                lat_decimals = df[lat_col].apply(decimal_places)
                lon_decimals = df[lon_col].apply(decimal_places)

                low_precision = ((lat_decimals < 3) | (lon_decimals < 3)).mean() * 100

                if low_precision > 20:
                    issues.append(
                        QualityIssue(
                            source_name=source.name,
                            category=IssueCategory.SPATIAL,
                            severity=IssueSeverity.INFO,
                            message=f"{low_precision:.1f}% of coordinates have <3 decimal places",
                            details={"low_precision_percent": low_precision},
                        )
                    )

        except Exception:
            pass

        return issues


# Default checks to run
DEFAULT_CHECKS: list[QualityCheck] = [
    TemporalOverlapCheck(),
    CRSAlignmentCheck(),
    BoundsOverlapCheck(),
    TemporalGapCheck(),
    SpatialCoverageCheck(),
    MissingValueCheck(),
    GeocodingPrecisionCheck(),
]
