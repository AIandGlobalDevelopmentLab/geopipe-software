"""Fast preflight data quality checks for geopipe data sources.

This module provides pre-loading validation that checks configuration,
accessibility, and metadata without loading full datasets. Use these checks
before expensive I/O operations to catch issues early with actionable suggestions.

Example:
    >>> from geopipe.sources.raster import RasterSource
    >>> from geopipe.quality.preflight import check_data_quality
    >>>
    >>> source = RasterSource("nightlights", path="data/*.tif")
    >>> result = check_data_quality(source)
    >>> if result.should_block:
    ...     print(result.summary())
    ...     raise ValueError("Preflight check failed")
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from geopipe.quality.checks import IssueCategory, IssueSeverity, QualityIssue

if TYPE_CHECKING:
    from geopipe.sources.base import DataSource


class PreflightStatus(str, Enum):
    """Overall preflight check status."""

    PASS = "pass"  # All checks passed
    WARN = "warn"  # Warnings present, can proceed
    FAIL = "fail"  # Critical errors, should not proceed


class PreflightIssue(BaseModel):
    """A preflight validation issue with actionable suggestion.

    Attributes:
        check_name: Name of the check that found this issue
        category: Category of the issue (spatial, temporal, schema, etc.)
        severity: Severity level (error, warning, info)
        message: Human-readable description of the issue
        suggestion: Actionable fix suggestion
        details: Additional details for programmatic access
    """

    check_name: str
    category: IssueCategory
    severity: IssueSeverity
    message: str
    suggestion: str = Field(..., description="Actionable fix suggestion")
    details: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        icon = {"error": "[X]", "warning": "[!]", "info": "[i]"}[self.severity.value]
        return f"{icon} {self.message}\n    Suggestion: {self.suggestion}"


class PreflightResult(BaseModel):
    """Result of preflight checks on a data source.

    Attributes:
        source_name: Name of the checked source
        source_type: Type of source (raster, tabular, earthengine, etc.)
        status: Overall status (pass/warn/fail)
        issues: List of issues found
        checks_run: Names of checks that were executed
        duration_ms: Time taken for checks in milliseconds

    Example:
        >>> result = source.preflight_check()
        >>> if result.should_block:
        ...     print(result.summary())
        ...     sys.exit(1)
    """

    source_name: str
    source_type: str
    status: PreflightStatus = PreflightStatus.PASS
    issues: list[PreflightIssue] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def should_block(self) -> bool:
        """Returns True if execution should be blocked."""
        return self.status == PreflightStatus.FAIL

    @property
    def has_warnings(self) -> bool:
        """Returns True if any warnings are present."""
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def errors(self) -> list[PreflightIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[PreflightIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status_text = {
            PreflightStatus.PASS: "PASS",
            PreflightStatus.WARN: "WARN",
            PreflightStatus.FAIL: "FAIL",
        }

        lines = [
            f"Preflight Check: {self.source_name} ({self.source_type})",
            f"Status: {status_text[self.status]}",
            f"Checks run: {len(self.checks_run)} in {self.duration_ms:.1f}ms",
            "",
        ]

        if self.errors:
            lines.append("ERRORS (blocking):")
            for issue in self.errors:
                lines.append(f"  {issue}")

        if self.warnings:
            lines.append("WARNINGS:")
            for issue in self.warnings:
                lines.append(f"  {issue}")

        if not self.issues:
            lines.append("All checks passed.")

        return "\n".join(lines)

    def to_quality_issues(self) -> list[QualityIssue]:
        """Convert to QualityIssue format for report integration."""
        return [
            QualityIssue(
                source_name=self.source_name,
                category=issue.category,
                severity=issue.severity,
                message=issue.message,
                details={**issue.details, "suggestion": issue.suggestion},
            )
            for issue in self.issues
        ]


class PreflightCheck(ABC):
    """Base class for preflight validation checks.

    Preflight checks are fast, pre-loading validations that check
    configuration, accessibility, and metadata without loading full data.

    Subclasses should set:
        - name: Unique identifier for the check
        - category: IssueCategory for issues found
        - applies_to: Set of source types this check applies to

    Example:
        >>> class MyCheck(PreflightCheck):
        ...     name = "my_check"
        ...     category = IssueCategory.DATA_QUALITY
        ...     applies_to = {"tabular", "raster"}
        ...
        ...     def check(self, source):
        ...         issues = []
        ...         # validation logic here
        ...         return issues
    """

    name: str
    category: IssueCategory
    description: str = ""
    applies_to: set[str] = {"all"}

    @abstractmethod
    def check(self, source: "DataSource") -> list[PreflightIssue]:
        """Run preflight check on a data source.

        Args:
            source: Data source to validate

        Returns:
            List of preflight issues found (empty if check passes)
        """
        ...

    def applies_to_source(self, source: "DataSource") -> bool:
        """Check if this preflight check applies to the given source."""
        if "all" in self.applies_to:
            return True
        source_type = source.__class__.__name__.lower().replace("source", "")
        return source_type in self.applies_to

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Universal Checks (All Sources)
# =============================================================================


class PathAccessibilityCheck(PreflightCheck):
    """Check that path/URL is accessible."""

    name = "path_accessibility"
    category = IssueCategory.DATA_QUALITY
    description = "Verify path or URL is accessible"
    applies_to = {"all"}

    def check(self, source: "DataSource") -> list[PreflightIssue]:
        issues = []
        path = getattr(source, "path", None)

        if path is None:
            return issues

        # Handle URL-based sources
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            issues.extend(self._check_url(path))
        elif isinstance(path, str):
            issues.extend(self._check_local_path(path))

        return issues

    def _check_local_path(self, path: str) -> list[PreflightIssue]:
        """Check local file path accessibility."""
        issues = []
        path_pattern = str(Path(path).expanduser())
        matches = glob(path_pattern, recursive=True)

        if not matches:
            if "*" in path or "?" in path:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"No files match pattern: {path}",
                        suggestion=(
                            f"Check that files exist matching '{path}'. "
                            f"Verify the directory exists and glob pattern is correct."
                        ),
                        details={"pattern": path, "expanded": path_pattern},
                    )
                )
            else:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"File not found: {path}",
                        suggestion="Create the file or update the path configuration.",
                        details={"path": path},
                    )
                )
        else:
            # Check at least first file is readable
            first_match = Path(matches[0])
            if first_match.exists() and not first_match.is_file():
                if not first_match.is_dir():
                    issues.append(
                        PreflightIssue(
                            check_name=self.name,
                            category=self.category,
                            severity=IssueSeverity.ERROR,
                            message=f"Path exists but is not a regular file: {matches[0]}",
                            suggestion="Ensure path points to files, not special nodes.",
                            details={"path": matches[0]},
                        )
                    )

        return issues

    def _check_url(self, url: str) -> list[PreflightIssue]:
        """Check URL accessibility with HEAD request."""
        issues = []

        try:
            import requests

            response = requests.head(url, timeout=10, allow_redirects=True)

            if response.status_code >= 400:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"URL not accessible: {url} (HTTP {response.status_code})",
                        suggestion=(
                            "Verify the URL is correct and accessible. "
                            "Check authentication if required."
                        ),
                        details={"url": url, "status_code": response.status_code},
                    )
                )
        except ImportError:
            pass  # requests not installed, skip URL check
        except Exception as e:
            error_type = type(e).__name__
            if "Timeout" in error_type:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.WARNING,
                        message=f"URL request timed out: {url}",
                        suggestion="Check network connectivity. The remote server may be slow.",
                        details={"url": url},
                    )
                )
            else:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Cannot connect to URL: {url}",
                        suggestion=f"Check network connectivity and URL validity. Error: {e}",
                        details={"url": url, "error": str(e)},
                    )
                )

        return issues


class RequiredConfigCheck(PreflightCheck):
    """Check that required configuration fields are present."""

    name = "required_config"
    category = IssueCategory.SCHEMA
    description = "Verify required configuration fields"
    applies_to = {"all"}

    REQUIRED_FIELDS: dict[str, list[str]] = {
        "tabular": ["name", "path"],
        "raster": ["name", "path"],
        "earthengine": ["name", "collection", "bands"],
        "stac": ["name", "catalog_url", "collection", "assets"],
        "planetarycomputer": ["name", "collection", "bands"],
    }

    def check(self, source: "DataSource") -> list[PreflightIssue]:
        issues = []
        source_type = source.__class__.__name__.lower().replace("source", "")
        required = self.REQUIRED_FIELDS.get(source_type, ["name"])

        config = getattr(source, "config", {}) or {}

        for field in required:
            value = getattr(source, field, None)
            if value is None:
                value = config.get(field)

            if value is None or (isinstance(value, (list, str)) and len(value) == 0):
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Required field '{field}' is missing or empty",
                        suggestion=f"Add '{field}' to your source configuration.",
                        details={"field": field, "source_type": source_type},
                    )
                )

        return issues


# =============================================================================
# Raster-Specific Checks
# =============================================================================


class RasterFormatCheck(PreflightCheck):
    """Validate GeoTIFF/COG format and metadata."""

    name = "raster_format"
    category = IssueCategory.DATA_QUALITY
    description = "Validate raster file format and metadata"
    applies_to = {"raster"}

    # GeoTIFF magic bytes (little-endian and big-endian, standard and BigTIFF)
    TIFF_MAGIC = [b"II\x2a\x00", b"MM\x00\x2a", b"II\x2b\x00", b"MM\x00\x2b"]

    def check(self, source: "DataSource") -> list[PreflightIssue]:
        issues = []

        # Get files from source
        try:
            files = source.list_files()
        except Exception:
            files = []

        if not files:
            return issues  # PathAccessibilityCheck handles missing files

        # Check first file only for speed
        first_file = Path(files[0])

        # Magic bytes check
        issues.extend(self._check_magic_bytes(first_file))

        # Metadata check using rasterio if available
        issues.extend(self._check_metadata(source, first_file))

        return issues

    def _check_magic_bytes(self, file_path: Path) -> list[PreflightIssue]:
        """Check file starts with valid TIFF magic bytes."""
        issues = []

        try:
            with open(file_path, "rb") as f:
                header = f.read(4)

            if not any(header == magic for magic in self.TIFF_MAGIC):
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"File is not a valid GeoTIFF: {file_path.name}",
                        suggestion=(
                            "Ensure file is a GeoTIFF format. "
                            "Use 'gdalinfo' to inspect or convert with 'gdal_translate'."
                        ),
                        details={"file": str(file_path), "header_bytes": header.hex()},
                    )
                )
        except IOError as e:
            issues.append(
                PreflightIssue(
                    check_name=self.name,
                    category=self.category,
                    severity=IssueSeverity.ERROR,
                    message=f"Cannot read file: {file_path.name}",
                    suggestion=f"Check file permissions and integrity. Error: {e}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )

        return issues

    def _check_metadata(
        self, source: "DataSource", file_path: Path
    ) -> list[PreflightIssue]:
        """Check CRS, nodata, and bounds using rasterio."""
        issues = []

        try:
            import rasterio
        except ImportError:
            return issues  # Skip if rasterio not available

        try:
            with rasterio.open(file_path) as src:
                # CRS check
                if src.crs is None:
                    issues.append(
                        PreflightIssue(
                            check_name=self.name,
                            category=IssueCategory.SPATIAL,
                            severity=IssueSeverity.ERROR,
                            message=f"No CRS defined in raster: {file_path.name}",
                            suggestion=(
                                "Define a CRS using 'gdal_edit.py -a_srs EPSG:4326 file.tif' "
                                "or specify 'crs' in source configuration."
                            ),
                            details={"file": str(file_path)},
                        )
                    )

                # NoData check
                if src.nodata is None:
                    issues.append(
                        PreflightIssue(
                            check_name=self.name,
                            category=IssueCategory.DATA_QUALITY,
                            severity=IssueSeverity.WARNING,
                            message=f"No nodata value defined in raster: {file_path.name}",
                            suggestion=(
                                "Define nodata value using 'gdal_edit.py -a_nodata -9999 file.tif' "
                                "or specify 'nodata' in source configuration."
                            ),
                            details={"file": str(file_path)},
                        )
                    )

                # Band count check
                expected_band = getattr(source, "band", None)
                if expected_band is not None and expected_band > src.count:
                    issues.append(
                        PreflightIssue(
                            check_name=self.name,
                            category=IssueCategory.SCHEMA,
                            severity=IssueSeverity.ERROR,
                            message=f"Requested band {expected_band} but raster has only {src.count} band(s)",
                            suggestion=f"Change 'band' to a value between 1 and {src.count}.",
                            details={
                                "requested_band": expected_band,
                                "available_bands": src.count,
                            },
                        )
                    )

                # Bounds validity check
                bounds = src.bounds
                if not self._bounds_valid(bounds):
                    issues.append(
                        PreflightIssue(
                            check_name=self.name,
                            category=IssueCategory.SPATIAL,
                            severity=IssueSeverity.WARNING,
                            message=f"Raster bounds may be corrupted or unusual: {bounds}",
                            suggestion="Verify raster georeferencing is correct using 'gdalinfo'.",
                            details={"bounds": tuple(bounds), "file": str(file_path)},
                        )
                    )

        except Exception as e:
            if "rasterio" in str(type(e).__module__):
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Cannot open raster file: {file_path.name}",
                        suggestion=f"File may be corrupted. Try 'gdalinfo {file_path}' to diagnose. Error: {e}",
                        details={"error": str(e), "file": str(file_path)},
                    )
                )

        return issues

    def _bounds_valid(self, bounds) -> bool:
        """Check if bounds are reasonable."""
        # Check for inverted bounds
        if bounds.left > bounds.right or bounds.bottom > bounds.top:
            return False
        # Check for unreasonably large values (beyond typical projections)
        if any(
            abs(v) > 1e8
            for v in [bounds.left, bounds.right, bounds.top, bounds.bottom]
        ):
            return False
        return True


# =============================================================================
# Tabular-Specific Checks
# =============================================================================


class TabularFormatCheck(PreflightCheck):
    """Validate tabular file format and required columns."""

    name = "tabular_format"
    category = IssueCategory.SCHEMA
    description = "Validate tabular file format and columns"
    applies_to = {"tabular"}

    def check(self, source: "DataSource") -> list[PreflightIssue]:
        issues = []

        try:
            files = source.list_files()
        except Exception:
            files = []

        if not files:
            return issues

        first_file = Path(files[0])
        issues.extend(self._check_readable(source, first_file))

        return issues

    def _check_readable(
        self, source: "DataSource", file_path: Path
    ) -> list[PreflightIssue]:
        """Check file is readable and has required columns."""
        issues = []

        try:
            import pandas as pd

            suffix = file_path.suffix.lower()

            # Read only first few rows for speed
            if suffix == ".csv":
                df = pd.read_csv(file_path, nrows=5)
            elif suffix == ".parquet":
                df = pd.read_parquet(file_path).head(5)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(file_path, nrows=5)
            elif suffix == ".json":
                df = pd.read_json(file_path).head(5)
            else:
                # Try CSV as fallback
                df = pd.read_csv(file_path, nrows=5)

        except Exception as e:
            issues.append(
                PreflightIssue(
                    check_name=self.name,
                    category=self.category,
                    severity=IssueSeverity.ERROR,
                    message=f"Cannot parse file: {file_path.name}",
                    suggestion=f"Check file format and encoding. Error: {e}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )
            return issues

        # Check required columns
        lat_col = getattr(source, "lat_col", "latitude")
        lon_col = getattr(source, "lon_col", "longitude")
        geometry_col = getattr(source, "geometry_col", None)

        columns = list(df.columns)

        if geometry_col:
            if geometry_col not in columns:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Geometry column '{geometry_col}' not found",
                        suggestion=f"Available columns: {columns[:10]}{'...' if len(columns) > 10 else ''}. Specify correct 'geometry_col'.",
                        details={"expected": geometry_col, "available": columns},
                    )
                )
        else:
            if lat_col not in columns:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Latitude column '{lat_col}' not found",
                        suggestion=f"Available columns: {columns[:10]}{'...' if len(columns) > 10 else ''}. Specify correct 'lat_col'.",
                        details={"expected": lat_col, "available": columns},
                    )
                )
            if lon_col not in columns:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Longitude column '{lon_col}' not found",
                        suggestion=f"Available columns: {columns[:10]}{'...' if len(columns) > 10 else ''}. Specify correct 'lon_col'.",
                        details={"expected": lon_col, "available": columns},
                    )
                )

        return issues


class CoordinateRangeCheck(PreflightCheck):
    """Validate coordinate values are in valid ranges."""

    name = "coordinate_range"
    category = IssueCategory.SPATIAL
    description = "Check coordinates are in valid geographic ranges"
    applies_to = {"tabular"}

    def check(self, source: "DataSource") -> list[PreflightIssue]:
        issues = []

        try:
            files = source.list_files()
        except Exception:
            files = []

        if not files:
            return issues

        first_file = Path(files[0])

        try:
            import pandas as pd

            suffix = first_file.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(first_file, nrows=1000)
            elif suffix == ".parquet":
                df = pd.read_parquet(first_file).head(1000)
            else:
                return issues

        except Exception:
            return issues  # TabularFormatCheck handles parse errors

        lat_col = getattr(source, "lat_col", "latitude")
        lon_col = getattr(source, "lon_col", "longitude")

        if lat_col in df.columns:
            lat_vals = pd.to_numeric(df[lat_col], errors="coerce")
            invalid_lat = ((lat_vals < -90) | (lat_vals > 90)).sum()
            if invalid_lat > 0:
                pct = (invalid_lat / len(lat_vals)) * 100
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR if pct > 10 else IssueSeverity.WARNING,
                        message=f"{invalid_lat} rows ({pct:.1f}%) have invalid latitude (outside -90 to 90)",
                        suggestion="Check if lat/lon columns are swapped, or clean invalid values.",
                        details={"invalid_count": int(invalid_lat), "sample_size": len(lat_vals)},
                    )
                )

        if lon_col in df.columns:
            lon_vals = pd.to_numeric(df[lon_col], errors="coerce")
            invalid_lon = ((lon_vals < -180) | (lon_vals > 180)).sum()
            if invalid_lon > 0:
                pct = (invalid_lon / len(lon_vals)) * 100
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR if pct > 10 else IssueSeverity.WARNING,
                        message=f"{invalid_lon} rows ({pct:.1f}%) have invalid longitude (outside -180 to 180)",
                        suggestion="Check if lat/lon columns are swapped, or clean invalid values.",
                        details={"invalid_count": int(invalid_lon), "sample_size": len(lon_vals)},
                    )
                )

        return issues


# =============================================================================
# Remote Source Checks
# =============================================================================


class AuthenticationCheck(PreflightCheck):
    """Check authentication configuration for remote sources."""

    name = "authentication"
    category = IssueCategory.DATA_QUALITY
    description = "Verify authentication is configured"
    applies_to = {"earthengine", "planetarycomputer", "stac"}

    AUTH_ENV_VARS: dict[str, list[tuple[str, str]]] = {
        "earthengine": [
            ("GOOGLE_APPLICATION_CREDENTIALS", "service account credentials file"),
        ],
        "planetarycomputer": [
            ("PC_SDK_SUBSCRIPTION_KEY", "Planetary Computer subscription key"),
        ],
    }

    def check(self, source: "DataSource") -> list[PreflightIssue]:
        issues = []

        source_type = source.__class__.__name__.lower().replace("source", "")

        # Check for required packages
        issues.extend(self._check_packages(source_type))

        # Check environment variables
        env_vars = self.AUTH_ENV_VARS.get(source_type, [])
        for env_var, description in env_vars:
            if not os.environ.get(env_var):
                auth = getattr(source, "auth", {}) or {}
                if not auth:
                    issues.append(
                        PreflightIssue(
                            check_name=self.name,
                            category=self.category,
                            severity=IssueSeverity.WARNING,
                            message=f"Environment variable {env_var} not set",
                            suggestion=(
                                f"Set {env_var} for {description}, or configure 'auth' in source. "
                                f"Example: export {env_var}=your_value"
                            ),
                            details={"env_var": env_var, "source_type": source_type},
                        )
                    )

        # Source-specific checks
        if source_type == "stac":
            issues.extend(self._check_stac_catalog(source))

        return issues

    def _check_packages(self, source_type: str) -> list[PreflightIssue]:
        """Check required packages are installed."""
        issues = []

        package_map = {
            "earthengine": ("ee", "earthengine-api"),
            "planetarycomputer": ("planetary_computer", "planetary-computer"),
            "stac": ("pystac_client", "pystac-client"),
        }

        if source_type in package_map:
            module_name, pip_name = package_map[source_type]
            try:
                __import__(module_name)
            except ImportError:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"{pip_name} package not installed",
                        suggestion=f"Install with: pip install geopipe[remote] or pip install {pip_name}",
                        details={"package": pip_name, "source_type": source_type},
                    )
                )

        return issues

    def _check_stac_catalog(self, source: "DataSource") -> list[PreflightIssue]:
        """Test STAC catalog connectivity."""
        issues = []

        try:
            from pystac_client import Client
        except ImportError:
            return issues  # Already reported in _check_packages

        catalog_url = getattr(source, "catalog_url", None)
        if catalog_url:
            try:
                headers = getattr(source, "headers", {}) or {}
                client = Client.open(catalog_url, headers=headers)

                # Try to get collection to verify access
                collection_id = getattr(source, "collection", None)
                if collection_id:
                    try:
                        client.get_collection(collection_id)
                    except Exception:
                        issues.append(
                            PreflightIssue(
                                check_name=self.name,
                                category=self.category,
                                severity=IssueSeverity.ERROR,
                                message=f"Collection '{collection_id}' not found in catalog",
                                suggestion="Verify collection ID. Use the STAC catalog browser to find available collections.",
                                details={
                                    "catalog": catalog_url,
                                    "collection": collection_id,
                                },
                            )
                        )
            except Exception as e:
                issues.append(
                    PreflightIssue(
                        check_name=self.name,
                        category=self.category,
                        severity=IssueSeverity.ERROR,
                        message=f"Cannot connect to STAC catalog: {catalog_url}",
                        suggestion=f"Check URL and network connectivity. Error: {e}",
                        details={"catalog_url": catalog_url, "error": str(e)},
                    )
                )

        return issues


# =============================================================================
# Check Registry and Runner
# =============================================================================

DEFAULT_PREFLIGHT_CHECKS: list[PreflightCheck] = [
    PathAccessibilityCheck(),
    RequiredConfigCheck(),
    RasterFormatCheck(),
    TabularFormatCheck(),
    CoordinateRangeCheck(),
    AuthenticationCheck(),
]


def run_preflight_checks(
    source: "DataSource",
    checks: list[PreflightCheck] | None = None,
) -> PreflightResult:
    """Run preflight checks on a data source.

    Args:
        source: Data source to validate
        checks: Custom checks to run (uses defaults if None)

    Returns:
        PreflightResult with all issues found

    Example:
        >>> from geopipe.sources.raster import RasterSource
        >>> source = RasterSource("nightlights", path="data/*.tif")
        >>> result = run_preflight_checks(source)
        >>> if result.should_block:
        ...     print(result.summary())
    """
    checks = checks or DEFAULT_PREFLIGHT_CHECKS

    source_type = source.__class__.__name__.lower().replace("source", "")

    result = PreflightResult(
        source_name=source.name,
        source_type=source_type,
    )

    start = time.perf_counter()

    for check in checks:
        if not check.applies_to_source(source):
            continue

        try:
            issues = check.check(source)
            result.issues.extend(issues)
            result.checks_run.append(check.name)
        except Exception as e:
            # Don't fail entirely if a check crashes
            result.issues.append(
                PreflightIssue(
                    check_name=check.name,
                    category=IssueCategory.DATA_QUALITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Check '{check.name}' failed with error",
                    suggestion=f"This may indicate a configuration issue. Error: {e}",
                    details={"check": check.name, "error": str(e)},
                )
            )
            result.checks_run.append(check.name)

    result.duration_ms = (time.perf_counter() - start) * 1000

    # Determine overall status
    if any(i.severity == IssueSeverity.ERROR for i in result.issues):
        result.status = PreflightStatus.FAIL
    elif any(i.severity == IssueSeverity.WARNING for i in result.issues):
        result.status = PreflightStatus.WARN
    else:
        result.status = PreflightStatus.PASS

    return result


def check_data_quality(source: "DataSource") -> PreflightResult:
    """Simple API for preflight quality checks.

    Alias for run_preflight_checks() with default checks.

    Args:
        source: Data source to validate

    Returns:
        PreflightResult with all issues found

    Example:
        >>> from geopipe.sources.tabular import TabularSource
        >>> source = TabularSource("census", path="data/census.csv", lat_col="lat", lon_col="lon")
        >>> result = check_data_quality(source)
        >>> print(result.summary())
    """
    return run_preflight_checks(source)
