"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geopipe.quality.preflight import PreflightResult

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Unique identifier for this source")
    path: str = Field(..., description="Path pattern to data files (supports globs)")
    description: str | None = Field(None, description="Human-readable description")


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    A DataSource represents a single data input that can be loaded, validated,
    and prepared for fusion with other sources. Subclasses implement specific
    loading logic for different data types (raster, tabular, vector).

    Attributes:
        name: Unique identifier for this source
        path: Path pattern to data files
        config: Full configuration dictionary

    Example:
        >>> source = RasterSource("nightlights", path="data/viirs/*.tif")
        >>> gdf = source.load(bounds=region_bounds)
    """

    def __init__(self, name: str, path: str, **kwargs: Any) -> None:
        """
        Initialize a data source.

        Args:
            name: Unique identifier for this source
            path: Path pattern to data files (supports glob patterns)
            **kwargs: Additional configuration options passed to subclass
        """
        self.name = name
        self.path = path
        self.config = {"name": name, "path": path, **kwargs}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration. Override in subclasses for custom validation."""
        pass

    @abstractmethod
    def load(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[str, str] | None = None,
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """
        Load data from this source.

        Args:
            bounds: Optional spatial bounds (minx, miny, maxx, maxy) in WGS84
            temporal_range: Optional temporal range (start_date, end_date) as ISO strings

        Returns:
            GeoDataFrame or DataFrame with loaded data
        """
        pass

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return the schema of this data source.

        Returns:
            Dictionary describing columns, types, and metadata
        """
        pass

    def list_files(self) -> list[Path]:
        """
        List all files matching the path pattern.

        Returns:
            List of Path objects for matching files
        """
        from glob import glob

        path_pattern = str(Path(self.path).expanduser())
        return [Path(p) for p in sorted(glob(path_pattern, recursive=True))]

    def validate(self) -> list[str]:
        """
        Validate that the data source is accessible and well-formed.

        Returns:
            List of warning/error messages (empty if valid)
        """
        issues = []

        files = self.list_files()
        if not files:
            issues.append(f"No files found matching pattern: {self.path}")

        return issues

    def preflight_check(self) -> "PreflightResult":
        """
        Run fast preflight validation checks.

        Validates configuration, accessibility, and metadata without
        loading the full dataset. Use before expensive load() operations.

        Returns:
            PreflightResult with issues and pass/fail status

        Example:
            >>> source = RasterSource("nightlights", path="data/*.tif")
            >>> result = source.preflight_check()
            >>> if result.should_block:
            ...     print(result.summary())
            ...     raise ValueError("Preflight check failed")
            >>> # Safe to proceed with load()
            >>> data = source.load()
        """
        from geopipe.quality.preflight import PreflightResult, run_preflight_checks

        return run_preflight_checks(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, path={self.path!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert source configuration to dictionary."""
        return self.config.copy()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DataSource":
        """
        Create a DataSource from a configuration dictionary.

        Args:
            config: Dictionary with 'type', 'name', 'path', and other options

        Returns:
            Appropriate DataSource subclass instance
        """
        source_type = config.pop("type", "tabular")
        name = config.pop("name")
        path = config.pop("path")

        # Import here to avoid circular imports
        from geopipe.sources.raster import RasterSource
        from geopipe.sources.tabular import TabularSource

        source_classes: dict[str, type] = {
            "raster": RasterSource,
            "tabular": TabularSource,
        }

        # Lazy load remote sources to avoid import errors when not installed
        if source_type == "earthengine":
            from geopipe.sources.earthengine import EarthEngineSource
            source_classes["earthengine"] = EarthEngineSource
        elif source_type == "planetary_computer":
            from geopipe.sources.planetary import PlanetaryComputerSource
            source_classes["planetary_computer"] = PlanetaryComputerSource
        elif source_type == "stac":
            from geopipe.sources.stac import STACSource
            source_classes["stac"] = STACSource

        source_cls = source_classes.get(source_type, TabularSource)
        return source_cls(name=name, path=path, **config)
