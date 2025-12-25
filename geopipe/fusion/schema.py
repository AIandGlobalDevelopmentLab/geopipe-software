"""Declarative fusion schema for multi-source data integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geopipe.quality.report import QualityReport
    from geopipe.specs.variants import Spec

import geopandas as gpd
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from geopipe.sources.base import DataSource
from geopipe.sources.raster import RasterSource
from geopipe.sources.tabular import TabularSource


console = Console()


class FusionSchema(BaseModel):
    """
    Declarative schema for fusing multiple geospatial data sources.

    A FusionSchema defines how to combine heterogeneous data sources
    (satellite imagery, tabular data, etc.) into a unified dataset.

    Attributes:
        name: Identifier for this fusion schema
        resolution: Target spatial resolution (e.g., "5km", "1deg")
        temporal_range: Time range for data (start, end)
        sources: List of data sources to fuse
        output: Output path pattern

    Example:
        >>> schema = FusionSchema(
        ...     name="aid_effects",
        ...     resolution="5km",
        ...     temporal_range=("2000-01-01", "2020-12-31"),
        ...     sources=[
        ...         RasterSource("nightlights", path="data/viirs/*.tif"),
        ...         TabularSource("conflict", path="data/acled.csv"),
        ...     ],
        ...     output="data/consolidated.parquet",
        ... )
        >>> result = schema.execute()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Identifier for this fusion schema")
    resolution: str = Field("5km", description="Target spatial resolution")
    temporal_range: tuple[str, str] | None = Field(
        None, description="Time range (start_date, end_date)"
    )
    sources: list[DataSource] = Field(
        default_factory=list, description="Data sources to fuse"
    )
    output: str = Field(..., description="Output path for fused data")
    target_crs: str = Field("EPSG:4326", description="Target coordinate reference system")
    robustness: dict[str, Any] | None = Field(
        None, description="Robustness specification block for generating variants"
    )

    def __init__(self, sources: list[DataSource] | None = None, **data: Any) -> None:
        """Initialize schema, handling source objects."""
        if sources is not None:
            # Convert source dicts to DataSource objects if needed
            processed_sources = []
            for source in sources:
                if isinstance(source, DataSource):
                    processed_sources.append(source)
                elif isinstance(source, dict):
                    processed_sources.append(DataSource.from_dict(source.copy()))
                else:
                    processed_sources.append(source)
            data["sources"] = processed_sources
        super().__init__(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FusionSchema":
        """
        Load a FusionSchema from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            FusionSchema instance
        """
        path = Path(path)
        with open(path) as f:
            config = yaml.safe_load(f)

        # Parse sources
        sources = []
        for source_config in config.pop("sources", []):
            source_type = source_config.pop("type", "tabular")
            name = source_config.pop("name")
            source_path = source_config.pop("path")

            if source_type == "raster":
                sources.append(RasterSource(name=name, path=source_path, **source_config))
            else:
                sources.append(TabularSource(name=name, path=source_path, **source_config))

        # Handle temporal_range
        if "temporal_range" in config:
            tr = config["temporal_range"]
            if isinstance(tr, list):
                config["temporal_range"] = tuple(tr)

        return cls(sources=sources, **config)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save this schema to a YAML file.

        Args:
            path: Path to write YAML configuration
        """
        config = {
            "name": self.name,
            "resolution": self.resolution,
            "temporal_range": list(self.temporal_range) if self.temporal_range else None,
            "sources": [s.to_dict() for s in self.sources],
            "output": self.output,
            "target_crs": self.target_crs,
        }

        if self.robustness:
            config["robustness"] = self.robustness

        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def validate_sources(self) -> list[str]:
        """
        Validate all data sources are accessible.

        Returns:
            List of validation issues (empty if all valid)
        """
        issues = []
        for source in self.sources:
            source_issues = source.validate()
            issues.extend([f"{source.name}: {issue}" for issue in source_issues])
        return issues

    def expand_specs(self) -> list[tuple["Spec", "FusionSchema"]]:
        """
        Expand robustness block into spec+schema pairs.

        If no robustness block is defined, returns a single "MAIN" spec
        with this schema unchanged.

        Returns:
            List of (Spec, FusionSchema) tuples for each specification

        Example:
            >>> schema = FusionSchema.from_yaml("schema.yaml")
            >>> for spec, configured_schema in schema.expand_specs():
            ...     configured_schema.execute()
        """
        from geopipe.specs.dsl import RobustnessDSL, expand_robustness_specs
        from geopipe.specs.variants import Spec

        if not self.robustness:
            # No robustness block - return single "MAIN" spec
            return [(Spec("MAIN"), self)]

        # Convert schema to dict for template substitution
        schema_dict = {
            "name": self.name,
            "resolution": self.resolution,
            "temporal_range": list(self.temporal_range) if self.temporal_range else None,
            "sources": [s.to_dict() for s in self.sources],
            "output": self.output,
            "target_crs": self.target_crs,
            "robustness": self.robustness,
        }

        specs, schema_dicts = expand_robustness_specs(schema_dict)

        # Convert schema dicts back to FusionSchema objects
        result = []
        for spec, sd in zip(specs, schema_dicts):
            # Need to create a new FusionSchema from the substituted dict
            new_schema = FusionSchema.model_validate(sd)
            result.append((spec, new_schema))

        return result

    def execute(
        self,
        target_geometries: gpd.GeoDataFrame | None = None,
        show_progress: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Execute the fusion schema to produce a unified dataset.

        Args:
            target_geometries: Optional target geometries to aggregate to.
                If None, uses the first source's geometry.
            show_progress: Whether to show progress bar

        Returns:
            Fused GeoDataFrame with all source data
        """
        if not self.sources:
            raise ValueError("No sources defined in schema")

        # Validate sources first
        issues = self.validate_sources()
        if issues:
            console.print("[yellow]Validation warnings:[/yellow]")
            for issue in issues:
                console.print(f"  - {issue}")

        # Get or create target geometries
        if target_geometries is None:
            target_geometries = self._create_target_grid()

        result = target_geometries.copy()

        # Fuse each source
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fusing data sources...", total=len(self.sources))

                for source in self.sources:
                    progress.update(task, description=f"Processing {source.name}...")
                    result = self._fuse_source(result, source)
                    progress.advance(task)
        else:
            for source in self.sources:
                result = self._fuse_source(result, source)

        # Save output
        self._save_output(result)

        return result

    def _create_target_grid(self) -> gpd.GeoDataFrame:
        """Create a target grid based on resolution specification."""
        from shapely.geometry import box
        import numpy as np

        # Parse resolution
        resolution_km = self._parse_resolution()

        # Determine bounds from first source or use global
        bounds = (-180, -60, 180, 85)  # Default global bounds

        for source in self.sources:
            try:
                files = source.list_files()
                if files:
                    # Try to get bounds from first file
                    break
            except Exception:
                continue

        minx, miny, maxx, maxy = bounds

        # Convert resolution to degrees (approximate)
        resolution_deg = resolution_km / 111.0  # ~111 km per degree at equator

        # Create grid
        x_coords = np.arange(minx, maxx, resolution_deg)
        y_coords = np.arange(miny, maxy, resolution_deg)

        geometries = []
        for y in y_coords:
            for x in x_coords:
                geom = box(x, y, x + resolution_deg, y + resolution_deg)
                geometries.append(geom)

        return gpd.GeoDataFrame(
            {"grid_id": range(len(geometries)), "geometry": geometries},
            crs=self.target_crs,
        )

    def _parse_resolution(self) -> float:
        """Parse resolution string to kilometers."""
        res = self.resolution.lower().strip()

        if res.endswith("km"):
            return float(res[:-2])
        elif res.endswith("m"):
            return float(res[:-1]) / 1000
        elif res.endswith("deg"):
            return float(res[:-3]) * 111.0
        else:
            # Assume km if no unit
            return float(res)

    def _fuse_source(
        self,
        target: gpd.GeoDataFrame,
        source: DataSource,
    ) -> gpd.GeoDataFrame:
        """Fuse a single source into the target GeoDataFrame."""
        try:
            if isinstance(source, RasterSource):
                return source.aggregate_to_geometries(target)
            elif isinstance(source, TabularSource):
                return source.spatial_join_to(target)
            else:
                # Generic load and join
                bounds = tuple(target.total_bounds)
                source_data = source.load(bounds=bounds, temporal_range=self.temporal_range)
                return gpd.sjoin_nearest(target, source_data, how="left")

        except Exception as e:
            console.print(f"[red]Error fusing {source.name}: {e}[/red]")
            return target

    def _save_output(self, result: gpd.GeoDataFrame) -> None:
        """Save the fused result to the output path."""
        output_path = Path(self.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == ".parquet":
            result.to_parquet(output_path)
        elif suffix == ".csv":
            result.to_csv(output_path, index=False)
        elif suffix in {".geojson", ".json"}:
            result.to_file(output_path, driver="GeoJSON")
        elif suffix == ".gpkg":
            result.to_file(output_path, driver="GPKG")
        else:
            # Default to Parquet
            result.to_parquet(output_path)

        console.print(f"[green]Saved fused data to {output_path}[/green]")

    def add_source(self, source: DataSource) -> "FusionSchema":
        """
        Add a data source to this schema.

        Args:
            source: DataSource to add

        Returns:
            Self for chaining
        """
        self.sources.append(source)
        return self

    def summary(self) -> str:
        """Return a summary of this fusion schema."""
        lines = [
            f"FusionSchema: {self.name}",
            f"  Resolution: {self.resolution}",
            f"  Temporal range: {self.temporal_range}",
            f"  Output: {self.output}",
            f"  Sources ({len(self.sources)}):",
        ]

        for source in self.sources:
            lines.append(f"    - {source.name} ({source.__class__.__name__})")

        return "\n".join(lines)

    def audit(
        self,
        checks: list | None = None,
        sample_size: int = 1000,
    ) -> "QualityReport":
        """
        Audit data quality before fusion.

        Runs quality checks on all sources and returns a report with
        issues, scores, and recommendations.

        Args:
            checks: Custom quality checks to run (uses defaults if None)
            sample_size: Sample size for data-level checks

        Returns:
            QualityReport with issues and scores

        Example:
            >>> schema = FusionSchema.from_yaml("schema.yaml")
            >>> report = schema.audit()
            >>> print(report.summary())
            >>> if report.has_errors:
            ...     raise ValueError("Data quality issues found")
        """
        from geopipe.quality.report import QualityReport

        return QualityReport.from_schema(self, checks=checks, sample_size=sample_size)

    def __repr__(self) -> str:
        return f"FusionSchema(name={self.name!r}, sources={len(self.sources)})"
