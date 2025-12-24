"""Raster data source for satellite imagery (GeoTIFF, COG)."""

from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from geopipe.sources.base import DataSource


AggregationType = Literal["mean", "sum", "min", "max", "std", "median", "mode", "count"]


class RasterSource(DataSource):
    """
    Data source for raster data (GeoTIFF, Cloud-Optimized GeoTIFF).

    Handles loading and aggregating raster data to vector geometries. Supports
    various aggregation methods and can work with multi-band rasters.

    Attributes:
        aggregation: Method for aggregating raster values to polygons
        band: Band index to read (1-indexed, or None for all bands)
        nodata: Value to treat as nodata

    Example:
        >>> source = RasterSource(
        ...     "nightlights",
        ...     path="data/viirs/*.tif",
        ...     aggregation="mean",
        ... )
        >>> gdf = source.load(bounds=(-20, -10, 50, 40))
    """

    def __init__(
        self,
        name: str,
        path: str,
        aggregation: AggregationType = "mean",
        band: int | None = 1,
        nodata: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a raster data source.

        Args:
            name: Unique identifier for this source
            path: Path pattern to raster files (supports glob patterns)
            aggregation: Aggregation method for zonal statistics
            band: Band index to read (1-indexed), None for all bands
            nodata: Value to treat as nodata (uses file metadata if None)
            **kwargs: Additional configuration options
        """
        self.aggregation = aggregation
        self.band = band
        self.nodata = nodata
        super().__init__(name=name, path=path, aggregation=aggregation, band=band, **kwargs)

    def _validate_config(self) -> None:
        """Validate raster-specific configuration."""
        valid_aggregations = {"mean", "sum", "min", "max", "std", "median", "mode", "count"}
        if self.aggregation not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregation '{self.aggregation}'. "
                f"Must be one of: {valid_aggregations}"
            )

    def load(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[str, str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Load raster data within specified bounds.

        Args:
            bounds: Spatial bounds (minx, miny, maxx, maxy) in WGS84
            temporal_range: Not used for raster sources (temporal info from filename)

        Returns:
            GeoDataFrame with geometry and raster values
        """
        try:
            import rioxarray
            import xarray as xr
        except ImportError as e:
            raise ImportError(
                "rioxarray is required for raster loading. "
                "Install with: pip install rioxarray"
            ) from e

        files = self.list_files()
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {self.path}")

        datasets = []
        for file_path in files:
            ds = self._load_single_raster(file_path, bounds)
            if ds is not None:
                datasets.append(ds)

        if not datasets:
            return gpd.GeoDataFrame(
                {self.name: [], "geometry": []},
                crs="EPSG:4326",
            )

        # Combine datasets
        combined = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]

        # Convert to GeoDataFrame
        return self._raster_to_geodataframe(combined)

    def _load_single_raster(
        self,
        file_path: Path,
        bounds: tuple[float, float, float, float] | None,
    ) -> "xr.DataArray | None":
        """Load a single raster file, optionally clipped to bounds."""
        import rioxarray  # noqa: F401
        import xarray as xr

        try:
            da = xr.open_dataarray(file_path, engine="rasterio")

            # Select band if specified
            if self.band is not None and "band" in da.dims:
                da = da.sel(band=self.band)

            # Clip to bounds if specified
            if bounds is not None:
                minx, miny, maxx, maxy = bounds
                da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

            # Handle nodata
            if self.nodata is not None:
                da = da.where(da != self.nodata)
            elif da.rio.nodata is not None:
                da = da.where(da != da.rio.nodata)

            return da

        except Exception as e:
            # Log warning but continue with other files
            import warnings
            warnings.warn(f"Failed to load {file_path}: {e}")
            return None

    def _raster_to_geodataframe(self, da: "xr.DataArray") -> gpd.GeoDataFrame:
        """Convert xarray DataArray to GeoDataFrame with pixel geometries."""
        # Get coordinate values
        if "x" in da.dims and "y" in da.dims:
            x_coords = da.x.values
            y_coords = da.y.values
        elif "lon" in da.dims and "lat" in da.dims:
            x_coords = da.lon.values
            y_coords = da.lat.values
        else:
            raise ValueError(f"Cannot determine spatial dimensions. Found: {da.dims}")

        # Calculate pixel size
        x_res = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        y_res = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0

        # Flatten data
        values = da.values.flatten()
        valid_mask = ~np.isnan(values)

        # Create geometries (pixel centers as boxes)
        geometries = []
        x_flat = np.tile(x_coords, len(y_coords))
        y_flat = np.repeat(y_coords, len(x_coords))

        for x, y in zip(x_flat[valid_mask], y_flat[valid_mask]):
            geom = box(
                x - x_res / 2,
                y - y_res / 2,
                x + x_res / 2,
                y + y_res / 2,
            )
            geometries.append(geom)

        return gpd.GeoDataFrame(
            {self.name: values[valid_mask], "geometry": geometries},
            crs=da.rio.crs or "EPSG:4326",
        )

    def aggregate_to_geometries(
        self,
        geometries: gpd.GeoDataFrame,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Aggregate raster values to provided geometries (zonal statistics).

        Args:
            geometries: GeoDataFrame with target geometries
            bounds: Optional bounds to limit raster loading

        Returns:
            Input GeoDataFrame with added column for aggregated values
        """
        try:
            from rasterstats import zonal_stats
        except ImportError as e:
            raise ImportError(
                "rasterstats is required for zonal statistics. "
                "Install with: pip install rasterstats"
            ) from e

        files = self.list_files()
        if not files:
            geometries[self.name] = np.nan
            return geometries

        # Use first file for zonal stats (extend for multi-file later)
        raster_path = str(files[0])

        stats = zonal_stats(
            geometries,
            raster_path,
            stats=[self.aggregation],
            band=self.band or 1,
            nodata=self.nodata,
        )

        geometries = geometries.copy()
        geometries[self.name] = [s[self.aggregation] for s in stats]

        return geometries

    def get_schema(self) -> dict[str, Any]:
        """Return the schema of this raster source."""
        return {
            "name": self.name,
            "type": "raster",
            "columns": {
                self.name: {"dtype": "float64", "description": f"Raster values ({self.aggregation})"},
                "geometry": {"dtype": "geometry", "description": "Pixel or aggregation geometry"},
            },
            "aggregation": self.aggregation,
            "band": self.band,
        }
