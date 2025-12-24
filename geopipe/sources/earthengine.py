"""Google Earth Engine data source."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd

from geopipe.sources.base import DataSource
from geopipe.sources.remote_base import RemoteSourceMixin


class EarthEngineSource(DataSource, RemoteSourceMixin):
    """
    Google Earth Engine image collection data source.

    Supports VIIRS, Landsat, Sentinel-2, MODIS, and any EE ImageCollection.
    Automatically handles authentication and exports imagery to local GeoTIFF.

    Attributes:
        collection: EE collection ID (e.g., "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        bands: List of band names to extract
        center: Center coordinates (lon, lat) for patch extraction
        bounds: Explicit bounds (minx, miny, maxx, maxy)
        patch_size_km: Size of patch in km (used with center)
        resolution: Target resolution in meters
        temporal_range: Time range for compositing
        reducer: EE reducer for temporal compositing ("mean", "median", "max", etc.)
        output_dir: Directory for downloaded files
        scale_factor: Optional scale factor to apply to values
        cloud_mask: Whether to apply cloud masking (for optical sensors)
        auth: Authentication config (credentials_file, project_id)

    Example:
        >>> source = EarthEngineSource(
        ...     name="nightlights",
        ...     collection="NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
        ...     bands=["avg_rad"],
        ...     center=(9.05, 7.49),
        ...     patch_size_km=50,
        ...     resolution=500,
        ...     temporal_range=("2020-01-01", "2020-12-31"),
        ...     output_dir="data/downloads",
        ... )
        >>> gdf = source.load()
    """

    # Map of collection IDs to their cloud mask functions
    CLOUD_MASK_COLLECTIONS = {
        "COPERNICUS/S2_SR_HARMONIZED": "_mask_s2_clouds",
        "COPERNICUS/S2_SR": "_mask_s2_clouds",
        "LANDSAT/LC08/C02/T1_L2": "_mask_landsat_clouds",
        "LANDSAT/LC09/C02/T1_L2": "_mask_landsat_clouds",
    }

    def __init__(
        self,
        name: str,
        collection: str,
        bands: list[str],
        center: tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        patch_size_km: float = 50.0,
        resolution: float = 500.0,
        temporal_range: tuple[str, str] | None = None,
        reducer: Literal["mean", "median", "max", "min", "sum", "first"] = "mean",
        output_dir: str = "data/remote",
        scale_factor: float | None = None,
        cloud_mask: bool = False,
        auth: dict[str, str] | None = None,
        path: str | None = None,  # Accepted but ignored (collection used instead)
        **kwargs: Any,
    ) -> None:
        """
        Initialize Earth Engine data source.

        Args:
            name: Unique identifier for this source
            collection: Earth Engine collection ID
            bands: List of band names to extract
            center: Center point (lon, lat) - alternative to bounds
            bounds: Spatial bounds (minx, miny, maxx, maxy) in WGS84
            patch_size_km: Size of patch in km when using center
            resolution: Target resolution in meters
            temporal_range: Time range as (start_date, end_date) ISO strings
            reducer: Temporal reducer ("mean", "median", "max", "min", "sum", "first")
            output_dir: Directory to store downloaded files
            scale_factor: Optional scale factor for band values
            cloud_mask: Apply cloud masking for supported collections
            auth: Authentication dict with "credentials_file" and/or "project_id"
            **kwargs: Additional configuration options
        """
        # Set attributes before calling super().__init__() since it calls _validate_config
        self.collection = collection
        self.bands = bands
        self.center = center
        self.patch_size_km = patch_size_km
        self.resolution = resolution
        self.temporal_range = temporal_range
        self.reducer = reducer
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.cloud_mask = cloud_mask
        self.auth = auth or {}

        # Compute bounds from center if needed
        self._bounds = self._compute_bounds(center, patch_size_km, bounds)

        # Track EE initialization state
        self._ee_initialized = False

        # Use collection ID as path for base class
        super().__init__(name=name, path=collection, **kwargs)

        # Store in config
        self.config.update({
            "type": "earthengine",
            "collection": collection,
            "bands": bands,
            "bounds": self._bounds,
            "temporal_range": temporal_range,
            "resolution": resolution,
            "reducer": reducer,
            "output_dir": output_dir,
        })

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.collection:
            raise ValueError("collection is required")
        if not self.bands:
            raise ValueError("bands list is required and cannot be empty")
        valid_reducers = {"mean", "median", "max", "min", "sum", "first"}
        if self.reducer not in valid_reducers:
            raise ValueError(f"reducer must be one of {valid_reducers}")

    def _initialize_ee(self) -> Any:
        """
        Initialize Earth Engine API.

        Returns:
            The ee module

        Raises:
            ImportError: If earthengine-api not installed
            Exception: If authentication fails
        """
        if self._ee_initialized:
            import ee
            return ee

        try:
            import ee
        except ImportError as e:
            raise ImportError(
                "earthengine-api is required for Earth Engine sources. "
                "Install with: pip install geopipe[remote]"
            ) from e

        # Try authentication methods in order
        credentials_file = self.auth.get("credentials_file") or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        project_id = self.auth.get("project_id") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )

        try:
            if credentials_file:
                # Service account authentication
                credentials = ee.ServiceAccountCredentials(None, credentials_file)
                ee.Initialize(credentials, project=project_id)
            else:
                # Try default credentials or interactive auth
                try:
                    ee.Initialize(project=project_id)
                except Exception:
                    # Fall back to interactive authentication
                    ee.Authenticate()
                    ee.Initialize(project=project_id)

            self._ee_initialized = True
            return ee

        except Exception as e:
            raise RuntimeError(
                f"Earth Engine authentication failed. "
                f"Run 'earthengine authenticate' or set GOOGLE_APPLICATION_CREDENTIALS. "
                f"Error: {e}"
            ) from e

    def _build_image_collection(self, ee: Any) -> Any:
        """
        Build filtered and processed ImageCollection.

        Args:
            ee: Earth Engine module

        Returns:
            ee.ImageCollection
        """
        # Get collection
        collection = ee.ImageCollection(self.collection)

        # Filter by time
        if self.temporal_range:
            collection = collection.filterDate(
                self.temporal_range[0], self.temporal_range[1]
            )

        # Filter by bounds
        bounds = self._bounds
        geometry = ee.Geometry.Rectangle(
            [bounds[0], bounds[1], bounds[2], bounds[3]]
        )
        collection = collection.filterBounds(geometry)

        # Apply cloud masking if requested
        if self.cloud_mask and self.collection in self.CLOUD_MASK_COLLECTIONS:
            mask_method = getattr(self, self.CLOUD_MASK_COLLECTIONS[self.collection])
            collection = collection.map(lambda img: mask_method(ee, img))

        # Select bands
        collection = collection.select(self.bands)

        # Apply scale factor if specified
        if self.scale_factor:
            collection = collection.map(
                lambda img: img.multiply(self.scale_factor)
            )

        return collection

    def _mask_s2_clouds(self, ee: Any, image: Any) -> Any:
        """
        Apply cloud mask for Sentinel-2.

        Args:
            ee: Earth Engine module
            image: ee.Image

        Returns:
            Cloud-masked image
        """
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask)

    def _mask_landsat_clouds(self, ee: Any, image: Any) -> Any:
        """
        Apply cloud mask for Landsat Collection 2.

        Args:
            ee: Earth Engine module
            image: ee.Image

        Returns:
            Cloud-masked image
        """
        qa = image.select("QA_PIXEL")
        dilated_cloud = 1 << 1
        cloud = 1 << 3
        cloud_shadow = 1 << 4
        mask = (
            qa.bitwiseAnd(dilated_cloud).eq(0)
            .And(qa.bitwiseAnd(cloud).eq(0))
            .And(qa.bitwiseAnd(cloud_shadow).eq(0))
        )
        return image.updateMask(mask)

    def _apply_reducer(self, ee: Any, collection: Any) -> Any:
        """
        Apply temporal reducer to ImageCollection.

        Args:
            ee: Earth Engine module
            collection: ee.ImageCollection

        Returns:
            ee.Image with reduced values
        """
        reducers = {
            "mean": collection.mean,
            "median": collection.median,
            "max": collection.max,
            "min": collection.min,
            "sum": collection.sum,
            "first": collection.first,
        }

        reducer_fn = reducers.get(self.reducer, collection.mean)
        return reducer_fn()

    def _export_direct(
        self,
        ee: Any,
        image: Any,
        bounds: tuple[float, float, float, float],
        output_path: Path,
    ) -> Path:
        """
        Download image directly using computePixels (for small areas).

        Args:
            ee: Earth Engine module
            image: ee.Image to download
            bounds: Spatial bounds
            output_path: Path to save file

        Returns:
            Path to downloaded file
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("numpy is required for Earth Engine downloads") from e

        geometry = ee.Geometry.Rectangle(
            [bounds[0], bounds[1], bounds[2], bounds[3]]
        )

        # Get the data as numpy array
        try:
            # Use getDownloadURL for GeoTIFF
            url = image.getDownloadURL({
                "name": "export",
                "bands": self.bands,
                "region": geometry,
                "scale": self.resolution,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            })

            self._download_with_progress(
                url,
                output_path,
                description=f"Downloading {self.name}",
            )

            return output_path

        except Exception as e:
            # If direct download fails, try computePixels
            warnings.warn(f"Direct download failed, trying computePixels: {e}")

            try:
                pixels = image.clipToBoundsAndScale(
                    geometry=geometry,
                    scale=self.resolution,
                ).getInfo()

                # This would need more processing to save as GeoTIFF
                # For now, raise an error suggesting the export approach
                raise NotImplementedError(
                    "computePixels approach not fully implemented. "
                    "Use smaller area or set up GCS export."
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to download Earth Engine data: {e2}"
                ) from e2

    def download(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[str, str] | None = None,
    ) -> Path:
        """
        Download imagery and return path to GeoTIFF.

        Args:
            bounds: Optional bounds override
            temporal_range: Optional temporal range override

        Returns:
            Path to downloaded GeoTIFF file
        """
        bounds = bounds or self._bounds
        temporal_range = temporal_range or self.temporal_range

        # Check cache
        cache_key = self._compute_cache_key(
            source_type="earthengine",
            collection=self.collection,
            bands=self.bands,
            bounds=bounds,
            temporal_range=temporal_range,
            resolution=self.resolution,
            reducer=self.reducer,
            cloud_mask=self.cloud_mask,
        )

        cached = self._check_cache(self.output_dir, "earthengine", cache_key)
        if cached:
            return cached

        # Initialize EE
        ee = self._initialize_ee()

        # Build collection and reduce
        collection = self._build_image_collection(ee)
        image = self._apply_reducer(ee, collection)

        # Download
        output_path = self._get_cache_path(self.output_dir, "earthengine", cache_key)
        self._export_direct(ee, image, bounds, output_path)

        # Save metadata
        self._save_cache_metadata(
            output_path,
            {
                "source_type": "earthengine",
                "collection": self.collection,
                "bands": self.bands,
                "bounds": bounds,
                "temporal_range": temporal_range,
                "resolution": self.resolution,
                "reducer": self.reducer,
            },
        )

        return output_path

    def load(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[str, str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Load data, downloading if necessary.

        Args:
            bounds: Optional bounds override
            temporal_range: Optional temporal range override

        Returns:
            GeoDataFrame with raster data
        """
        raster_path = self.download(bounds, temporal_range)

        try:
            import rioxarray
        except ImportError as e:
            raise ImportError(
                "rioxarray is required for loading raster data."
            ) from e

        ds = rioxarray.open_rasterio(raster_path)
        return self._raster_to_geodataframe(ds)

    def _raster_to_geodataframe(self, ds: Any) -> gpd.GeoDataFrame:
        """
        Convert xarray DataArray to GeoDataFrame.

        Args:
            ds: xarray DataArray

        Returns:
            GeoDataFrame with pixel geometries
        """
        from shapely.geometry import box

        # Get coordinates
        if "x" in ds.coords:
            x_coords = ds.coords["x"].values
            y_coords = ds.coords["y"].values
        else:
            x_coords = ds.coords["lon"].values
            y_coords = ds.coords["lat"].values

        # Calculate pixel size
        pixel_width = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        pixel_height = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0

        # Create records
        records = []
        data = ds.values

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if len(data.shape) == 3:
                    values = {
                        self.bands[b] if b < len(self.bands) else f"band_{b+1}": data[b, i, j]
                        for b in range(data.shape[0])
                    }
                else:
                    values = {self.bands[0] if self.bands else "value": data[i, j]}

                geom = box(
                    x - pixel_width / 2,
                    y - pixel_height / 2,
                    x + pixel_width / 2,
                    y + pixel_height / 2,
                )
                records.append({"geometry": geom, **values})

        gdf = gpd.GeoDataFrame(records)
        if hasattr(ds, "rio") and ds.rio.crs:
            gdf = gdf.set_crs(ds.rio.crs)
        else:
            gdf = gdf.set_crs("EPSG:4326")

        return gdf

    def get_schema(self) -> dict[str, Any]:
        """Return schema describing this data source."""
        return {
            "name": self.name,
            "type": "earthengine",
            "collection": self.collection,
            "bands": self.bands,
            "bounds": self._bounds,
            "temporal_range": self.temporal_range,
            "resolution": self.resolution,
            "reducer": self.reducer,
        }

    def validate(self) -> list[str]:
        """Validate Earth Engine authentication and collection access."""
        issues = []

        try:
            ee = self._initialize_ee()

            # Try to access the collection
            try:
                collection = ee.ImageCollection(self.collection)
                # Try to get first image to verify access
                collection.first().getInfo()
            except Exception as e:
                issues.append(f"Cannot access collection '{self.collection}': {e}")

        except Exception as e:
            issues.append(f"Earth Engine authentication failed: {e}")

        return issues

    def list_files(self) -> list[Path]:
        """List cached files."""
        cache_dir = self._get_cache_dir(self.output_dir, "earthengine")
        return list(cache_dir.glob("*.tif"))

    def list_collections(self) -> list[str]:
        """
        List common Earth Engine collections.

        Returns:
            List of commonly used collection IDs
        """
        return [
            # Nightlights
            "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
            "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS",
            # Sentinel-2
            "COPERNICUS/S2_SR_HARMONIZED",
            "COPERNICUS/S2_HARMONIZED",
            # Landsat
            "LANDSAT/LC09/C02/T1_L2",
            "LANDSAT/LC08/C02/T1_L2",
            # MODIS
            "MODIS/061/MOD13A2",  # Vegetation indices
            "MODIS/061/MOD11A2",  # Land surface temperature
            "MODIS/061/MCD12Q1",  # Land cover
            # Climate
            "ECMWF/ERA5_LAND/MONTHLY_AGGR",
            "NASA/GPM_L3/IMERG_MONTHLY_V06",  # Precipitation
            # Elevation
            "USGS/SRTMGL1_003",
            "NASA/NASADEM_HGT/001",
        ]
