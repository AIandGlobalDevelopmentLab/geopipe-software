"""Generic STAC catalog data source."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd

from geopipe.sources.base import DataSource
from geopipe.sources.remote_base import RemoteSourceMixin


class STACSource(DataSource, RemoteSourceMixin):
    """
    Generic STAC catalog data source.

    Works with any STAC-compliant catalog (AWS Open Data, USGS, Element84, etc.).
    Downloads assets to local GeoTIFF files with automatic caching.

    Attributes:
        catalog_url: URL to STAC catalog or API endpoint
        collection: Collection ID within the catalog
        assets: List of asset keys to download
        bounds: Spatial bounds (minx, miny, maxx, maxy)
        center: Center coordinates (lon, lat) - alternative to bounds
        patch_size_km: Patch size in km (used with center)
        temporal_range: Time range for filtering
        resolution: Target resolution in meters
        output_dir: Directory for downloaded files
        max_items: Maximum number of items to return
        headers: Additional HTTP headers for authentication

    Example:
        >>> source = STACSource(
        ...     name="landsat",
        ...     catalog_url="https://earth-search.aws.element84.com/v1",
        ...     collection="landsat-c2-l2",
        ...     assets=["red", "green", "blue"],
        ...     bounds=(-105.5, 39.5, -105.0, 40.0),
        ...     temporal_range=("2023-01-01", "2023-12-31"),
        ...     output_dir="data/downloads",
        ... )
        >>> gdf = source.load()
    """

    def __init__(
        self,
        name: str,
        catalog_url: str,
        collection: str,
        assets: list[str],
        bounds: tuple[float, float, float, float] | None = None,
        center: tuple[float, float] | None = None,
        patch_size_km: float = 50.0,
        temporal_range: tuple[str, str] | None = None,
        resolution: float | None = None,
        output_dir: str = "data/remote",
        max_items: int = 100,
        headers: dict[str, str] | None = None,
        query: dict[str, Any] | None = None,
        path: str | None = None,  # Accepted but ignored (catalog_url used instead)
        **kwargs: Any,
    ) -> None:
        """
        Initialize STAC data source.

        Args:
            name: Unique identifier for this source
            catalog_url: URL to STAC catalog API
            collection: Collection ID to search
            assets: List of asset keys to download
            bounds: Spatial bounds (minx, miny, maxx, maxy) in WGS84
            center: Center point (lon, lat) - alternative to bounds
            patch_size_km: Size of patch in km when using center
            temporal_range: Time range as (start_date, end_date) ISO strings
            resolution: Target resolution in meters (for resampling)
            output_dir: Directory to store downloaded files
            max_items: Maximum number of STAC items to return
            headers: Additional HTTP headers for catalog requests
            query: Additional STAC query parameters
            **kwargs: Additional configuration options
        """
        # Set attributes before calling super().__init__() since it calls _validate_config
        self.catalog_url = catalog_url
        self.collection = collection
        self.assets = assets
        self.center = center
        self.patch_size_km = patch_size_km
        self.temporal_range = temporal_range
        self.resolution = resolution
        self.output_dir = output_dir
        self.max_items = max_items
        self.headers = headers or {}
        self.query = query or {}

        # Compute bounds from center if needed
        self._bounds = self._compute_bounds(center, patch_size_km, bounds)

        # Use catalog_url as path for base class compatibility
        super().__init__(name=name, path=catalog_url, **kwargs)

        # Store in config
        self.config.update({
            "type": "stac",
            "catalog_url": catalog_url,
            "collection": collection,
            "assets": assets,
            "bounds": self._bounds,
            "temporal_range": temporal_range,
            "resolution": resolution,
            "output_dir": output_dir,
        })

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.catalog_url:
            raise ValueError("catalog_url is required")
        if not self.collection:
            raise ValueError("collection is required")
        if not self.assets:
            raise ValueError("assets list is required and cannot be empty")

    def _open_catalog(self) -> Any:
        """
        Open STAC catalog client.

        Returns:
            pystac_client.Client instance
        """
        try:
            from pystac_client import Client
        except ImportError as e:
            raise ImportError(
                "pystac-client is required for STAC sources. "
                "Install with: pip install geopipe[remote]"
            ) from e

        return Client.open(self.catalog_url, headers=self.headers)

    def _search_items(
        self,
        bounds: tuple[float, float, float, float],
        temporal_range: tuple[str, str] | None,
    ) -> list[Any]:
        """
        Search catalog for matching items.

        Args:
            bounds: Spatial bounds
            temporal_range: Time range

        Returns:
            List of STAC items
        """
        catalog = self._open_catalog()

        search_params = {
            "collections": [self.collection],
            "bbox": bounds,
            "max_items": self.max_items,
        }

        if temporal_range:
            search_params["datetime"] = f"{temporal_range[0]}/{temporal_range[1]}"

        if self.query:
            search_params["query"] = self.query

        search = catalog.search(**search_params)
        items = list(search.items())

        return items

    def _download_asset(
        self,
        item: Any,
        asset_key: str,
        output_dir: Path,
    ) -> Path | None:
        """
        Download a single asset from a STAC item.

        Args:
            item: STAC item
            asset_key: Asset key to download
            output_dir: Directory to save file

        Returns:
            Path to downloaded file, or None if failed
        """
        if asset_key not in item.assets:
            warnings.warn(f"Asset '{asset_key}' not found in item {item.id}")
            return None

        asset = item.assets[asset_key]
        url = asset.href

        # Generate filename from item ID and asset key
        filename = f"{item.id}_{asset_key}.tif"
        output_path = output_dir / filename

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            return output_path

        try:
            self._download_with_progress(
                url,
                output_path,
                description=f"Downloading {asset_key}",
            )
            return output_path
        except Exception as e:
            warnings.warn(f"Failed to download {asset_key} from {item.id}: {e}")
            return None

    def _merge_assets_to_raster(
        self,
        asset_paths: list[Path],
        output_path: Path,
        bounds: tuple[float, float, float, float],
    ) -> Path:
        """
        Merge multiple asset files into a single multi-band raster.

        Args:
            asset_paths: List of paths to asset files
            output_path: Path for merged output
            bounds: Bounds to clip to

        Returns:
            Path to merged raster
        """
        try:
            import rioxarray
            import xarray as xr
        except ImportError as e:
            raise ImportError(
                "rioxarray is required for raster operations. "
                "Install with: pip install rioxarray"
            ) from e

        datasets = []
        for path in asset_paths:
            try:
                ds = rioxarray.open_rasterio(path)

                # Convert bounds to dataset CRS if needed
                ds_crs = ds.rio.crs
                if ds_crs and str(ds_crs) != "EPSG:4326":
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
                    minx, miny = transformer.transform(bounds[0], bounds[1])
                    maxx, maxy = transformer.transform(bounds[2], bounds[3])
                    clip_bounds = (minx, miny, maxx, maxy)
                else:
                    clip_bounds = bounds

                # Clip to bounds
                ds = ds.rio.clip_box(*clip_bounds)
                datasets.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {path}: {e}")

        if not datasets:
            raise ValueError("No valid raster files to merge")

        # Stack along new band dimension
        merged = xr.concat(datasets, dim="band")

        # Resample if resolution specified
        if self.resolution:
            # Calculate scale factor
            current_res = abs(float(merged.rio.resolution()[0]))
            if current_res != self.resolution:
                scale = current_res / self.resolution
                merged = merged.rio.reproject(
                    merged.rio.crs,
                    shape=(
                        int(merged.shape[-2] * scale),
                        int(merged.shape[-1] * scale),
                    ),
                )

        # Save to file
        merged.rio.to_raster(output_path)
        return output_path

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
            Path to downloaded/merged GeoTIFF file
        """
        bounds = bounds or self._bounds
        temporal_range = temporal_range or self.temporal_range

        # Check cache
        cache_key = self._compute_cache_key(
            source_type="stac",
            collection=self.collection,
            bands=self.assets,
            bounds=bounds,
            temporal_range=temporal_range,
            resolution=self.resolution or 0,
            catalog_url=self.catalog_url,
        )

        cached = self._check_cache(self.output_dir, "stac", cache_key)
        if cached:
            return cached

        # Search for items
        items = self._search_items(bounds, temporal_range)
        if not items:
            raise ValueError(
                f"No STAC items found for collection '{self.collection}' "
                f"in bounds {bounds}"
            )

        # Download assets from first (most recent) item
        # For production, might want to composite multiple items
        item = items[0]
        cache_dir = self._get_cache_dir(self.output_dir, "stac")
        asset_dir = cache_dir / "assets"
        asset_dir.mkdir(exist_ok=True)

        asset_paths = []
        for asset_key in self.assets:
            path = self._download_asset(item, asset_key, asset_dir)
            if path:
                asset_paths.append(path)

        if not asset_paths:
            raise ValueError(f"Failed to download any assets from item {item.id}")

        # Merge to single raster
        output_path = self._get_cache_path(self.output_dir, "stac", cache_key)
        self._merge_assets_to_raster(asset_paths, output_path, bounds)

        # Save metadata
        self._save_cache_metadata(
            output_path,
            {
                "source_type": "stac",
                "catalog_url": self.catalog_url,
                "collection": self.collection,
                "item_id": item.id,
                "assets": self.assets,
                "bounds": bounds,
                "temporal_range": temporal_range,
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

        # Load and convert to GeoDataFrame
        try:
            import rioxarray
        except ImportError as e:
            raise ImportError(
                "rioxarray is required for loading raster data. "
                "Install with: pip install rioxarray"
            ) from e

        ds = rioxarray.open_rasterio(raster_path)

        # Convert to GeoDataFrame with pixel geometries
        return self._raster_to_geodataframe(ds)

    def _raster_to_geodataframe(self, ds: Any) -> gpd.GeoDataFrame:
        """
        Convert xarray DataArray to GeoDataFrame.

        Args:
            ds: xarray DataArray with spatial coordinates

        Returns:
            GeoDataFrame with pixel geometries and values
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
        if len(x_coords) > 1:
            pixel_width = abs(x_coords[1] - x_coords[0])
        else:
            pixel_width = 1.0
        if len(y_coords) > 1:
            pixel_height = abs(y_coords[1] - y_coords[0])
        else:
            pixel_height = 1.0

        # Create records
        records = []
        data = ds.values

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if len(data.shape) == 3:
                    # Multi-band: data[band, y, x]
                    values = {f"band_{b+1}": data[b, i, j] for b in range(data.shape[0])}
                else:
                    # Single band: data[y, x]
                    values = {"value": data[i, j]}

                geom = box(
                    x - pixel_width / 2,
                    y - pixel_height / 2,
                    x + pixel_width / 2,
                    y + pixel_height / 2,
                )
                records.append({"geometry": geom, **values})

        gdf = gpd.GeoDataFrame(records, crs=ds.rio.crs or "EPSG:4326")
        return gdf

    def get_schema(self) -> dict[str, Any]:
        """Return schema describing this data source."""
        return {
            "name": self.name,
            "type": "stac",
            "catalog_url": self.catalog_url,
            "collection": self.collection,
            "assets": self.assets,
            "bounds": self._bounds,
            "temporal_range": self.temporal_range,
        }

    def validate(self) -> list[str]:
        """Validate that the STAC catalog is accessible."""
        issues = []

        try:
            catalog = self._open_catalog()
            # Try to get collection info
            try:
                catalog.get_collection(self.collection)
            except Exception:
                issues.append(
                    f"Collection '{self.collection}' not found in catalog"
                )
        except Exception as e:
            issues.append(f"Cannot connect to STAC catalog: {e}")

        return issues

    def list_files(self) -> list[Path]:
        """List cached files (overrides base class file listing)."""
        cache_dir = self._get_cache_dir(self.output_dir, "stac")
        return list(cache_dir.glob("*.tif"))
