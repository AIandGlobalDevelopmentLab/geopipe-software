"""Microsoft Planetary Computer data source."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd

from geopipe.sources.base import DataSource
from geopipe.sources.remote_base import RemoteSourceMixin


# Planetary Computer STAC API URL
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class PlanetaryComputerSource(DataSource, RemoteSourceMixin):
    """
    Microsoft Planetary Computer STAC data source.

    Provides access to Sentinel-2, Landsat, NAIP, and other collections
    with built-in token signing for authenticated access to Azure blob storage.

    Attributes:
        collection: PC collection ID (e.g., "sentinel-2-l2a", "landsat-c2-l2")
        bands: List of band/asset names to extract
        bounds: Spatial bounds (minx, miny, maxx, maxy)
        center: Center coordinates (lon, lat) - alternative to bounds
        patch_size_km: Size of patch in km
        temporal_range: Time range for filtering
        resolution: Target resolution in meters
        cloud_cover_max: Maximum cloud cover percentage (0-100)
        output_dir: Directory for downloaded files
        stream: If True, use stackstac for lazy loading instead of download
        mosaic_method: How to combine multiple images ("first", "median", "mean")

    Example:
        >>> source = PlanetaryComputerSource(
        ...     name="sentinel2",
        ...     collection="sentinel-2-l2a",
        ...     bands=["B04", "B03", "B02", "B08"],
        ...     bounds=(-122.5, 37.5, -122.0, 38.0),
        ...     temporal_range=("2023-06-01", "2023-08-31"),
        ...     cloud_cover_max=20,
        ...     output_dir="data/downloads",
        ... )
        >>> gdf = source.load()
    """

    def __init__(
        self,
        name: str,
        collection: str,
        bands: list[str],
        bounds: tuple[float, float, float, float] | None = None,
        center: tuple[float, float] | None = None,
        patch_size_km: float = 50.0,
        temporal_range: tuple[str, str] | None = None,
        resolution: float = 10.0,
        cloud_cover_max: float = 100.0,
        output_dir: str = "data/remote",
        stream: bool = False,
        mosaic_method: Literal["first", "median", "mean"] = "first",
        path: str | None = None,  # Accepted but ignored (PC_STAC_URL used instead)
        **kwargs: Any,
    ) -> None:
        """
        Initialize Planetary Computer data source.

        Args:
            name: Unique identifier for this source
            collection: PC collection ID
            bands: List of band/asset names to extract
            bounds: Spatial bounds (minx, miny, maxx, maxy) in WGS84
            center: Center point (lon, lat) - alternative to bounds
            patch_size_km: Size of patch in km when using center
            temporal_range: Time range as (start_date, end_date) ISO strings
            resolution: Target resolution in meters
            cloud_cover_max: Maximum cloud cover percentage (0-100)
            output_dir: Directory to store downloaded files
            stream: If True, use stackstac lazy loading
            mosaic_method: How to combine multiple images
            **kwargs: Additional configuration options
        """
        # Set attributes before calling super().__init__() since it calls _validate_config
        self.collection = collection
        self.bands = bands
        self.center = center
        self.patch_size_km = patch_size_km
        self.temporal_range = temporal_range
        self.resolution = resolution
        self.cloud_cover_max = cloud_cover_max
        self.output_dir = output_dir
        self.stream = stream
        self.mosaic_method = mosaic_method

        # Compute bounds from center if needed
        self._bounds = self._compute_bounds(center, patch_size_km, bounds)

        # Use PC STAC URL as path for base class
        super().__init__(name=name, path=PC_STAC_URL, **kwargs)

        # Store in config
        self.config.update({
            "type": "planetary_computer",
            "collection": collection,
            "bands": bands,
            "bounds": self._bounds,
            "temporal_range": temporal_range,
            "resolution": resolution,
            "cloud_cover_max": cloud_cover_max,
            "output_dir": output_dir,
        })

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.collection:
            raise ValueError("collection is required")
        if not self.bands:
            raise ValueError("bands list is required and cannot be empty")
        if not 0 <= self.cloud_cover_max <= 100:
            raise ValueError("cloud_cover_max must be between 0 and 100")

    def _get_catalog(self) -> Any:
        """
        Get Planetary Computer STAC catalog client.

        Returns:
            pystac_client.Client instance with PC modifier
        """
        try:
            from pystac_client import Client
            import planetary_computer as pc
        except ImportError as e:
            raise ImportError(
                "pystac-client and planetary-computer are required. "
                "Install with: pip install geopipe[remote]"
            ) from e

        return Client.open(PC_STAC_URL, modifier=pc.sign_inplace)

    def _sign_item(self, item: Any) -> Any:
        """
        Sign a STAC item for Azure blob access.

        Args:
            item: STAC item to sign

        Returns:
            Signed STAC item
        """
        try:
            import planetary_computer as pc
        except ImportError as e:
            raise ImportError(
                "planetary-computer is required for signing. "
                "Install with: pip install geopipe[remote]"
            ) from e

        return pc.sign(item)

    def _build_query(self) -> dict[str, Any]:
        """
        Build STAC query parameters with cloud cover filter.

        Returns:
            Query dictionary for STAC search
        """
        query = {}

        # Add cloud cover filter for collections that support it
        cloud_cover_collections = [
            "sentinel-2-l2a",
            "landsat-c2-l2",
            "landsat-c2-l1",
        ]
        if self.collection in cloud_cover_collections:
            query["eo:cloud_cover"] = {"lt": self.cloud_cover_max}

        return query

    def _search_items(
        self,
        bounds: tuple[float, float, float, float],
        temporal_range: tuple[str, str] | None,
    ) -> list[Any]:
        """
        Search Planetary Computer for matching items.

        Args:
            bounds: Spatial bounds
            temporal_range: Time range

        Returns:
            List of signed STAC items
        """
        catalog = self._get_catalog()

        search_params = {
            "collections": [self.collection],
            "bbox": bounds,
            "query": self._build_query(),
        }

        if temporal_range:
            search_params["datetime"] = f"{temporal_range[0]}/{temporal_range[1]}"

        search = catalog.search(**search_params)
        items = list(search.items())

        # Sort by cloud cover if available
        try:
            items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        except Exception:
            pass

        return items

    def _download_and_merge(
        self,
        items: list[Any],
        bounds: tuple[float, float, float, float],
        output_path: Path,
    ) -> Path:
        """
        Download assets from items and merge into single raster.

        Args:
            items: List of STAC items
            bounds: Spatial bounds
            output_path: Path for output file

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

        if self.mosaic_method == "first":
            # Just use first (best) item
            items_to_use = [items[0]]
        else:
            # Use all items for compositing
            items_to_use = items[:10]  # Limit to 10 for performance

        datasets = []
        for item in items_to_use:
            for band in self.bands:
                if band not in item.assets:
                    warnings.warn(f"Band '{band}' not found in item {item.id}")
                    continue

                asset = item.assets[band]
                try:
                    ds = rioxarray.open_rasterio(asset.href)

                    # Convert bounds to dataset CRS if needed
                    from shapely.geometry import box
                    from pyproj import Transformer

                    ds_crs = ds.rio.crs
                    if ds_crs and str(ds_crs) != "EPSG:4326":
                        # Transform bounds from WGS84 to dataset CRS
                        transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
                        minx, miny = transformer.transform(bounds[0], bounds[1])
                        maxx, maxy = transformer.transform(bounds[2], bounds[3])
                        clip_bounds = (minx, miny, maxx, maxy)
                    else:
                        clip_bounds = bounds

                    ds = ds.rio.clip_box(*clip_bounds)

                    # Resample to target resolution
                    if self.resolution:
                        current_res = abs(float(ds.rio.resolution()[0]))
                        if abs(current_res - self.resolution) > 0.1:
                            # Need to resample
                            scale = current_res / self.resolution
                            new_shape = (
                                int(ds.shape[-2] * scale),
                                int(ds.shape[-1] * scale),
                            )
                            ds = ds.rio.reproject(
                                ds.rio.crs,
                                shape=new_shape,
                            )

                    datasets.append(ds)
                except Exception as e:
                    warnings.warn(f"Failed to load {band} from {item.id}: {e}")

        if not datasets:
            raise ValueError("Failed to load any bands from items")

        # Stack bands
        if len(datasets) == 1:
            merged = datasets[0]
        else:
            # If multiple bands from same image, stack them
            if self.mosaic_method == "first":
                merged = xr.concat(datasets, dim="band")
            else:
                # Composite across time
                stacked = xr.concat(datasets, dim="time")
                if self.mosaic_method == "median":
                    merged = stacked.median(dim="time")
                else:  # mean
                    merged = stacked.mean(dim="time")

        merged.rio.to_raster(output_path)
        return output_path

    def _stream_with_stackstac(
        self,
        items: list[Any],
        bounds: tuple[float, float, float, float],
    ) -> Any:
        """
        Stream data lazily using stackstac.

        Args:
            items: List of STAC items
            bounds: Spatial bounds

        Returns:
            xarray DataArray (lazily loaded)
        """
        try:
            import stackstac
        except ImportError as e:
            raise ImportError(
                "stackstac is required for streaming. "
                "Install with: pip install geopipe[remote]"
            ) from e

        stack = stackstac.stack(
            items,
            assets=self.bands,
            bounds=bounds,
            resolution=self.resolution,
            epsg=4326,
        )

        return stack

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
            source_type="planetary_computer",
            collection=self.collection,
            bands=self.bands,
            bounds=bounds,
            temporal_range=temporal_range,
            resolution=self.resolution,
            cloud_cover_max=self.cloud_cover_max,
            mosaic_method=self.mosaic_method,
        )

        cached = self._check_cache(self.output_dir, "planetary_computer", cache_key)
        if cached:
            return cached

        # Search for items
        items = self._search_items(bounds, temporal_range)
        if not items:
            raise ValueError(
                f"No items found for collection '{self.collection}' "
                f"with cloud_cover < {self.cloud_cover_max}% in bounds {bounds}"
            )

        # Download and merge
        output_path = self._get_cache_path(
            self.output_dir, "planetary_computer", cache_key
        )
        self._download_and_merge(items, bounds, output_path)

        # Save metadata
        self._save_cache_metadata(
            output_path,
            {
                "source_type": "planetary_computer",
                "collection": self.collection,
                "bands": self.bands,
                "bounds": bounds,
                "temporal_range": temporal_range,
                "cloud_cover_max": self.cloud_cover_max,
                "item_ids": [item.id for item in items[:5]],
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
        bounds = bounds or self._bounds
        temporal_range = temporal_range or self.temporal_range

        if self.stream:
            # Use stackstac for lazy loading
            items = self._search_items(bounds, temporal_range)
            if not items:
                raise ValueError(f"No items found for {self.collection}")

            ds = self._stream_with_stackstac(items, bounds)
            # Compute and convert to GeoDataFrame
            ds = ds.compute()
            return self._raster_to_geodataframe(ds)
        else:
            # Download to disk first
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
        elif "lon" in ds.coords:
            x_coords = ds.coords["lon"].values
            y_coords = ds.coords["lat"].values
        else:
            # Try to get from dims
            x_coords = ds.x.values if hasattr(ds, "x") else ds.coords[ds.dims[-1]].values
            y_coords = ds.y.values if hasattr(ds, "y") else ds.coords[ds.dims[-2]].values

        # Calculate pixel size
        pixel_width = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        pixel_height = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0

        # Create records
        records = []
        data = ds.values

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if len(data.shape) == 3:
                    values = {f"band_{b+1}": data[b, i, j] for b in range(data.shape[0])}
                elif len(data.shape) == 4:
                    # time, band, y, x - take first time step
                    values = {f"band_{b+1}": data[0, b, i, j] for b in range(data.shape[1])}
                else:
                    values = {"value": data[i, j]}

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
            "type": "planetary_computer",
            "collection": self.collection,
            "bands": self.bands,
            "bounds": self._bounds,
            "temporal_range": self.temporal_range,
            "resolution": self.resolution,
            "cloud_cover_max": self.cloud_cover_max,
        }

    def validate(self) -> list[str]:
        """Validate Planetary Computer access."""
        issues = []

        try:
            catalog = self._get_catalog()
            try:
                catalog.get_collection(self.collection)
            except Exception:
                issues.append(
                    f"Collection '{self.collection}' not found in Planetary Computer"
                )
        except Exception as e:
            issues.append(f"Cannot connect to Planetary Computer: {e}")

        return issues

    def list_files(self) -> list[Path]:
        """List cached files."""
        cache_dir = self._get_cache_dir(self.output_dir, "planetary_computer")
        return list(cache_dir.glob("*.tif"))
