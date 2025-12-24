"""Tabular data source with spatial join capabilities."""

from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd

from geopipe.sources.base import DataSource


SpatialJoinType = Literal["nearest", "intersects", "within", "contains", "buffer_5km", "buffer_10km", "buffer_25km", "buffer_50km"]
TemporalAlignType = Literal["exact", "yearly_sum", "yearly_mean", "monthly_sum", "monthly_mean", "latest", "interpolate"]


class TabularSource(DataSource):
    """
    Data source for tabular data (CSV, Parquet) with spatial join capabilities.

    Handles loading tabular data with coordinates and provides methods for
    spatial and temporal alignment with other data sources.

    Attributes:
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        time_col: Column name for temporal information
        spatial_join: Method for spatial joining
        temporal_align: Method for temporal alignment

    Example:
        >>> source = TabularSource(
        ...     "conflict",
        ...     path="data/acled.csv",
        ...     lat_col="latitude",
        ...     lon_col="longitude",
        ...     spatial_join="buffer_10km",
        ...     temporal_align="yearly_sum",
        ... )
        >>> gdf = source.load(temporal_range=("2010-01-01", "2020-12-31"))
    """

    def __init__(
        self,
        name: str,
        path: str,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        time_col: str | None = None,
        spatial_join: SpatialJoinType = "nearest",
        temporal_align: TemporalAlignType = "exact",
        value_cols: list[str] | None = None,
        geometry_col: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a tabular data source.

        Args:
            name: Unique identifier for this source
            path: Path pattern to data files (CSV, Parquet)
            lat_col: Column name for latitude (ignored if geometry_col set)
            lon_col: Column name for longitude (ignored if geometry_col set)
            time_col: Column name for temporal information
            spatial_join: Method for spatial joining to target geometries
            temporal_align: Method for temporal alignment
            value_cols: Columns to include in output (None = all numeric)
            geometry_col: Column with WKT or geometry objects (alternative to lat/lon)
            **kwargs: Additional configuration options
        """
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.time_col = time_col
        self.spatial_join = spatial_join
        self.temporal_align = temporal_align
        self.value_cols = value_cols
        self.geometry_col = geometry_col

        super().__init__(
            name=name,
            path=path,
            lat_col=lat_col,
            lon_col=lon_col,
            time_col=time_col,
            spatial_join=spatial_join,
            temporal_align=temporal_align,
            **kwargs,
        )

    def load(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[str, str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Load tabular data within specified bounds and time range.

        Args:
            bounds: Spatial bounds (minx, miny, maxx, maxy) in WGS84
            temporal_range: Temporal range (start_date, end_date) as ISO strings

        Returns:
            GeoDataFrame with point geometries
        """
        files = self.list_files()
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {self.path}")

        dfs = []
        for file_path in files:
            df = self._load_single_file(file_path)
            if df is not None and len(df) > 0:
                dfs.append(df)

        if not dfs:
            return gpd.GeoDataFrame(
                {"geometry": []},
                crs="EPSG:4326",
            )

        df = pd.concat(dfs, ignore_index=True)

        # Convert to GeoDataFrame
        gdf = self._to_geodataframe(df)

        # Apply spatial filter
        if bounds is not None:
            gdf = self._filter_by_bounds(gdf, bounds)

        # Apply temporal filter
        if temporal_range is not None and self.time_col is not None:
            gdf = self._filter_by_time(gdf, temporal_range)

        return gdf

    def _load_single_file(self, file_path: Path) -> pd.DataFrame | None:
        """Load a single tabular file."""
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".parquet":
                return pd.read_parquet(file_path)
            elif suffix == ".csv":
                return pd.read_csv(file_path)
            elif suffix in {".xlsx", ".xls"}:
                return pd.read_excel(file_path)
            elif suffix == ".json":
                return pd.read_json(file_path)
            else:
                # Try CSV as default
                return pd.read_csv(file_path)

        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load {file_path}: {e}")
            return None

    def _to_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame with point geometries."""
        from shapely.geometry import Point
        from shapely import wkt

        if self.geometry_col is not None and self.geometry_col in df.columns:
            # Parse geometry from WKT or existing geometry
            geometries = df[self.geometry_col].apply(
                lambda x: wkt.loads(x) if isinstance(x, str) else x
            )
        elif self.lat_col in df.columns and self.lon_col in df.columns:
            # Create points from lat/lon
            geometries = [
                Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
                for lon, lat in zip(df[self.lon_col], df[self.lat_col])
            ]
        else:
            raise ValueError(
                f"Cannot create geometries. Need either '{self.geometry_col}' column "
                f"or both '{self.lat_col}' and '{self.lon_col}' columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Select columns to keep
        if self.value_cols is not None:
            cols_to_keep = [c for c in self.value_cols if c in df.columns]
        else:
            # Keep all columns except geometry-related ones
            exclude = {self.lat_col, self.lon_col, self.geometry_col}
            cols_to_keep = [c for c in df.columns if c not in exclude]

        gdf = gpd.GeoDataFrame(
            df[cols_to_keep],
            geometry=geometries,
            crs="EPSG:4326",
        )

        return gdf

    def _filter_by_bounds(
        self,
        gdf: gpd.GeoDataFrame,
        bounds: tuple[float, float, float, float],
    ) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame to spatial bounds."""
        from shapely.geometry import box

        minx, miny, maxx, maxy = bounds
        bbox = box(minx, miny, maxx, maxy)

        return gdf[gdf.geometry.intersects(bbox)].copy()

    def _filter_by_time(
        self,
        gdf: gpd.GeoDataFrame,
        temporal_range: tuple[str, str],
    ) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame to temporal range."""
        if self.time_col not in gdf.columns:
            return gdf

        start_date, end_date = temporal_range

        # Convert to datetime if needed
        time_series = pd.to_datetime(gdf[self.time_col], errors="coerce")

        mask = (time_series >= start_date) & (time_series <= end_date)
        return gdf[mask].copy()

    def aggregate_temporal(
        self,
        gdf: gpd.GeoDataFrame,
        method: TemporalAlignType | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Aggregate data temporally according to specified method.

        Args:
            gdf: GeoDataFrame with temporal data
            method: Aggregation method (uses self.temporal_align if None)

        Returns:
            Temporally aggregated GeoDataFrame
        """
        method = method or self.temporal_align

        if self.time_col is None or self.time_col not in gdf.columns:
            return gdf

        if method == "exact":
            return gdf

        # Convert time column
        gdf = gdf.copy()
        gdf["_time"] = pd.to_datetime(gdf[self.time_col], errors="coerce")

        # Determine grouping frequency
        if method in {"yearly_sum", "yearly_mean"}:
            gdf["_period"] = gdf["_time"].dt.year
        elif method in {"monthly_sum", "monthly_mean"}:
            gdf["_period"] = gdf["_time"].dt.to_period("M")
        elif method == "latest":
            return gdf.sort_values("_time").groupby("geometry", as_index=False).last()
        else:
            return gdf

        # Aggregate numeric columns
        numeric_cols = gdf.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in {"_period"}]

        agg_func = "sum" if "sum" in method else "mean"

        result = gdf.groupby(["geometry", "_period"], as_index=False)[numeric_cols].agg(agg_func)
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)

        return result.drop(columns=["_period"], errors="ignore")

    def spatial_join_to(
        self,
        target: gpd.GeoDataFrame,
        method: SpatialJoinType | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Join this source's data to target geometries.

        Args:
            target: Target GeoDataFrame to join to
            method: Join method (uses self.spatial_join if None)

        Returns:
            Target GeoDataFrame with joined values from this source
        """
        method = method or self.spatial_join

        # Load source data with bounds from target
        bounds = tuple(target.total_bounds)
        source_gdf = self.load(bounds=bounds)

        if len(source_gdf) == 0:
            return target

        # Parse buffer distance if specified
        if method.startswith("buffer_"):
            buffer_km = int(method.split("_")[1].replace("km", ""))
            return self._buffer_join(target, source_gdf, buffer_km)

        # Standard spatial joins
        if method == "nearest":
            return gpd.sjoin_nearest(target, source_gdf, how="left")
        elif method == "intersects":
            return gpd.sjoin(target, source_gdf, how="left", predicate="intersects")
        elif method == "within":
            return gpd.sjoin(target, source_gdf, how="left", predicate="within")
        elif method == "contains":
            return gpd.sjoin(target, source_gdf, how="left", predicate="contains")
        else:
            raise ValueError(f"Unknown spatial join method: {method}")

    def _buffer_join(
        self,
        target: gpd.GeoDataFrame,
        source: gpd.GeoDataFrame,
        buffer_km: float,
    ) -> gpd.GeoDataFrame:
        """Join with buffer around target geometries."""
        # Convert to projected CRS for accurate buffering
        target_proj = target.to_crs("EPSG:3857")
        source_proj = source.to_crs("EPSG:3857")

        # Buffer target geometries (convert km to meters)
        buffer_m = buffer_km * 1000
        target_buffered = target_proj.copy()
        target_buffered.geometry = target_proj.geometry.buffer(buffer_m)

        # Spatial join
        joined = gpd.sjoin(target_buffered, source_proj, how="left", predicate="intersects")

        # Aggregate if multiple matches
        numeric_cols = source.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) > 0:
            agg_dict = {col: "sum" for col in numeric_cols if col in joined.columns}
            if agg_dict:
                joined = joined.groupby(joined.index, as_index=False).agg(
                    {**agg_dict, "geometry": "first"}
                )

        # Restore original geometry
        joined = gpd.GeoDataFrame(joined, geometry="geometry", crs="EPSG:3857")
        joined = joined.to_crs("EPSG:4326")
        joined.geometry = target.geometry

        return joined

    def get_schema(self) -> dict[str, Any]:
        """Return the schema of this tabular source."""
        # Try to infer schema from first file
        files = self.list_files()
        columns = {}

        if files:
            try:
                df = self._load_single_file(files[0])
                if df is not None:
                    for col in df.columns:
                        columns[col] = {"dtype": str(df[col].dtype)}
            except Exception:
                pass

        return {
            "name": self.name,
            "type": "tabular",
            "columns": columns,
            "spatial_join": self.spatial_join,
            "temporal_align": self.temporal_align,
        }
