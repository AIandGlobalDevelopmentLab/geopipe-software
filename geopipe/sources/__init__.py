"""Data source connectors for various geospatial data types."""

from geopipe.sources.base import DataSource
from geopipe.sources.raster import RasterSource
from geopipe.sources.tabular import TabularSource

# Remote sources (lazy imports to avoid dependency errors)
try:
    from geopipe.sources.earthengine import EarthEngineSource
except ImportError:
    EarthEngineSource = None  # type: ignore

try:
    from geopipe.sources.planetary import PlanetaryComputerSource
except ImportError:
    PlanetaryComputerSource = None  # type: ignore

try:
    from geopipe.sources.stac import STACSource
except ImportError:
    STACSource = None  # type: ignore

__all__ = [
    "DataSource",
    "RasterSource",
    "TabularSource",
    "EarthEngineSource",
    "PlanetaryComputerSource",
    "STACSource",
]
