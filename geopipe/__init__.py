"""
geopipe: Geospatial data pipeline framework for causal inference workflows.

This package provides tools for:
- Declarative data fusion from multiple geospatial sources
- Pipeline orchestration with checkpointing
- HPC cluster integration (SLURM, PBS)
- Robustness specification management
- Remote data access (Earth Engine, Planetary Computer, STAC)
"""

from geopipe.sources.base import DataSource
from geopipe.sources.raster import RasterSource
from geopipe.sources.tabular import TabularSource
from geopipe.fusion.schema import FusionSchema
from geopipe.pipeline.tasks import task
from geopipe.pipeline.dag import Pipeline

# Remote sources (lazy imports to avoid dependency errors if not installed)
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

__version__ = "0.1.0"

__all__ = [
    "DataSource",
    "RasterSource",
    "TabularSource",
    "EarthEngineSource",
    "PlanetaryComputerSource",
    "STACSource",
    "FusionSchema",
    "Pipeline",
    "task",
    "__version__",
]
