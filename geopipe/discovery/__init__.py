"""Cross-catalog data discovery for geospatial datasets."""

from geopipe.discovery.catalog import (
    CatalogRegistry,
    DatasetInfo,
    discover,
)
from geopipe.discovery.results import (
    CategoryType,
    DiscoveryResult,
)

__all__ = [
    "CatalogRegistry",
    "DatasetInfo",
    "discover",
    "CategoryType",
    "DiscoveryResult",
]
