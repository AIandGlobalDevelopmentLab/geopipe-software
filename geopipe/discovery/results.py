"""Discovery result model for data catalog queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from geopipe.sources.base import DataSource


CategoryType = Literal[
    "nightlights",
    "optical",
    "sar",
    "elevation",
    "climate",
    "land_cover",
    "vegetation",
    "population",
    "socioeconomic",
    "infrastructure",
]


class DiscoveryResult(BaseModel):
    """
    Result from a data discovery query.

    Represents a discovered dataset with metadata and methods to convert
    it to a usable DataSource.

    Attributes:
        dataset_id: Unique identifier for this dataset
        name: Human-readable name
        provider: Data provider (earthengine, planetary_computer, stac)
        collection: Collection identifier in the provider's catalog
        categories: Thematic categories (nightlights, optical, etc.)
        bounds: Spatial extent (minx, miny, maxx, maxy)
        temporal_range: Time coverage (start, end or None for ongoing)
        resolution_m: Spatial resolution in meters
        available_bands: List of available band names
        item_count: Number of items/scenes found
        description: Dataset description
        complementary_ids: IDs of related/complementary datasets

    Example:
        >>> result = discover(bounds=(-122, 37, -121, 38), categories=["nightlights"])[0]
        >>> source = result.to_source(name="ntl")
        >>> schema.add_source(source)
    """

    dataset_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    provider: Literal["earthengine", "planetary_computer", "stac"] = Field(
        ..., description="Data provider"
    )
    collection: str = Field(..., description="Collection identifier")
    categories: list[CategoryType] = Field(default_factory=list, description="Thematic categories")
    bounds: tuple[float, float, float, float] | None = Field(
        None, description="Spatial extent (minx, miny, maxx, maxy)"
    )
    temporal_range: tuple[str, str | None] | None = Field(
        None, description="Temporal coverage (start, end)"
    )
    resolution_m: float = Field(..., description="Spatial resolution in meters")
    available_bands: list[str] = Field(default_factory=list, description="Available bands")
    item_count: int = Field(0, description="Number of items found")
    description: str = Field("", description="Dataset description")
    complementary_ids: list[str] = Field(
        default_factory=list, description="Related dataset IDs"
    )

    def to_source(
        self,
        name: str | None = None,
        bands: list[str] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> "DataSource":
        """
        Convert this discovery result to a configured DataSource.

        Creates the appropriate source type based on the provider.

        Args:
            name: Name for the source (defaults to dataset_id)
            bands: Bands to use (defaults to all available)
            bounds: Spatial bounds (required for remote sources if not set)
            **kwargs: Additional source configuration

        Returns:
            Configured DataSource ready for fusion

        Example:
            >>> result = discover(bounds, categories=["nightlights"])[0]
            >>> source = result.to_source(name="my_ntl", bands=["avg_rad"])
        """
        source_name = name or self.dataset_id
        source_bands = bands or self.available_bands
        source_bounds = bounds or self.bounds

        if self.provider == "earthengine":
            from geopipe.sources.earthengine import EarthEngineSource

            return EarthEngineSource(
                name=source_name,
                collection=self.collection,
                bands=source_bands,
                bounds=source_bounds,
                temporal_range=self.temporal_range,
                **kwargs,
            )

        elif self.provider == "planetary_computer":
            from geopipe.sources.planetary import PlanetaryComputerSource

            return PlanetaryComputerSource(
                name=source_name,
                collection=self.collection,
                bands=source_bands,
                bounds=source_bounds,
                temporal_range=self.temporal_range,
                **kwargs,
            )

        else:  # stac
            from geopipe.sources.stac import STACSource

            # For STAC, we need to separate catalog URL from collection
            # If collection looks like a URL, use it as catalog_url
            if self.collection.startswith("http"):
                catalog_url = self.collection
                collection_id = kwargs.pop("collection_id", "default")
            else:
                catalog_url = kwargs.pop("catalog_url", "https://planetarycomputer.microsoft.com/api/stac/v1")
                collection_id = self.collection

            return STACSource(
                name=source_name,
                catalog_url=catalog_url,
                collection=collection_id,
                assets=source_bands,  # STAC uses "assets" instead of "bands"
                bounds=source_bounds,
                temporal_range=self.temporal_range,
                **kwargs,
            )

    def suggest_complementary(
        self,
        registry: Any = None,
    ) -> list["DiscoveryResult"]:
        """
        Suggest complementary datasets based on known relationships.

        Uses the known_datasets registry to find related datasets
        that work well with this one.

        Args:
            registry: CatalogRegistry instance (uses default if None)

        Returns:
            List of DiscoveryResults for complementary datasets

        Example:
            >>> viirs = discover(bounds, categories=["nightlights"])[0]
            >>> related = viirs.suggest_complementary()
            >>> print([r.name for r in related])
            ['Sentinel-2 L2A', 'Landsat Collection 2']
        """
        if registry is None:
            from geopipe.discovery.catalog import CatalogRegistry
            registry = CatalogRegistry()
            registry.load()

        results = []
        for comp_id in self.complementary_ids:
            dataset_info = registry.get(comp_id)
            if dataset_info:
                # Convert DatasetInfo to DiscoveryResult
                results.append(
                    DiscoveryResult(
                        dataset_id=dataset_info.id,
                        name=dataset_info.name,
                        provider=dataset_info.provider,
                        collection=dataset_info.collection,
                        categories=dataset_info.categories,
                        temporal_range=dataset_info.temporal_range,
                        resolution_m=dataset_info.spatial_resolution_m,
                        available_bands=dataset_info.bands,
                        description=dataset_info.description,
                        complementary_ids=dataset_info.complementary,
                    )
                )

        return results

    def summary(self) -> str:
        """Generate a brief summary of this dataset."""
        temporal = "ongoing" if self.temporal_range and self.temporal_range[1] is None else str(self.temporal_range)
        return (
            f"{self.name} ({self.provider})\n"
            f"  Resolution: {self.resolution_m}m\n"
            f"  Temporal: {temporal}\n"
            f"  Bands: {', '.join(self.available_bands[:5])}"
            + ("..." if len(self.available_bands) > 5 else "")
        )

    def __repr__(self) -> str:
        return f"DiscoveryResult(id={self.dataset_id!r}, provider={self.provider!r})"
