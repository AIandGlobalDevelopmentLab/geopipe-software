"""Data catalog registry and discovery functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from geopipe.discovery.results import CategoryType, DiscoveryResult


console = Console()


class DatasetInfo(BaseModel):
    """
    Information about a known dataset in the registry.

    Attributes:
        id: Unique identifier (e.g., "viirs_dnb_monthly")
        name: Human-readable name
        provider: Data provider
        collection: Collection identifier in provider's catalog
        categories: Thematic categories
        temporal_range: Time range (start, end or None for ongoing)
        spatial_resolution_m: Resolution in meters
        bands: Available band names
        description: Dataset description
        complementary: IDs of related datasets
    """

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    provider: Literal["earthengine", "planetary_computer", "stac"] = Field(
        ..., description="Data provider"
    )
    collection: str = Field(..., description="Collection ID in provider")
    categories: list[CategoryType] = Field(default_factory=list)
    temporal_range: tuple[str, str | None] = Field(..., description="Start date, end date or None")
    spatial_resolution_m: float = Field(..., description="Resolution in meters")
    bands: list[str] = Field(default_factory=list)
    description: str = Field("")
    complementary: list[str] = Field(default_factory=list)


class CatalogRegistry:
    """
    Registry of known geospatial datasets.

    Manages a collection of curated dataset definitions loaded from YAML,
    enabling quick discovery without querying remote catalogs.

    Attributes:
        datasets: Dictionary mapping dataset IDs to DatasetInfo

    Example:
        >>> registry = CatalogRegistry()
        >>> registry.load()  # Load default known_datasets.yaml
        >>> viirs = registry.get("viirs_dnb_monthly")
        >>> nightlights = registry.filter(categories=["nightlights"])
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self.datasets: dict[str, DatasetInfo] = {}
        self._loaded = False

    def load(self, yaml_path: Path | str | None = None) -> None:
        """
        Load datasets from YAML file.

        Args:
            yaml_path: Path to YAML file. If None, uses the default
                known_datasets.yaml in the discovery module.
        """
        if yaml_path is None:
            yaml_path = Path(__file__).parent / "known_datasets.yaml"

        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            console.print(f"[yellow]Dataset registry not found: {yaml_path}[/yellow]")
            return

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        for ds in data.get("datasets", []):
            # Handle temporal_range as list from YAML
            tr = ds.get("temporal_range", [None, None])
            if isinstance(tr, list):
                ds["temporal_range"] = tuple(tr)

            dataset_info = DatasetInfo(**ds)
            self.datasets[dataset_info.id] = dataset_info

        self._loaded = True

    def get(self, dataset_id: str) -> DatasetInfo | None:
        """
        Get a specific dataset by ID.

        Args:
            dataset_id: Unique dataset identifier

        Returns:
            DatasetInfo if found, None otherwise
        """
        if not self._loaded:
            self.load()
        return self.datasets.get(dataset_id)

    def filter(
        self,
        categories: list[CategoryType] | None = None,
        provider: Literal["earthengine", "planetary_computer", "stac"] | None = None,
        min_resolution_m: float | None = None,
        max_resolution_m: float | None = None,
    ) -> list[DatasetInfo]:
        """
        Filter datasets by criteria.

        Args:
            categories: Filter by category (matches any)
            provider: Filter by provider
            min_resolution_m: Minimum resolution (inclusive)
            max_resolution_m: Maximum resolution (inclusive)

        Returns:
            List of matching DatasetInfo objects
        """
        if not self._loaded:
            self.load()

        results = []

        for ds in self.datasets.values():
            # Category filter
            if categories:
                if not any(cat in ds.categories for cat in categories):
                    continue

            # Provider filter
            if provider and ds.provider != provider:
                continue

            # Resolution filters
            if min_resolution_m and ds.spatial_resolution_m < min_resolution_m:
                continue
            if max_resolution_m and ds.spatial_resolution_m > max_resolution_m:
                continue

            results.append(ds)

        return results

    def list_all(self) -> list[DatasetInfo]:
        """Get all datasets in the registry."""
        if not self._loaded:
            self.load()
        return list(self.datasets.values())

    def list_categories(self) -> list[CategoryType]:
        """Get all unique categories in the registry."""
        if not self._loaded:
            self.load()

        categories = set()
        for ds in self.datasets.values():
            categories.update(ds.categories)
        return sorted(categories)

    def display(self, datasets: list[DatasetInfo] | None = None) -> None:
        """Display datasets in a rich table."""
        if datasets is None:
            datasets = self.list_all()

        table = Table(title="Known Datasets")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Provider", style="green")
        table.add_column("Resolution", justify="right")
        table.add_column("Categories", style="dim")

        for ds in datasets:
            table.add_row(
                ds.id,
                ds.name,
                ds.provider,
                f"{ds.spatial_resolution_m}m",
                ", ".join(ds.categories),
            )

        console.print(table)

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self.datasets)

    def __repr__(self) -> str:
        return f"CatalogRegistry(datasets={len(self.datasets)})"


def discover(
    bounds: tuple[float, float, float, float] | None = None,
    temporal_range: tuple[str, str] | None = None,
    categories: list[CategoryType] | None = None,
    providers: list[Literal["earthengine", "planetary_computer", "stac"]] | None = None,
    max_results: int = 50,
    query_remote: bool = False,
) -> list[DiscoveryResult]:
    """
    Discover available geospatial datasets.

    Searches the known datasets registry and optionally queries remote
    catalogs to find datasets matching the specified criteria.

    Args:
        bounds: Spatial bounds (minx, miny, maxx, maxy) in EPSG:4326
        temporal_range: Time range (start_date, end_date)
        categories: Filter by category (nightlights, optical, etc.)
        providers: Filter by provider
        max_results: Maximum number of results
        query_remote: If True, also query remote catalogs (slower)

    Returns:
        List of DiscoveryResult objects sorted by relevance

    Example:
        >>> # Find nightlights data for San Francisco
        >>> results = discover(
        ...     bounds=(-122.5, 37.7, -122.4, 37.8),
        ...     categories=["nightlights"],
        ... )
        >>> for r in results:
        ...     print(r.name, r.resolution_m)

        >>> # Convert to source for fusion
        >>> source = results[0].to_source(name="sf_ntl")
    """
    registry = CatalogRegistry()
    registry.load()

    results: list[DiscoveryResult] = []

    # Filter known datasets
    provider_filter = providers[0] if providers and len(providers) == 1 else None
    matching_datasets = registry.filter(
        categories=categories,
        provider=provider_filter,
    )

    # Filter by provider list if multiple
    if providers and len(providers) > 1:
        matching_datasets = [ds for ds in matching_datasets if ds.provider in providers]

    # Convert to DiscoveryResults
    for ds in matching_datasets:
        # Check temporal overlap if specified
        if temporal_range:
            ds_start = ds.temporal_range[0]
            ds_end = ds.temporal_range[1]

            # Skip if dataset ends before query starts
            if ds_end and ds_end < temporal_range[0]:
                continue
            # Skip if dataset starts after query ends
            if ds_start > temporal_range[1]:
                continue

        results.append(
            DiscoveryResult(
                dataset_id=ds.id,
                name=ds.name,
                provider=ds.provider,
                collection=ds.collection,
                categories=ds.categories,
                bounds=bounds,  # Use query bounds
                temporal_range=ds.temporal_range,
                resolution_m=ds.spatial_resolution_m,
                available_bands=ds.bands,
                description=ds.description,
                complementary_ids=ds.complementary,
            )
        )

    # Query remote catalogs if requested
    if query_remote and bounds:
        remote_results = _query_remote_catalogs(
            bounds=bounds,
            temporal_range=temporal_range,
            categories=categories,
            providers=providers,
        )
        results.extend(remote_results)

    # Sort by resolution (higher resolution first)
    results.sort(key=lambda r: r.resolution_m)

    return results[:max_results]


def _query_remote_catalogs(
    bounds: tuple[float, float, float, float],
    temporal_range: tuple[str, str] | None = None,
    categories: list[CategoryType] | None = None,
    providers: list[str] | None = None,
) -> list[DiscoveryResult]:
    """
    Query remote catalogs for additional datasets.

    This is a placeholder for future implementation that would
    actually query Earth Engine, Planetary Computer, and STAC catalogs.
    """
    results = []

    # Query Planetary Computer if in providers
    if providers is None or "planetary_computer" in providers:
        try:
            pc_results = _query_planetary_computer(bounds, temporal_range)
            results.extend(pc_results)
        except Exception:
            pass  # Silently skip on error

    return results


def _query_planetary_computer(
    bounds: tuple[float, float, float, float],
    temporal_range: tuple[str, str] | None = None,
) -> list[DiscoveryResult]:
    """Query Microsoft Planetary Computer catalog."""
    try:
        import pystac_client
    except ImportError:
        return []

    results = []

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )

        # Search for items
        search_params: dict[str, Any] = {
            "bbox": bounds,
            "max_items": 10,
        }

        if temporal_range:
            search_params["datetime"] = f"{temporal_range[0]}/{temporal_range[1]}"

        # Get collections instead of searching items
        for collection in catalog.get_collections():
            coll_id = collection.id

            # Extract info from collection
            results.append(
                DiscoveryResult(
                    dataset_id=f"pc_{coll_id}",
                    name=collection.title or coll_id,
                    provider="planetary_computer",
                    collection=coll_id,
                    categories=[],
                    bounds=bounds,
                    temporal_range=temporal_range,
                    resolution_m=10.0,  # Placeholder
                    available_bands=[],
                    description=collection.description or "",
                )
            )

            if len(results) >= 10:
                break

    except Exception:
        pass

    return results
