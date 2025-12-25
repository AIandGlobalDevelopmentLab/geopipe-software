"""Tests for geopipe discovery module."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from geopipe.discovery.catalog import CatalogRegistry, DatasetInfo, discover
from geopipe.discovery.results import DiscoveryResult, CategoryType


class TestDatasetInfo:
    """Tests for DatasetInfo model."""

    def test_create_dataset_info(self):
        """Test creating a dataset info."""
        info = DatasetInfo(
            id="test_dataset",
            name="Test Dataset",
            provider="earthengine",
            collection="TEST/COLLECTION",
            categories=["nightlights"],
            temporal_range=("2020-01-01", None),
            spatial_resolution_m=500,
            bands=["band1", "band2"],
            description="A test dataset",
            complementary=["other_dataset"],
        )

        assert info.id == "test_dataset"
        assert info.name == "Test Dataset"
        assert info.provider == "earthengine"
        assert info.collection == "TEST/COLLECTION"
        assert info.categories == ["nightlights"]
        assert info.temporal_range == ("2020-01-01", None)
        assert info.spatial_resolution_m == 500
        assert info.bands == ["band1", "band2"]
        assert info.complementary == ["other_dataset"]

    def test_dataset_info_defaults(self):
        """Test default values."""
        info = DatasetInfo(
            id="minimal",
            name="Minimal",
            provider="stac",
            collection="test",
            temporal_range=("2020-01-01", "2020-12-31"),
            spatial_resolution_m=10,
        )

        assert info.categories == []
        assert info.bands == []
        assert info.description == ""
        assert info.complementary == []


class TestCatalogRegistry:
    """Tests for CatalogRegistry."""

    def test_empty_registry(self):
        """Test creating empty registry."""
        registry = CatalogRegistry()
        assert len(registry.datasets) == 0

    def test_load_known_datasets(self):
        """Test loading the default known_datasets.yaml."""
        registry = CatalogRegistry()
        registry.load()

        # Should have loaded datasets
        assert len(registry) > 0

        # Check a known dataset exists
        viirs = registry.get("viirs_dnb_monthly")
        assert viirs is not None
        assert viirs.name == "VIIRS Day/Night Band Monthly"
        assert viirs.provider == "earthengine"

    def test_load_custom_yaml(self, tmp_path):
        """Test loading custom YAML file."""
        yaml_content = """
datasets:
  - id: custom_dataset
    name: Custom Dataset
    provider: stac
    collection: custom/collection
    categories: [optical]
    temporal_range: ["2020-01-01", null]
    spatial_resolution_m: 10
    bands: [B1, B2]
"""
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(yaml_content)

        registry = CatalogRegistry()
        registry.load(yaml_file)

        assert len(registry) == 1
        ds = registry.get("custom_dataset")
        assert ds is not None
        assert ds.name == "Custom Dataset"
        assert ds.temporal_range == ("2020-01-01", None)

    def test_get_nonexistent(self):
        """Test getting a dataset that doesn't exist."""
        registry = CatalogRegistry()
        registry.load()

        result = registry.get("nonexistent_dataset_xyz")
        assert result is None

    def test_filter_by_category(self):
        """Test filtering by category."""
        registry = CatalogRegistry()
        registry.load()

        nightlights = registry.filter(categories=["nightlights"])

        assert len(nightlights) > 0
        for ds in nightlights:
            assert "nightlights" in ds.categories

    def test_filter_by_provider(self):
        """Test filtering by provider."""
        registry = CatalogRegistry()
        registry.load()

        ee_datasets = registry.filter(provider="earthengine")

        assert len(ee_datasets) > 0
        for ds in ee_datasets:
            assert ds.provider == "earthengine"

    def test_filter_by_resolution(self):
        """Test filtering by resolution."""
        registry = CatalogRegistry()
        registry.load()

        high_res = registry.filter(max_resolution_m=50)

        for ds in high_res:
            assert ds.spatial_resolution_m <= 50

    def test_filter_combined(self):
        """Test combining multiple filters."""
        registry = CatalogRegistry()
        registry.load()

        results = registry.filter(
            categories=["optical"],
            provider="earthengine",
        )

        for ds in results:
            assert ds.provider == "earthengine"
            assert "optical" in ds.categories

    def test_list_all(self):
        """Test listing all datasets."""
        registry = CatalogRegistry()
        registry.load()

        all_datasets = registry.list_all()
        assert len(all_datasets) == len(registry)

    def test_list_categories(self):
        """Test listing available categories."""
        registry = CatalogRegistry()
        registry.load()

        categories = registry.list_categories()

        assert len(categories) > 0
        assert "nightlights" in categories
        assert "optical" in categories

    def test_repr(self):
        """Test string representation."""
        registry = CatalogRegistry()
        registry.load()

        repr_str = repr(registry)
        assert "CatalogRegistry" in repr_str


class TestDiscoveryResult:
    """Tests for DiscoveryResult."""

    def test_create_result(self):
        """Test creating a discovery result."""
        result = DiscoveryResult(
            dataset_id="test",
            name="Test Dataset",
            provider="earthengine",
            collection="TEST/COLLECTION",
            categories=["nightlights"],
            bounds=(-122.5, 37.7, -122.4, 37.8),
            temporal_range=("2020-01-01", None),
            resolution_m=500,
            available_bands=["band1"],
            item_count=100,
            description="Test description",
        )

        assert result.dataset_id == "test"
        assert result.provider == "earthengine"
        assert result.resolution_m == 500

    def test_to_source_earthengine(self):
        """Test converting to Earth Engine source."""
        result = DiscoveryResult(
            dataset_id="viirs",
            name="VIIRS",
            provider="earthengine",
            collection="NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
            bounds=(-122.5, 37.7, -122.4, 37.8),
            temporal_range=("2020-01-01", "2020-12-31"),
            resolution_m=500,
            available_bands=["avg_rad"],
        )

        source = result.to_source(name="my_ntl")

        assert source.name == "my_ntl"
        assert source.collection == "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"

    def test_to_source_planetary_computer(self):
        """Test converting to Planetary Computer source."""
        result = DiscoveryResult(
            dataset_id="sentinel2",
            name="Sentinel-2",
            provider="planetary_computer",
            collection="sentinel-2-l2a",
            bounds=(-122.5, 37.7, -122.4, 37.8),
            temporal_range=("2020-01-01", "2020-12-31"),
            resolution_m=10,
            available_bands=["B02", "B03", "B04"],
        )

        source = result.to_source(name="s2", bands=["B04"])

        assert source.name == "s2"
        assert source.collection == "sentinel-2-l2a"

    def test_to_source_stac(self):
        """Test converting to STAC source."""
        result = DiscoveryResult(
            dataset_id="custom",
            name="Custom STAC",
            provider="stac",
            collection="https://example.com/stac",
            bounds=(-122.5, 37.7, -122.4, 37.8),
            temporal_range=("2020-01-01", "2020-12-31"),
            resolution_m=30,
            available_bands=["band1"],
        )

        source = result.to_source()

        assert source.name == "custom"

    def test_summary(self):
        """Test summary generation."""
        result = DiscoveryResult(
            dataset_id="test",
            name="Test Dataset",
            provider="earthengine",
            collection="TEST/COLLECTION",
            temporal_range=("2020-01-01", None),
            resolution_m=500,
            available_bands=["b1", "b2", "b3"],
        )

        summary = result.summary()

        assert "Test Dataset" in summary
        assert "500" in summary  # Resolution could be 500 or 500.0
        assert "earthengine" in summary

    def test_repr(self):
        """Test string representation."""
        result = DiscoveryResult(
            dataset_id="test",
            name="Test",
            provider="earthengine",
            collection="TEST",
            resolution_m=500,
        )

        repr_str = repr(result)
        assert "DiscoveryResult" in repr_str
        assert "test" in repr_str


class TestDiscover:
    """Tests for discover() function."""

    def test_discover_all(self):
        """Test discovering all datasets."""
        results = discover()

        assert len(results) > 0
        for r in results:
            assert isinstance(r, DiscoveryResult)

    def test_discover_by_category(self):
        """Test filtering by category."""
        results = discover(categories=["nightlights"])

        assert len(results) > 0
        for r in results:
            assert "nightlights" in r.categories

    def test_discover_by_provider(self):
        """Test filtering by provider."""
        results = discover(providers=["earthengine"])

        assert len(results) > 0
        for r in results:
            assert r.provider == "earthengine"

    def test_discover_multiple_providers(self):
        """Test filtering by multiple providers."""
        results = discover(providers=["earthengine", "planetary_computer"])

        providers = {r.provider for r in results}
        # Should include at least one of the specified providers
        assert len(providers.intersection({"earthengine", "planetary_computer"})) > 0

    def test_discover_with_bounds(self):
        """Test discovery with bounds."""
        bounds = (-122.5, 37.7, -122.4, 37.8)
        results = discover(bounds=bounds)

        # All results should have the query bounds
        for r in results:
            assert r.bounds == bounds

    def test_discover_with_temporal_filter(self):
        """Test discovery with temporal filter."""
        # Filter for recent data
        results = discover(temporal_range=("2020-01-01", "2020-12-31"))

        # Should return datasets that overlap with this period
        assert len(results) > 0

    def test_discover_max_results(self):
        """Test limiting results."""
        results = discover(max_results=5)

        assert len(results) <= 5

    def test_discover_sorted_by_resolution(self):
        """Test that results are sorted by resolution."""
        results = discover(max_results=10)

        if len(results) > 1:
            # Should be sorted by resolution (ascending)
            for i in range(len(results) - 1):
                assert results[i].resolution_m <= results[i + 1].resolution_m


class TestDiscoveryResultComplementary:
    """Tests for complementary dataset suggestions."""

    def test_suggest_complementary(self):
        """Test suggesting complementary datasets."""
        # Get VIIRS result
        results = discover(categories=["nightlights"])
        viirs_result = None
        for r in results:
            if "viirs" in r.dataset_id.lower():
                viirs_result = r
                break

        if viirs_result and viirs_result.complementary_ids:
            complementary = viirs_result.suggest_complementary()
            # May return empty if complementary datasets not in registry
            assert isinstance(complementary, list)

    def test_suggest_complementary_empty(self):
        """Test when no complementary datasets."""
        result = DiscoveryResult(
            dataset_id="lonely",
            name="Lonely Dataset",
            provider="stac",
            collection="test",
            resolution_m=10,
            complementary_ids=[],  # No complementary
        )

        complementary = result.suggest_complementary()
        assert complementary == []


class TestCatalogRegistryDisplay:
    """Tests for display functionality."""

    def test_display_does_not_error(self):
        """Test that display runs without error."""
        registry = CatalogRegistry()
        registry.load()

        # Just verify it doesn't raise
        datasets = registry.filter(categories=["nightlights"])
        registry.display(datasets[:3])

    def test_display_empty(self):
        """Test displaying empty list."""
        registry = CatalogRegistry()
        registry.display([])


class TestCategoryType:
    """Tests for CategoryType literal."""

    def test_valid_categories(self):
        """Test that valid categories work."""
        valid = [
            "nightlights", "optical", "sar", "elevation",
            "climate", "land_cover", "vegetation", "population",
            "socioeconomic", "infrastructure",
        ]

        for cat in valid:
            result = DiscoveryResult(
                dataset_id="test",
                name="Test",
                provider="stac",
                collection="test",
                categories=[cat],  # type: ignore
                resolution_m=10,
            )
            assert cat in result.categories
