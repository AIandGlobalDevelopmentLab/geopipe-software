"""Tests for remote data sources (Earth Engine, Planetary Computer, STAC)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from geopipe.sources.base import DataSource
from geopipe.sources.remote_base import RemoteSourceMixin


class TestRemoteSourceMixin:
    """Tests for RemoteSourceMixin shared utilities."""

    def test_compute_bounds_from_explicit_bounds(self):
        """Test bounds passthrough when explicitly provided."""
        mixin = RemoteSourceMixin()
        bounds = (-122.5, 37.5, -122.0, 38.0)
        result = mixin._compute_bounds(
            center=None, patch_size_km=None, bounds=bounds
        )
        assert result == bounds

    def test_compute_bounds_from_center_and_patch_size(self):
        """Test bounds computation from center + patch_size."""
        mixin = RemoteSourceMixin()
        center = (0.0, 0.0)  # lon, lat at equator
        patch_size_km = 100.0

        result = mixin._compute_bounds(
            center=center, patch_size_km=patch_size_km, bounds=None
        )

        # At equator, 1 degree ≈ 111 km
        # So 100 km ≈ 0.9 degrees
        assert len(result) == 4
        minx, miny, maxx, maxy = result
        assert minx < 0 < maxx
        assert miny < 0 < maxy
        # Half of 100km should be about 0.45 degrees at equator
        assert abs(maxx - minx) == pytest.approx(0.9, rel=0.1)
        assert abs(maxy - miny) == pytest.approx(0.9, rel=0.1)

    def test_compute_bounds_error_when_missing_params(self):
        """Test error when neither bounds nor center+patch_size provided."""
        mixin = RemoteSourceMixin()
        with pytest.raises(ValueError, match="Must provide either"):
            mixin._compute_bounds(center=None, patch_size_km=None, bounds=None)

    def test_compute_bounds_error_when_partial_params(self):
        """Test error when only center provided without patch_size."""
        mixin = RemoteSourceMixin()
        with pytest.raises(ValueError, match="Must provide either"):
            mixin._compute_bounds(center=(0, 0), patch_size_km=None, bounds=None)

    def test_compute_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        mixin = RemoteSourceMixin()

        key1 = mixin._compute_cache_key(
            source_type="stac",
            collection="sentinel-2-l2a",
            bands=["B04", "B08"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2023-01-01", "2023-12-31"),
            resolution=10.0,
        )

        key2 = mixin._compute_cache_key(
            source_type="stac",
            collection="sentinel-2-l2a",
            bands=["B08", "B04"],  # Different order
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2023-01-01", "2023-12-31"),
            resolution=10.0,
        )

        # Keys should be same because bands are sorted
        assert key1 == key2
        assert len(key1) == 16  # SHA256 truncated to 16 chars

    def test_compute_cache_key_different_for_different_params(self):
        """Test that different params produce different keys."""
        mixin = RemoteSourceMixin()

        key1 = mixin._compute_cache_key(
            source_type="stac",
            collection="sentinel-2-l2a",
            bands=["B04"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2023-01-01", "2023-12-31"),
            resolution=10.0,
        )

        key2 = mixin._compute_cache_key(
            source_type="stac",
            collection="sentinel-2-l2a",
            bands=["B04"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2024-01-01", "2024-12-31"),  # Different year
            resolution=10.0,
        )

        assert key1 != key2

    def test_get_cache_dir_creates_directory(self):
        """Test cache directory creation."""
        mixin = RemoteSourceMixin()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = mixin._get_cache_dir(tmpdir, "stac")
            assert cache_dir.exists()
            assert cache_dir.is_dir()
            assert cache_dir == Path(tmpdir) / "stac"

    def test_check_cache_returns_none_when_missing(self):
        """Test cache miss returns None."""
        mixin = RemoteSourceMixin()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mixin._check_cache(tmpdir, "stac", "nonexistent")
            assert result is None

    def test_check_cache_returns_path_when_exists(self):
        """Test cache hit returns path."""
        mixin = RemoteSourceMixin()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file
            cache_dir = Path(tmpdir) / "stac"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "testkey.tif"
            cache_file.touch()

            result = mixin._check_cache(tmpdir, "stac", "testkey")
            assert result == cache_file

    def test_save_cache_metadata(self):
        """Test metadata saving alongside cache file."""
        mixin = RemoteSourceMixin()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test.tif"
            cache_path.touch()

            metadata = {"collection": "test", "bounds": [0, 0, 1, 1]}
            mixin._save_cache_metadata(cache_path, metadata)

            meta_path = cache_path.with_suffix(".json")
            assert meta_path.exists()
            with open(meta_path) as f:
                loaded = json.load(f)
            assert loaded == metadata

    def test_get_env_or_config_prefers_config(self):
        """Test config value takes precedence over env var."""
        mixin = RemoteSourceMixin()
        with patch.dict(os.environ, {"TEST_VAR": "env_value"}):
            result = mixin._get_env_or_config("config_value", "TEST_VAR")
            assert result == "config_value"

    def test_get_env_or_config_falls_back_to_env(self):
        """Test env var is used when config is None."""
        mixin = RemoteSourceMixin()
        with patch.dict(os.environ, {"TEST_VAR": "env_value"}):
            result = mixin._get_env_or_config(None, "TEST_VAR")
            assert result == "env_value"

    def test_get_env_or_config_returns_default(self):
        """Test default is returned when both config and env are missing."""
        mixin = RemoteSourceMixin()
        with patch.dict(os.environ, {}, clear=True):
            result = mixin._get_env_or_config(None, "MISSING_VAR", "default")
            assert result == "default"


class TestSTACSource:
    """Tests for STACSource class."""

    def test_init_with_bounds(self):
        """Test initialization with explicit bounds."""
        from geopipe.sources.stac import STACSource

        source = STACSource(
            name="test",
            catalog_url="https://example.com/stac",
            collection="test-collection",
            assets=["red", "green", "blue"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
        )

        assert source.name == "test"
        assert source.catalog_url == "https://example.com/stac"
        assert source.collection == "test-collection"
        assert source.assets == ["red", "green", "blue"]
        assert source._bounds == (-122.5, 37.5, -122.0, 38.0)

    def test_init_with_center_and_patch(self):
        """Test initialization with center + patch_size."""
        from geopipe.sources.stac import STACSource

        source = STACSource(
            name="test",
            catalog_url="https://example.com/stac",
            collection="test-collection",
            assets=["red"],
            center=(0.0, 0.0),
            patch_size_km=100.0,
        )

        assert source._bounds is not None
        minx, miny, maxx, maxy = source._bounds
        assert minx < 0 < maxx
        assert miny < 0 < maxy

    def test_validate_config_errors(self):
        """Test configuration validation errors."""
        from geopipe.sources.stac import STACSource

        with pytest.raises(ValueError, match="catalog_url is required"):
            STACSource(
                name="test",
                catalog_url="",
                collection="test",
                assets=["red"],
                bounds=(0, 0, 1, 1),
            )

        with pytest.raises(ValueError, match="collection is required"):
            STACSource(
                name="test",
                catalog_url="https://example.com",
                collection="",
                assets=["red"],
                bounds=(0, 0, 1, 1),
            )

        with pytest.raises(ValueError, match="assets list is required"):
            STACSource(
                name="test",
                catalog_url="https://example.com",
                collection="test",
                assets=[],
                bounds=(0, 0, 1, 1),
            )

    def test_get_schema(self):
        """Test schema export."""
        from geopipe.sources.stac import STACSource

        source = STACSource(
            name="test",
            catalog_url="https://example.com/stac",
            collection="test-collection",
            assets=["red"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2023-01-01", "2023-12-31"),
        )

        schema = source.get_schema()
        assert schema["name"] == "test"
        assert schema["type"] == "stac"
        assert schema["collection"] == "test-collection"
        assert schema["assets"] == ["red"]
        assert schema["temporal_range"] == ("2023-01-01", "2023-12-31")

    def test_list_files_returns_cached(self):
        """Test list_files returns cached files."""
        from geopipe.sources.stac import STACSource

        with tempfile.TemporaryDirectory() as tmpdir:
            source = STACSource(
                name="test",
                catalog_url="https://example.com",
                collection="test",
                assets=["red"],
                bounds=(0, 0, 1, 1),
                output_dir=tmpdir,
            )

            # Create some cache files
            cache_dir = Path(tmpdir) / "stac"
            cache_dir.mkdir(parents=True)
            (cache_dir / "file1.tif").touch()
            (cache_dir / "file2.tif").touch()

            files = source.list_files()
            assert len(files) == 2

    @patch("geopipe.sources.stac.STACSource._open_catalog")
    def test_search_items(self, mock_open_catalog):
        """Test STAC catalog search."""
        from geopipe.sources.stac import STACSource

        # Mock catalog and search
        mock_catalog = MagicMock()
        mock_search = MagicMock()
        mock_item = MagicMock()
        mock_item.id = "test-item-1"
        mock_search.items.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search
        mock_open_catalog.return_value = mock_catalog

        source = STACSource(
            name="test",
            catalog_url="https://example.com",
            collection="test",
            assets=["red"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
        )

        items = source._search_items(
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2023-01-01", "2023-12-31"),
        )

        assert len(items) == 1
        assert items[0].id == "test-item-1"
        mock_catalog.search.assert_called_once()


class TestPlanetaryComputerSource:
    """Tests for PlanetaryComputerSource class."""

    def test_init(self):
        """Test initialization."""
        from geopipe.sources.planetary import PlanetaryComputerSource

        source = PlanetaryComputerSource(
            name="test",
            collection="sentinel-2-l2a",
            bands=["B04", "B08"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
            cloud_cover_max=20,
            resolution=10,
        )

        assert source.name == "test"
        assert source.collection == "sentinel-2-l2a"
        assert source.bands == ["B04", "B08"]
        assert source.cloud_cover_max == 20
        assert source.resolution == 10

    def test_validate_config_cloud_cover(self):
        """Test cloud cover validation."""
        from geopipe.sources.planetary import PlanetaryComputerSource

        with pytest.raises(ValueError, match="cloud_cover_max must be between"):
            PlanetaryComputerSource(
                name="test",
                collection="sentinel-2-l2a",
                bands=["B04"],
                bounds=(0, 0, 1, 1),
                cloud_cover_max=150,  # Invalid
            )

    def test_build_query_includes_cloud_cover(self):
        """Test cloud cover filter is included in query."""
        from geopipe.sources.planetary import PlanetaryComputerSource

        source = PlanetaryComputerSource(
            name="test",
            collection="sentinel-2-l2a",
            bands=["B04"],
            bounds=(0, 0, 1, 1),
            cloud_cover_max=30,
        )

        query = source._build_query()
        assert "eo:cloud_cover" in query
        assert query["eo:cloud_cover"] == {"lt": 30}

    def test_get_schema(self):
        """Test schema export."""
        from geopipe.sources.planetary import PlanetaryComputerSource

        source = PlanetaryComputerSource(
            name="sentinel",
            collection="sentinel-2-l2a",
            bands=["B04", "B08"],
            bounds=(-122.5, 37.5, -122.0, 38.0),
            temporal_range=("2023-06-01", "2023-08-31"),
            resolution=10,
        )

        schema = source.get_schema()
        assert schema["name"] == "sentinel"
        assert schema["type"] == "planetary_computer"
        assert schema["collection"] == "sentinel-2-l2a"
        assert schema["bands"] == ["B04", "B08"]


class TestEarthEngineSource:
    """Tests for EarthEngineSource class."""

    def test_init(self):
        """Test initialization."""
        from geopipe.sources.earthengine import EarthEngineSource

        source = EarthEngineSource(
            name="viirs",
            collection="NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
            bands=["avg_rad"],
            center=(9.05, 7.49),
            patch_size_km=50,
            resolution=500,
            temporal_range=("2020-01-01", "2020-12-31"),
            reducer="mean",
        )

        assert source.name == "viirs"
        assert source.collection == "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
        assert source.bands == ["avg_rad"]
        assert source.resolution == 500
        assert source.reducer == "mean"

    def test_validate_config_invalid_reducer(self):
        """Test invalid reducer validation."""
        from geopipe.sources.earthengine import EarthEngineSource

        with pytest.raises(ValueError, match="reducer must be one of"):
            EarthEngineSource(
                name="test",
                collection="test",
                bands=["band1"],
                bounds=(0, 0, 1, 1),
                reducer="invalid",
            )

    def test_get_schema(self):
        """Test schema export."""
        from geopipe.sources.earthengine import EarthEngineSource

        source = EarthEngineSource(
            name="viirs",
            collection="NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
            bands=["avg_rad"],
            bounds=(-10, -5, 10, 5),
            resolution=500,
        )

        schema = source.get_schema()
        assert schema["name"] == "viirs"
        assert schema["type"] == "earthengine"
        assert schema["collection"] == "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"

    def test_list_collections(self):
        """Test listing common collections."""
        from geopipe.sources.earthengine import EarthEngineSource

        source = EarthEngineSource(
            name="test",
            collection="test",
            bands=["band1"],
            bounds=(0, 0, 1, 1),
        )

        collections = source.list_collections()
        assert "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG" in collections
        assert "COPERNICUS/S2_SR_HARMONIZED" in collections
        assert "LANDSAT/LC09/C02/T1_L2" in collections


class TestDataSourceFactory:
    """Tests for DataSource.from_dict factory with remote sources."""

    def test_from_dict_creates_stac_source(self):
        """Test factory creates STACSource from dict."""
        config = {
            "type": "stac",
            "name": "landsat",
            "path": "https://example.com/stac",
            "catalog_url": "https://example.com/stac",
            "collection": "landsat-c2-l2",
            "assets": ["red", "green", "blue"],
            "bounds": (-105.5, 39.5, -105.0, 40.0),
        }

        source = DataSource.from_dict(config.copy())

        from geopipe.sources.stac import STACSource
        assert isinstance(source, STACSource)
        assert source.name == "landsat"
        assert source.collection == "landsat-c2-l2"

    def test_from_dict_creates_planetary_computer_source(self):
        """Test factory creates PlanetaryComputerSource from dict."""
        config = {
            "type": "planetary_computer",
            "name": "sentinel",
            "path": "pc",
            "collection": "sentinel-2-l2a",
            "bands": ["B04", "B08"],
            "bounds": (-122.5, 37.5, -122.0, 38.0),
        }

        source = DataSource.from_dict(config.copy())

        from geopipe.sources.planetary import PlanetaryComputerSource
        assert isinstance(source, PlanetaryComputerSource)
        assert source.collection == "sentinel-2-l2a"

    def test_from_dict_creates_earthengine_source(self):
        """Test factory creates EarthEngineSource from dict."""
        config = {
            "type": "earthengine",
            "name": "viirs",
            "path": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
            "collection": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG",
            "bands": ["avg_rad"],
            "bounds": (-10, -5, 10, 5),
            "resolution": 500,
        }

        source = DataSource.from_dict(config.copy())

        from geopipe.sources.earthengine import EarthEngineSource
        assert isinstance(source, EarthEngineSource)
        assert source.collection == "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"


# Integration tests (skipped by default)
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_REMOTE_TESTS"),
    reason="Remote integration tests disabled (set RUN_REMOTE_TESTS=1 to enable)"
)
class TestRemoteIntegration:
    """Integration tests requiring network access."""

    def test_stac_public_catalog_search(self):
        """Test search against real public STAC catalog."""
        from geopipe.sources.stac import STACSource

        source = STACSource(
            name="test",
            catalog_url="https://earth-search.aws.element84.com/v1",
            collection="sentinel-2-l2a",
            assets=["red"],
            bounds=(-122.5, 37.5, -122.4, 37.6),
            temporal_range=("2023-06-01", "2023-06-30"),
            max_items=1,
        )

        # Just test that search works
        items = source._search_items(
            bounds=source._bounds,
            temporal_range=source.temporal_range,
        )
        assert len(items) > 0

    def test_stac_validate_public_catalog(self):
        """Test validation against public catalog."""
        from geopipe.sources.stac import STACSource

        source = STACSource(
            name="test",
            catalog_url="https://earth-search.aws.element84.com/v1",
            collection="sentinel-2-l2a",
            assets=["red"],
            bounds=(-122.5, 37.5, -122.4, 37.6),
        )

        issues = source.validate()
        assert len(issues) == 0  # Should validate successfully
