"""Tests for data source classes."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from geopipe.sources.base import DataSource
from geopipe.sources.tabular import TabularSource
from geopipe.sources.raster import RasterSource


class TestDataSource:
    """Tests for DataSource base class."""

    def test_from_dict_tabular(self):
        """Test creating TabularSource from dict."""
        config = {
            "type": "tabular",
            "name": "test",
            "path": "data/*.csv",
            "lat_col": "lat",
            "lon_col": "lon",
        }
        source = DataSource.from_dict(config)
        assert isinstance(source, TabularSource)
        assert source.name == "test"
        assert source.lat_col == "lat"

    def test_from_dict_raster(self):
        """Test creating RasterSource from dict."""
        config = {
            "type": "raster",
            "name": "imagery",
            "path": "data/*.tif",
            "aggregation": "mean",
        }
        source = DataSource.from_dict(config)
        assert isinstance(source, RasterSource)
        assert source.name == "imagery"
        assert source.aggregation == "mean"


class TestTabularSource:
    """Tests for TabularSource class."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file."""
        df = pd.DataFrame({
            "latitude": [40.7128, 34.0522, 41.8781],
            "longitude": [-74.0060, -118.2437, -87.6298],
            "value": [100, 200, 150],
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        })
        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_init(self, sample_csv):
        """Test TabularSource initialization."""
        source = TabularSource(
            name="test",
            path=str(sample_csv),
            lat_col="latitude",
            lon_col="longitude",
        )
        assert source.name == "test"
        assert source.lat_col == "latitude"
        assert source.lon_col == "longitude"

    def test_list_files(self, sample_csv):
        """Test listing matching files."""
        source = TabularSource(
            name="test",
            path=str(sample_csv),
        )
        files = source.list_files()
        assert len(files) == 1
        assert files[0] == sample_csv

    def test_load(self, sample_csv):
        """Test loading tabular data."""
        source = TabularSource(
            name="test",
            path=str(sample_csv),
            lat_col="latitude",
            lon_col="longitude",
        )
        gdf = source.load()
        assert len(gdf) == 3
        assert gdf.crs == "EPSG:4326"
        assert "value" in gdf.columns

    def test_load_with_bounds(self, sample_csv):
        """Test loading with spatial bounds filter."""
        source = TabularSource(
            name="test",
            path=str(sample_csv),
            lat_col="latitude",
            lon_col="longitude",
        )
        # Bounds that include only NYC
        bounds = (-75, 40, -73, 42)
        gdf = source.load(bounds=bounds)
        assert len(gdf) == 1

    def test_load_with_temporal_filter(self, sample_csv):
        """Test loading with temporal filter."""
        source = TabularSource(
            name="test",
            path=str(sample_csv),
            lat_col="latitude",
            lon_col="longitude",
            time_col="date",
        )
        gdf = source.load(temporal_range=("2020-01-15", "2020-02-15"))
        assert len(gdf) == 1

    def test_validate_missing_files(self, tmp_path):
        """Test validation with missing files."""
        source = TabularSource(
            name="test",
            path=str(tmp_path / "nonexistent/*.csv"),
        )
        issues = source.validate()
        assert len(issues) == 1
        assert "No files found" in issues[0]

    def test_get_schema(self, sample_csv):
        """Test schema retrieval."""
        source = TabularSource(
            name="test",
            path=str(sample_csv),
            spatial_join="buffer_10km",
        )
        schema = source.get_schema()
        assert schema["name"] == "test"
        assert schema["type"] == "tabular"
        assert schema["spatial_join"] == "buffer_10km"
        assert "latitude" in schema["columns"]


class TestRasterSource:
    """Tests for RasterSource class."""

    def test_init(self):
        """Test RasterSource initialization."""
        source = RasterSource(
            name="nightlights",
            path="data/*.tif",
            aggregation="mean",
            band=1,
        )
        assert source.name == "nightlights"
        assert source.aggregation == "mean"
        assert source.band == 1

    def test_invalid_aggregation(self):
        """Test validation of aggregation method."""
        with pytest.raises(ValueError, match="Invalid aggregation"):
            RasterSource(
                name="test",
                path="data/*.tif",
                aggregation="invalid",
            )

    def test_get_schema(self):
        """Test schema retrieval."""
        source = RasterSource(
            name="nightlights",
            path="data/*.tif",
            aggregation="sum",
        )
        schema = source.get_schema()
        assert schema["name"] == "nightlights"
        assert schema["type"] == "raster"
        assert schema["aggregation"] == "sum"

    def test_validate_missing_files(self, tmp_path):
        """Test validation with missing files."""
        source = RasterSource(
            name="test",
            path=str(tmp_path / "nonexistent/*.tif"),
        )
        issues = source.validate()
        assert len(issues) == 1
        assert "No files found" in issues[0]
