"""Tests for fusion schema."""

import tempfile
from pathlib import Path

import pytest
import yaml

from geopipe.fusion.schema import FusionSchema
from geopipe.sources.raster import RasterSource
from geopipe.sources.tabular import TabularSource


class TestFusionSchema:
    """Tests for FusionSchema class."""

    def test_init_with_sources(self):
        """Test initialization with source objects."""
        schema = FusionSchema(
            name="test",
            resolution="5km",
            sources=[
                RasterSource("raster1", path="data/*.tif"),
                TabularSource("table1", path="data/*.csv"),
            ],
            output="output.parquet",
        )
        assert schema.name == "test"
        assert schema.resolution == "5km"
        assert len(schema.sources) == 2

    def test_init_with_dicts(self):
        """Test initialization with source dicts."""
        schema = FusionSchema(
            name="test",
            resolution="5km",
            sources=[
                {"type": "raster", "name": "raster1", "path": "data/*.tif"},
                {"type": "tabular", "name": "table1", "path": "data/*.csv"},
            ],
            output="output.parquet",
        )
        assert len(schema.sources) == 2
        assert isinstance(schema.sources[0], RasterSource)
        assert isinstance(schema.sources[1], TabularSource)

    def test_from_yaml(self, tmp_path):
        """Test loading schema from YAML."""
        yaml_content = """
name: test_fusion
resolution: 10km
temporal_range:
  - "2010-01-01"
  - "2020-12-31"

sources:
  - type: raster
    name: nightlights
    path: data/viirs/*.tif
    aggregation: mean

  - type: tabular
    name: conflict
    path: data/acled.csv
    lat_col: latitude
    lon_col: longitude
    spatial_join: buffer_10km

output: output/fused.parquet
"""
        yaml_path = tmp_path / "schema.yaml"
        yaml_path.write_text(yaml_content)

        schema = FusionSchema.from_yaml(yaml_path)
        assert schema.name == "test_fusion"
        assert schema.resolution == "10km"
        assert schema.temporal_range == ("2010-01-01", "2020-12-31")
        assert len(schema.sources) == 2
        assert schema.output == "output/fused.parquet"

    def test_to_yaml(self, tmp_path):
        """Test saving schema to YAML."""
        schema = FusionSchema(
            name="test",
            resolution="5km",
            temporal_range=("2015-01-01", "2020-12-31"),
            sources=[
                RasterSource("raster1", path="data/*.tif"),
            ],
            output="output.parquet",
        )

        yaml_path = tmp_path / "output_schema.yaml"
        schema.to_yaml(yaml_path)

        assert yaml_path.exists()

        # Reload and verify
        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == "test"
        assert loaded["resolution"] == "5km"
        assert len(loaded["sources"]) == 1

    def test_add_source(self):
        """Test adding sources."""
        schema = FusionSchema(
            name="test",
            output="output.parquet",
        )
        schema.add_source(RasterSource("raster1", path="*.tif"))
        schema.add_source(TabularSource("table1", path="*.csv"))

        assert len(schema.sources) == 2

    def test_validate_sources_missing_files(self, tmp_path):
        """Test validation with missing files."""
        schema = FusionSchema(
            name="test",
            sources=[
                RasterSource("missing", path=str(tmp_path / "nonexistent/*.tif")),
            ],
            output="output.parquet",
        )
        issues = schema.validate_sources()
        assert len(issues) == 1
        assert "No files found" in issues[0]

    def test_summary(self):
        """Test summary generation."""
        schema = FusionSchema(
            name="test_schema",
            resolution="5km",
            temporal_range=("2010-01-01", "2020-12-31"),
            sources=[
                RasterSource("nightlights", path="*.tif"),
                TabularSource("conflict", path="*.csv"),
            ],
            output="output.parquet",
        )
        summary = schema.summary()

        assert "test_schema" in summary
        assert "5km" in summary
        assert "nightlights" in summary
        assert "conflict" in summary

    def test_parse_resolution(self):
        """Test resolution parsing."""
        schema = FusionSchema(name="test", output="out.parquet")

        schema.resolution = "5km"
        assert schema._parse_resolution() == 5.0

        schema.resolution = "1000m"
        assert schema._parse_resolution() == 1.0

        schema.resolution = "1deg"
        assert schema._parse_resolution() == 111.0

        schema.resolution = "10"
        assert schema._parse_resolution() == 10.0
