"""Tests for preflight quality checks."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from geopipe.quality.preflight import (
    PreflightStatus,
    PreflightIssue,
    PreflightResult,
    PreflightCheck,
    PathAccessibilityCheck,
    RequiredConfigCheck,
    RasterFormatCheck,
    TabularFormatCheck,
    CoordinateRangeCheck,
    AuthenticationCheck,
    DEFAULT_PREFLIGHT_CHECKS,
    run_preflight_checks,
    check_data_quality,
)
from geopipe.quality.checks import IssueCategory, IssueSeverity


class TestPreflightIssue:
    """Tests for PreflightIssue model."""

    def test_creation(self):
        issue = PreflightIssue(
            check_name="test_check",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.ERROR,
            message="Test error message",
            suggestion="Fix the error",
        )
        assert issue.check_name == "test_check"
        assert issue.severity == IssueSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.suggestion == "Fix the error"

    def test_str_representation(self):
        issue = PreflightIssue(
            check_name="test",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.ERROR,
            message="Error message",
            suggestion="Fix it",
        )
        result = str(issue)
        assert "[X]" in result
        assert "Error message" in result
        assert "Fix it" in result

    def test_warning_icon(self):
        issue = PreflightIssue(
            check_name="test",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.WARNING,
            message="Warning message",
            suggestion="Consider fixing",
        )
        result = str(issue)
        assert "[!]" in result

    def test_info_icon(self):
        issue = PreflightIssue(
            check_name="test",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.INFO,
            message="Info message",
            suggestion="FYI",
        )
        result = str(issue)
        assert "[i]" in result


class TestPreflightResult:
    """Tests for PreflightResult model."""

    def test_empty_result_passes(self):
        result = PreflightResult(
            source_name="test",
            source_type="tabular",
        )
        assert result.status == PreflightStatus.PASS
        assert result.should_block is False
        assert result.has_warnings is False
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_warning_does_not_block(self):
        result = PreflightResult(
            source_name="test",
            source_type="tabular",
            status=PreflightStatus.WARN,
            issues=[
                PreflightIssue(
                    check_name="test",
                    category=IssueCategory.DATA_QUALITY,
                    severity=IssueSeverity.WARNING,
                    message="Warning",
                    suggestion="Fix it",
                )
            ],
        )
        assert result.should_block is False
        assert result.has_warnings is True
        assert len(result.warnings) == 1
        assert len(result.errors) == 0

    def test_error_blocks(self):
        result = PreflightResult(
            source_name="test",
            source_type="tabular",
            status=PreflightStatus.FAIL,
            issues=[
                PreflightIssue(
                    check_name="test",
                    category=IssueCategory.DATA_QUALITY,
                    severity=IssueSeverity.ERROR,
                    message="Error",
                    suggestion="Fix it",
                )
            ],
        )
        assert result.should_block is True
        assert len(result.errors) == 1

    def test_summary(self):
        result = PreflightResult(
            source_name="test_source",
            source_type="tabular",
            status=PreflightStatus.PASS,
            checks_run=["check1", "check2"],
            duration_ms=15.5,
        )
        summary = result.summary()
        assert "test_source" in summary
        assert "tabular" in summary
        assert "PASS" in summary
        assert "15.5ms" in summary

    def test_to_quality_issues(self):
        result = PreflightResult(
            source_name="test",
            source_type="tabular",
            issues=[
                PreflightIssue(
                    check_name="test",
                    category=IssueCategory.DATA_QUALITY,
                    severity=IssueSeverity.ERROR,
                    message="Error message",
                    suggestion="Fix it",
                )
            ],
        )
        quality_issues = result.to_quality_issues()
        assert len(quality_issues) == 1
        assert quality_issues[0].source_name == "test"
        assert quality_issues[0].message == "Error message"
        assert "suggestion" in quality_issues[0].details


class TestPathAccessibilityCheck:
    """Tests for PathAccessibilityCheck."""

    def test_existing_file_passes(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b,c\n1,2,3")

        source = Mock()
        source.path = str(test_file)

        check = PathAccessibilityCheck()
        issues = check.check(source)

        assert len(issues) == 0

    def test_missing_file_fails(self, tmp_path):
        source = Mock()
        source.path = str(tmp_path / "nonexistent.csv")

        check = PathAccessibilityCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.ERROR
        assert "not found" in issues[0].message.lower()

    def test_glob_pattern_matches(self, tmp_path):
        # Create test files
        (tmp_path / "file1.csv").write_text("a,b\n1,2")
        (tmp_path / "file2.csv").write_text("a,b\n3,4")

        source = Mock()
        source.path = str(tmp_path / "*.csv")

        check = PathAccessibilityCheck()
        issues = check.check(source)

        assert len(issues) == 0

    def test_glob_pattern_no_matches(self, tmp_path):
        source = Mock()
        source.path = str(tmp_path / "*.xyz")

        check = PathAccessibilityCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert "No files match pattern" in issues[0].message

    def test_no_path_attribute(self):
        source = Mock(spec=[])  # No path attribute

        check = PathAccessibilityCheck()
        issues = check.check(source)

        assert len(issues) == 0


class TestRequiredConfigCheck:
    """Tests for RequiredConfigCheck."""

    def test_tabular_with_all_fields(self):
        source = Mock()
        source.__class__.__name__ = "TabularSource"
        source.name = "test_source"
        source.path = "data.csv"
        source.config = {"name": "test_source", "path": "data.csv"}

        check = RequiredConfigCheck()
        issues = check.check(source)

        assert len(issues) == 0

    def test_missing_name_field(self):
        source = Mock()
        source.__class__.__name__ = "TabularSource"
        source.name = None
        source.path = "data.csv"
        source.config = {}

        check = RequiredConfigCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert "name" in issues[0].message

    def test_earthengine_required_fields(self):
        source = Mock()
        source.__class__.__name__ = "EarthEngineSource"
        source.name = "ee_source"
        source.collection = None  # Missing
        source.bands = []  # Empty
        source.config = {}

        check = RequiredConfigCheck()
        issues = check.check(source)

        # Should have issues for missing collection and bands
        assert len(issues) >= 2
        field_names = [i.details.get("field") for i in issues]
        assert "collection" in field_names
        assert "bands" in field_names


class TestTabularFormatCheck:
    """Tests for TabularFormatCheck."""

    def test_valid_csv_passes(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "latitude": [40.0, 41.0],
            "longitude": [-74.0, -73.0],
            "value": [1, 2],
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"
        source.geometry_col = None

        check = TabularFormatCheck()
        issues = check.check(source)

        assert len(issues) == 0

    def test_missing_lat_column_fails(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "lat": [40.0],  # Wrong column name
            "longitude": [-74.0],
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"  # Expected name
        source.lon_col = "longitude"
        source.geometry_col = None

        check = TabularFormatCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert "latitude" in issues[0].message
        assert "lat_col" in issues[0].suggestion

    def test_missing_lon_column_fails(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "latitude": [40.0],
            "lon": [-74.0],  # Wrong column name
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"  # Expected name
        source.geometry_col = None

        check = TabularFormatCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert "longitude" in issues[0].message

    def test_geometry_column_missing(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "id": [1, 2],
            "value": [100, 200],
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"
        source.geometry_col = "geom"  # Using geometry column

        check = TabularFormatCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert "geom" in issues[0].message

    def test_invalid_csv_fails(self, tmp_path):
        test_file = tmp_path / "bad.csv"
        test_file.write_text("not,valid,csv\n\"unclosed quote")

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"
        source.geometry_col = None

        check = TabularFormatCheck()
        issues = check.check(source)

        assert len(issues) == 1
        assert "Cannot parse" in issues[0].message

    def test_parquet_file(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.parquet"
        pd.DataFrame({
            "lat": [40.0],
            "lon": [-74.0],
        }).to_parquet(test_file)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "lat"
        source.lon_col = "lon"
        source.geometry_col = None

        check = TabularFormatCheck()
        issues = check.check(source)

        assert len(issues) == 0


class TestCoordinateRangeCheck:
    """Tests for CoordinateRangeCheck."""

    def test_valid_coordinates_pass(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "latitude": [40.0, 41.0, -33.0, 0.0],
            "longitude": [-74.0, 150.0, 0.0, -180.0],
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"

        check = CoordinateRangeCheck()
        issues = check.check(source)

        assert len(issues) == 0

    def test_invalid_latitude_warns(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "latitude": [40.0, 91.0, -91.0],  # Invalid values
            "longitude": [-74.0, 0.0, 0.0],
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"

        check = CoordinateRangeCheck()
        issues = check.check(source)

        assert len(issues) >= 1
        assert any("latitude" in i.message.lower() for i in issues)

    def test_invalid_longitude_warns(self, tmp_path):
        import pandas as pd

        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "latitude": [40.0, 41.0, 42.0],
            "longitude": [-74.0, 181.0, -181.0],  # Invalid values
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"

        check = CoordinateRangeCheck()
        issues = check.check(source)

        assert len(issues) >= 1
        assert any("longitude" in i.message.lower() for i in issues)

    def test_swapped_coordinates_error(self, tmp_path):
        import pandas as pd

        # Simulate swapped coordinates (lon values in lat column)
        test_file = tmp_path / "test.csv"
        pd.DataFrame({
            "latitude": [-122.0, -118.0, -74.0],  # These look like longitudes
            "longitude": [37.0, 34.0, 40.0],  # These look like latitudes
        }).to_csv(test_file, index=False)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.lat_col = "latitude"
        source.lon_col = "longitude"

        check = CoordinateRangeCheck()
        issues = check.check(source)

        # Should detect invalid latitudes
        assert len(issues) >= 1
        assert any("swapped" in i.suggestion.lower() for i in issues)


class TestRasterFormatCheck:
    """Tests for RasterFormatCheck."""

    def test_valid_tiff_magic_bytes(self, tmp_path):
        # Create a file with valid TIFF magic bytes
        test_file = tmp_path / "test.tif"
        test_file.write_bytes(b"II\x2a\x00" + b"\x00" * 100)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])
        source.band = None

        check = RasterFormatCheck()
        # Only check magic bytes (metadata check would fail without real TIFF)
        issues = check._check_magic_bytes(test_file)

        assert len(issues) == 0

    def test_invalid_magic_bytes(self, tmp_path):
        test_file = tmp_path / "not_a_tiff.tif"
        test_file.write_bytes(b"NOTF" + b"\x00" * 100)

        source = Mock()
        source.list_files = Mock(return_value=[test_file])

        check = RasterFormatCheck()
        issues = check._check_magic_bytes(test_file)

        assert len(issues) == 1
        assert "not a valid GeoTIFF" in issues[0].message

    def test_no_files_returns_empty(self):
        source = Mock()
        source.list_files = Mock(return_value=[])

        check = RasterFormatCheck()
        issues = check.check(source)

        assert len(issues) == 0

    def test_applies_to_raster_only(self):
        check = RasterFormatCheck()
        assert "raster" in check.applies_to
        assert "tabular" not in check.applies_to


class TestAuthenticationCheck:
    """Tests for AuthenticationCheck."""

    def test_earthengine_missing_package(self):
        source = Mock()
        source.__class__.__name__ = "EarthEngineSource"
        source.auth = {}

        check = AuthenticationCheck()

        with patch.dict("sys.modules", {"ee": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'ee'")):
                issues = check._check_packages("earthengine")

        assert len(issues) == 1
        assert "earthengine-api" in issues[0].message

    def test_stac_missing_package(self):
        check = AuthenticationCheck()

        with patch.dict("sys.modules", {"pystac_client": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                issues = check._check_packages("stac")

        assert len(issues) == 1
        assert "pystac-client" in issues[0].message

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_env_var_warns(self):
        source = Mock()
        source.__class__.__name__ = "EarthEngineSource"
        source.auth = None

        check = AuthenticationCheck()
        issues = check.check(source)

        # Should warn about missing GOOGLE_APPLICATION_CREDENTIALS
        env_issues = [i for i in issues if "GOOGLE_APPLICATION_CREDENTIALS" in i.message]
        assert len(env_issues) == 1
        assert env_issues[0].severity == IssueSeverity.WARNING

    def test_applies_to_remote_sources(self):
        check = AuthenticationCheck()
        assert "earthengine" in check.applies_to
        assert "stac" in check.applies_to
        assert "planetarycomputer" in check.applies_to
        assert "tabular" not in check.applies_to


class TestRunPreflightChecks:
    """Tests for run_preflight_checks function."""

    def test_returns_preflight_result(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b,c\n1,2,3")

        source = Mock()
        source.name = "test_source"
        source.path = str(test_file)
        source.__class__.__name__ = "TabularSource"
        source.list_files = Mock(return_value=[test_file])
        source.config = {"name": "test_source", "path": str(test_file)}
        source.lat_col = "a"
        source.lon_col = "b"
        source.geometry_col = None

        result = run_preflight_checks(source)

        assert isinstance(result, PreflightResult)
        assert result.source_name == "test_source"
        assert result.source_type == "tabular"
        assert len(result.checks_run) > 0
        assert result.duration_ms >= 0

    def test_check_data_quality_alias(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("lat,lon,value\n40,-74,100")

        source = Mock()
        source.name = "test"
        source.path = str(test_file)
        source.__class__.__name__ = "TabularSource"
        source.list_files = Mock(return_value=[test_file])
        source.config = {"name": "test", "path": str(test_file)}
        source.lat_col = "lat"
        source.lon_col = "lon"
        source.geometry_col = None

        result = check_data_quality(source)

        assert isinstance(result, PreflightResult)

    def test_custom_checks(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2")

        source = Mock()
        source.name = "test"
        source.path = str(test_file)
        source.__class__.__name__ = "TabularSource"

        # Only run path check
        result = run_preflight_checks(source, checks=[PathAccessibilityCheck()])

        assert len(result.checks_run) == 1
        assert "path_accessibility" in result.checks_run

    def test_check_crash_handled(self, tmp_path):
        source = Mock()
        source.name = "test"
        source.__class__.__name__ = "TabularSource"

        # Create a check that crashes
        class CrashingCheck(PreflightCheck):
            name = "crashing"
            category = IssueCategory.DATA_QUALITY
            applies_to = {"all"}

            def check(self, source):
                raise RuntimeError("Check crashed!")

        result = run_preflight_checks(source, checks=[CrashingCheck()])

        # Should not raise, but add a warning
        assert len(result.issues) == 1
        assert result.issues[0].severity == IssueSeverity.WARNING
        assert "crashed" in result.issues[0].suggestion.lower()


class TestPreflightCheckBase:
    """Tests for PreflightCheck base class."""

    def test_applies_to_all(self):
        class AllCheck(PreflightCheck):
            name = "all_check"
            category = IssueCategory.DATA_QUALITY
            applies_to = {"all"}

            def check(self, source):
                return []

        check = AllCheck()

        tabular_source = Mock()
        tabular_source.__class__.__name__ = "TabularSource"

        raster_source = Mock()
        raster_source.__class__.__name__ = "RasterSource"

        assert check.applies_to_source(tabular_source) is True
        assert check.applies_to_source(raster_source) is True

    def test_applies_to_specific(self):
        class RasterOnlyCheck(PreflightCheck):
            name = "raster_only"
            category = IssueCategory.DATA_QUALITY
            applies_to = {"raster"}

            def check(self, source):
                return []

        check = RasterOnlyCheck()

        tabular_source = Mock()
        tabular_source.__class__.__name__ = "TabularSource"

        raster_source = Mock()
        raster_source.__class__.__name__ = "RasterSource"

        assert check.applies_to_source(tabular_source) is False
        assert check.applies_to_source(raster_source) is True


class TestDefaultChecks:
    """Tests for default check configuration."""

    def test_default_checks_exist(self):
        assert len(DEFAULT_PREFLIGHT_CHECKS) > 0

    def test_all_checks_have_names(self):
        for check in DEFAULT_PREFLIGHT_CHECKS:
            assert hasattr(check, "name")
            assert check.name

    def test_all_checks_have_categories(self):
        for check in DEFAULT_PREFLIGHT_CHECKS:
            assert hasattr(check, "category")
            assert isinstance(check.category, IssueCategory)


class TestPreflightIntegration:
    """Integration tests with real source classes."""

    def test_tabular_source_preflight(self, tmp_path):
        import pandas as pd
        from geopipe.sources.tabular import TabularSource

        test_file = tmp_path / "data.csv"
        pd.DataFrame({
            "lat": [40.7, 34.0],
            "lon": [-74.0, -118.2],
            "value": [100, 200],
        }).to_csv(test_file, index=False)

        source = TabularSource(
            name="test",
            path=str(test_file),
            lat_col="lat",
            lon_col="lon",
        )

        result = source.preflight_check()

        assert result.status == PreflightStatus.PASS
        assert result.source_type == "tabular"
        assert not result.should_block

    def test_raster_source_missing_file(self, tmp_path):
        from geopipe.sources.raster import RasterSource

        source = RasterSource(
            name="test",
            path=str(tmp_path / "nonexistent/*.tif"),
        )

        result = source.preflight_check()

        assert result.status == PreflightStatus.FAIL
        assert any("No files" in i.message for i in result.errors)

    def test_tabular_source_missing_columns(self, tmp_path):
        import pandas as pd
        from geopipe.sources.tabular import TabularSource

        test_file = tmp_path / "data.csv"
        pd.DataFrame({
            "x": [40.7],
            "y": [-74.0],
        }).to_csv(test_file, index=False)

        source = TabularSource(
            name="test",
            path=str(test_file),
            lat_col="latitude",  # Doesn't exist
            lon_col="longitude",  # Doesn't exist
        )

        result = source.preflight_check()

        assert result.status == PreflightStatus.FAIL
        assert len(result.errors) >= 2
