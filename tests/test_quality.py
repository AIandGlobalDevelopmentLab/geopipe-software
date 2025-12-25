"""Tests for geopipe quality intelligence module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from geopipe.quality.checks import (
    IssueSeverity,
    IssueCategory,
    QualityIssue,
    QualityCheck,
    TemporalOverlapCheck,
    CRSAlignmentCheck,
    BoundsOverlapCheck,
    TemporalGapCheck,
    MissingValueCheck,
    DEFAULT_CHECKS,
)
from geopipe.quality.report import QualityReport


class TestQualityIssue:
    """Tests for QualityIssue model."""

    def test_create_issue(self):
        """Test creating a quality issue."""
        issue = QualityIssue(
            source_name="test_source",
            category=IssueCategory.TEMPORAL,
            severity=IssueSeverity.WARNING,
            message="Test warning message",
        )

        assert issue.source_name == "test_source"
        assert issue.category == IssueCategory.TEMPORAL
        assert issue.severity == IssueSeverity.WARNING
        assert issue.message == "Test warning message"
        assert issue.auto_fixable is False
        assert issue.details == {}

    def test_issue_with_details(self):
        """Test creating issue with details."""
        issue = QualityIssue(
            source_name="source",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.ERROR,
            message="Critical error",
            details={"missing_count": 100, "threshold": 50},
            auto_fixable=True,
            fix_description="Remove rows with missing values",
        )

        assert issue.details["missing_count"] == 100
        assert issue.auto_fixable is True
        assert issue.fix_description == "Remove rows with missing values"

    def test_issue_str_format(self):
        """Test issue string representation."""
        issue = QualityIssue(
            source_name="nightlights",
            category=IssueCategory.SPATIAL,
            severity=IssueSeverity.WARNING,
            message="High NoData percentage",
        )

        result = str(issue)
        assert "nightlights" in result
        assert "High NoData percentage" in result


class TestTemporalOverlapCheck:
    """Tests for TemporalOverlapCheck."""

    def test_overlapping_ranges(self):
        """Test that overlapping ranges pass."""
        check = TemporalOverlapCheck()

        # Create mock sources with overlapping temporal ranges
        source1 = Mock()
        source1.name = "source1"
        source1.temporal_range = ("2015-01-01", "2020-12-31")

        source2 = Mock()
        source2.name = "source2"
        source2.temporal_range = ("2018-01-01", "2022-12-31")

        schema = Mock()
        schema.sources = [source1, source2]

        issues = check.check_all([source1, source2], schema)
        assert len(issues) == 0

    def test_non_overlapping_ranges(self):
        """Test that non-overlapping ranges fail."""
        check = TemporalOverlapCheck()

        source1 = Mock()
        source1.name = "source1"
        source1.temporal_range = ("2010-01-01", "2015-12-31")

        source2 = Mock()
        source2.name = "source2"
        source2.temporal_range = ("2018-01-01", "2022-12-31")

        schema = Mock()

        issues = check.check_all([source1, source2], schema)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.ERROR
        assert "do not overlap" in issues[0].message

    def test_single_source_no_issues(self):
        """Test that single source produces no issues."""
        check = TemporalOverlapCheck()

        source = Mock()
        source.name = "source"
        source.temporal_range = ("2015-01-01", "2020-12-31")

        schema = Mock()

        issues = check.check_all([source], schema)
        assert len(issues) == 0


class TestCRSAlignmentCheck:
    """Tests for CRSAlignmentCheck."""

    def test_matching_crs(self):
        """Test that matching CRS produces no issues."""
        check = CRSAlignmentCheck()

        source1 = Mock()
        source1.name = "source1"
        source1.crs = "EPSG:4326"

        source2 = Mock()
        source2.name = "source2"
        source2.crs = "EPSG:4326"

        schema = Mock()
        schema.target_crs = "EPSG:4326"

        issues = check.check_all([source1, source2], schema)
        assert len(issues) == 0

    def test_mismatched_crs(self):
        """Test that mismatched CRS produces warning."""
        check = CRSAlignmentCheck()

        source1 = Mock()
        source1.name = "source1"
        source1.crs = "EPSG:4326"

        source2 = Mock()
        source2.name = "source2"
        source2.crs = "EPSG:32632"  # UTM zone

        schema = Mock()
        schema.target_crs = "EPSG:4326"

        issues = check.check_all([source1, source2], schema)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.WARNING
        assert issues[0].auto_fixable is True


class TestBoundsOverlapCheck:
    """Tests for BoundsOverlapCheck."""

    def test_overlapping_bounds(self):
        """Test that overlapping bounds pass."""
        check = BoundsOverlapCheck()

        source1 = Mock()
        source1.name = "source1"
        source1._bounds = (-10, -5, 10, 5)

        source2 = Mock()
        source2.name = "source2"
        source2._bounds = (-5, -2, 5, 2)

        schema = Mock()

        issues = check.check_all([source1, source2], schema)
        assert len(issues) == 0

    def test_non_overlapping_bounds(self):
        """Test that non-overlapping bounds fail."""
        check = BoundsOverlapCheck()

        source1 = Mock()
        source1.name = "source1"
        source1._bounds = (-180, -90, -90, 0)  # Western hemisphere

        source2 = Mock()
        source2.name = "source2"
        source2._bounds = (90, 0, 180, 90)  # Eastern hemisphere

        schema = Mock()

        issues = check.check_all([source1, source2], schema)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.ERROR


class TestQualityReport:
    """Tests for QualityReport."""

    def test_create_empty_report(self):
        """Test creating empty report."""
        schema = Mock()
        schema.name = "test_schema"
        schema.sources = []

        report = QualityReport(schema)

        assert report.schema == schema
        assert report.issues == []
        assert report.overall_score == 100.0

    def test_compute_scores_no_issues(self):
        """Test score computation with no issues."""
        schema = Mock()
        schema.name = "test"

        # Create sources with proper name attribute
        s1 = Mock()
        s1.name = "s1"
        s2 = Mock()
        s2.name = "s2"
        schema.sources = [s1, s2]

        report = QualityReport(schema)
        report.compute_scores()

        assert report.overall_score == 100.0
        assert report.source_scores["s1"] == 100
        assert report.source_scores["s2"] == 100

    def test_compute_scores_with_issues(self):
        """Test score computation with issues."""
        schema = Mock()
        schema.name = "test"

        source1 = Mock()
        source1.name = "source1"
        schema.sources = [source1]

        report = QualityReport(schema)
        report.issues = [
            QualityIssue(
                source_name="source1",
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.WARNING,
                message="Test warning",
            ),
            QualityIssue(
                source_name="source1",
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.INFO,
                message="Test info",
            ),
        ]
        report.compute_scores()

        # Warning = -10, Info = -2 => 100 - 12 = 88
        assert report.source_scores["source1"] == 88
        assert report.overall_score == 88

    def test_has_errors(self):
        """Test has_errors property."""
        schema = Mock()
        schema.name = "test"
        schema.sources = []

        report = QualityReport(schema)

        assert report.has_errors is False

        report.issues.append(
            QualityIssue(
                source_name="test",
                category=IssueCategory.TEMPORAL,
                severity=IssueSeverity.ERROR,
                message="Error",
            )
        )

        assert report.has_errors is True

    def test_has_warnings(self):
        """Test has_warnings property."""
        schema = Mock()
        schema.name = "test"
        schema.sources = []

        report = QualityReport(schema)

        assert report.has_warnings is False

        report.issues.append(
            QualityIssue(
                source_name="test",
                category=IssueCategory.TEMPORAL,
                severity=IssueSeverity.WARNING,
                message="Warning",
            )
        )

        assert report.has_warnings is True

    def test_fixable_issues(self):
        """Test fixable_issues property."""
        schema = Mock()
        schema.name = "test"
        schema.sources = []

        report = QualityReport(schema)
        report.issues = [
            QualityIssue(
                source_name="test",
                category=IssueCategory.ALIGNMENT,
                severity=IssueSeverity.WARNING,
                message="Fixable issue",
                auto_fixable=True,
            ),
            QualityIssue(
                source_name="test",
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.ERROR,
                message="Not fixable",
                auto_fixable=False,
            ),
        ]

        fixable = report.fixable_issues
        assert len(fixable) == 1
        assert fixable[0].message == "Fixable issue"

    def test_filter_by_severity(self):
        """Test filtering issues by severity."""
        schema = Mock()
        schema.name = "test"
        schema.sources = []

        report = QualityReport(schema)
        report.issues = [
            QualityIssue(
                source_name="s1",
                category=IssueCategory.TEMPORAL,
                severity=IssueSeverity.ERROR,
                message="Error",
            ),
            QualityIssue(
                source_name="s2",
                category=IssueCategory.TEMPORAL,
                severity=IssueSeverity.WARNING,
                message="Warning",
            ),
        ]

        errors = report.filter(severity=IssueSeverity.ERROR)
        assert len(errors) == 1
        assert errors[0].severity == IssueSeverity.ERROR

    def test_to_markdown(self):
        """Test markdown export."""
        schema = Mock()
        schema.name = "test_schema"

        source = Mock()
        source.name = "source1"
        schema.sources = [source]

        report = QualityReport(schema)
        report.issues = [
            QualityIssue(
                source_name="source1",
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.WARNING,
                message="Test warning",
            )
        ]
        report.compute_scores()

        md = report.to_markdown()

        assert "# Data Quality Audit Report" in md
        assert "test_schema" in md
        assert "source1" in md
        assert "Test warning" in md

    def test_to_latex(self):
        """Test LaTeX export."""
        schema = Mock()
        schema.name = "test"

        source = Mock()
        source.name = "source_one"
        schema.sources = [source]

        report = QualityReport(schema)
        report.compute_scores()

        latex = report.to_latex()

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "source\\_one" in latex  # Escaped underscore

    def test_to_dict(self):
        """Test dictionary export."""
        schema = Mock()
        schema.name = "test"
        schema.sources = []

        report = QualityReport(schema)
        report.issues = [
            QualityIssue(
                source_name="test",
                category=IssueCategory.TEMPORAL,
                severity=IssueSeverity.INFO,
                message="Info message",
            )
        ]
        report.compute_scores()

        result = report.to_dict()

        assert result["schema_name"] == "test"
        # INFO severity reduces score by 2, but since source_name="test" is not
        # a recognized source, it's treated as cross-source, reducing overall score
        assert result["overall_score"] == 98.0  # 100 - 2 (info penalty)
        assert len(result["issues"]) == 1
        assert result["issues"][0]["message"] == "Info message"


class TestQualityReportFromSchema:
    """Tests for QualityReport.from_schema() integration."""

    def test_from_schema_with_mock_sources(self):
        """Test creating report from schema with mocked sources."""
        from geopipe.sources.tabular import TabularSource

        schema = Mock()
        schema.name = "integration_test"
        schema.temporal_range = ("2015-01-01", "2020-12-31")
        schema.target_crs = "EPSG:4326"

        # Create mock sources that pass basic checks
        source1 = Mock(spec=TabularSource)
        source1.name = "tabular_source"
        source1.temporal_range = ("2015-01-01", "2020-12-31")
        source1.crs = "EPSG:4326"
        source1._bounds = (-10, -5, 10, 5)
        source1.list_files = Mock(return_value=[])

        source2 = Mock()
        source2.name = "other_source"
        source2.temporal_range = ("2016-01-01", "2019-12-31")
        source2.crs = "EPSG:4326"
        source2._bounds = (-5, -2, 5, 2)
        source2.list_files = Mock(return_value=[])

        schema.sources = [source1, source2]

        # Use only cross-source checks for faster testing
        from geopipe.quality.checks import (
            TemporalOverlapCheck,
            CRSAlignmentCheck,
            BoundsOverlapCheck,
        )

        checks = [
            TemporalOverlapCheck(),
            CRSAlignmentCheck(),
            BoundsOverlapCheck(),
        ]

        report = QualityReport.from_schema(schema, checks=checks)

        assert report.schema == schema
        assert report.overall_score >= 0
        assert "temporal_overlap" in report._checks_run


class TestDefaultChecks:
    """Tests for DEFAULT_CHECKS list."""

    def test_default_checks_exist(self):
        """Test that default checks are defined."""
        assert len(DEFAULT_CHECKS) > 0

    def test_default_checks_are_quality_checks(self):
        """Test that all defaults are QualityCheck instances."""
        for check in DEFAULT_CHECKS:
            assert isinstance(check, QualityCheck)

    def test_default_checks_have_names(self):
        """Test that all checks have names."""
        for check in DEFAULT_CHECKS:
            assert hasattr(check, "name")
            assert check.name is not None
