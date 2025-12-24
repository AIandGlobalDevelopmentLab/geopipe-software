"""Tests for specification management."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from geopipe.specs.variants import Spec, SpecRegistry


class TestSpec:
    """Tests for Spec class."""

    def test_basic_spec(self):
        """Test basic spec creation."""
        spec = Spec("MAIN", buffer_km=5, include_ntl=True)
        assert spec.name == "MAIN"
        assert spec.buffer_km == 5
        assert spec.include_ntl is True

    def test_spec_with_description(self):
        """Test spec with description."""
        spec = Spec(
            "ROBUST_BUFFER",
            description="Larger buffer for sensitivity check",
            buffer_km=10,
        )
        assert spec.description == "Larger buffer for sensitivity check"

    def test_spec_attribute_access(self):
        """Test attribute-style parameter access."""
        spec = Spec("test", alpha=0.05, n_iter=1000)
        assert spec.alpha == 0.05
        assert spec.n_iter == 1000

    def test_spec_missing_param(self):
        """Test accessing non-existent parameter."""
        spec = Spec("test", a=1)
        with pytest.raises(AttributeError):
            _ = spec.nonexistent

    def test_spec_to_dict(self):
        """Test converting spec to dict."""
        spec = Spec("MAIN", description="Primary spec", buffer_km=5)
        d = spec.to_dict()
        assert d["name"] == "MAIN"
        assert d["description"] == "Primary spec"
        assert d["buffer_km"] == 5


class TestSpecRegistry:
    """Tests for SpecRegistry class."""

    @pytest.fixture
    def sample_registry(self):
        """Create a sample registry."""
        return SpecRegistry([
            Spec("MAIN", buffer_km=5, include_ntl=True),
            Spec("ROBUST_BUFFER", buffer_km=10, include_ntl=True),
            Spec("ROBUST_NO_NTL", buffer_km=5, include_ntl=False),
        ])

    def test_init(self, sample_registry):
        """Test registry initialization."""
        assert len(sample_registry) == 3

    def test_iteration(self, sample_registry):
        """Test iterating over specs."""
        names = [spec.name for spec in sample_registry]
        assert names == ["MAIN", "ROBUST_BUFFER", "ROBUST_NO_NTL"]

    def test_get_by_name(self, sample_registry):
        """Test getting spec by name."""
        spec = sample_registry.get("ROBUST_BUFFER")
        assert spec is not None
        assert spec.buffer_km == 10

    def test_get_missing(self, sample_registry):
        """Test getting non-existent spec."""
        assert sample_registry.get("NONEXISTENT") is None

    def test_getitem_by_name(self, sample_registry):
        """Test bracket access by name."""
        spec = sample_registry["MAIN"]
        assert spec.name == "MAIN"

    def test_getitem_by_index(self, sample_registry):
        """Test bracket access by index."""
        spec = sample_registry[0]
        assert spec.name == "MAIN"

    def test_getitem_missing(self, sample_registry):
        """Test bracket access with missing key."""
        with pytest.raises(KeyError):
            _ = sample_registry["MISSING"]

    def test_add_spec(self):
        """Test adding specs to registry."""
        registry = SpecRegistry()
        registry.add(Spec("A", x=1))
        registry.add(Spec("B", x=2))
        assert len(registry) == 2

    def test_summary(self, sample_registry):
        """Test summary generation."""
        summary = sample_registry.summary()
        assert "MAIN" in summary
        assert "ROBUST_BUFFER" in summary
        assert "buffer_km" in summary

    def test_load_results(self, tmp_path, sample_registry):
        """Test loading results from files."""
        # Create sample result files
        sample_registry.results_dir = tmp_path

        for spec_name in ["MAIN", "ROBUST_BUFFER", "ROBUST_NO_NTL"]:
            spec_dir = tmp_path / spec_name
            spec_dir.mkdir()
            df = pd.DataFrame({"estimate": [0.5], "std_error": [0.1]})
            df.to_csv(spec_dir / "estimates.csv", index=False)

        results = sample_registry.load_results("{spec}/estimates.csv")
        assert len(results) == 3
        assert "MAIN" in results
        assert results["MAIN"]["estimate"].iloc[0] == 0.5

    def test_compare_results(self, tmp_path, sample_registry):
        """Test comparing results across specs."""
        sample_registry.results_dir = tmp_path

        # Create results with different estimates
        for i, spec_name in enumerate(["MAIN", "ROBUST_BUFFER", "ROBUST_NO_NTL"]):
            spec_dir = tmp_path / spec_name
            spec_dir.mkdir()
            df = pd.DataFrame({
                "estimate": [0.5 + i * 0.1],
                "std_error": [0.1],
            })
            df.to_csv(spec_dir / "estimates.csv", index=False)

        comparison = sample_registry.compare_results("{spec}/estimates.csv")
        assert len(comparison) == 3
        assert "spec" in comparison.columns
        assert "estimate" in comparison.columns

    def test_to_latex(self, tmp_path, sample_registry):
        """Test LaTeX table generation."""
        sample_registry.results_dir = tmp_path

        for spec_name in ["MAIN", "ROBUST_BUFFER"]:
            spec_dir = tmp_path / spec_name
            spec_dir.mkdir()
            df = pd.DataFrame({"estimate": [0.5], "std_error": [0.1]})
            df.to_csv(spec_dir / "estimates.csv", index=False)

        latex = sample_registry.to_latex("{spec}/estimates.csv")
        assert r"\begin{table}" in latex
        assert "MAIN" in latex
        assert r"\end{table}" in latex
