"""Tests for geopipe robustness DSL module."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

import pandas as pd

from geopipe.specs.dsl import RobustnessDSL, expand_robustness_specs
from geopipe.specs.variants import Spec
from geopipe.specs.curve import SpecificationCurve


class TestRobustnessDSL:
    """Tests for RobustnessDSL parser and expander."""

    def test_empty_dimensions(self):
        """Test DSL with no dimensions."""
        dsl = RobustnessDSL({})
        specs = dsl.expand()
        assert len(specs) == 0
        assert dsl.count() == 0

    def test_single_dimension(self):
        """Test DSL with a single dimension."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10, 25],
            }
        })
        specs = dsl.expand()
        assert len(specs) == 3
        assert dsl.count() == 3

        # Check parameter values
        values = [s.params["buffer_km"] for s in specs]
        assert set(values) == {5, 10, 25}

    def test_multiple_dimensions_cartesian_product(self):
        """Test that multiple dimensions produce Cartesian product."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
                "include_ntl": [True, False],
            }
        })
        specs = dsl.expand()

        # 2 x 2 = 4 combinations
        assert len(specs) == 4
        assert dsl.count() == 4

        # Check all combinations exist
        param_combos = [(s.params["buffer_km"], s.params["include_ntl"]) for s in specs]
        expected = [(5, True), (5, False), (10, True), (10, False)]
        assert set(param_combos) == set(expected)

    def test_three_dimensions(self):
        """Test with three dimensions."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
                "include_ntl": [True, False],
                "temporal_lag": [0, 1, 2],
            }
        })
        specs = dsl.expand()

        # 2 x 2 x 3 = 12 combinations
        assert len(specs) == 12
        assert dsl.count() == 12

    def test_exclusions_single(self):
        """Test that single exclusion removes matching specs."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10, 25],
                "include_ntl": [True, False],
            },
            "exclude": [
                {"buffer_km": 25, "include_ntl": False},
            ],
        })
        specs = dsl.expand()

        # 3 x 2 = 6, minus 1 exclusion = 5
        assert len(specs) == 5
        assert dsl.count() == 5

        # Verify excluded combination is not present
        for spec in specs:
            if spec.params["buffer_km"] == 25:
                assert spec.params["include_ntl"] is True

    def test_exclusions_multiple(self):
        """Test multiple exclusions."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10, 25],
                "include_ntl": [True, False],
            },
            "exclude": [
                {"buffer_km": 25, "include_ntl": False},
                {"buffer_km": 5, "include_ntl": True},
            ],
        })
        specs = dsl.expand()

        # 6 - 2 = 4
        assert len(specs) == 4

    def test_exclusions_partial_match(self):
        """Test exclusion that matches only one dimension."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
                "include_ntl": [True, False],
            },
            "exclude": [
                {"buffer_km": 5},  # Matches all with buffer_km=5
            ],
        })
        specs = dsl.expand()

        # 4 - 2 (both buffer_km=5 variants) = 2
        assert len(specs) == 2

        for spec in specs:
            assert spec.params["buffer_km"] == 10

    def test_named_specs(self):
        """Test that named specs get their names assigned."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10, 25],
                "include_ntl": [True, False],
            },
            "named": {
                "MAIN": {"buffer_km": 10, "include_ntl": True},
                "MINIMAL": {"buffer_km": 5, "include_ntl": False},
            },
        })
        specs = dsl.expand()

        # Find named specs
        names = [s.name for s in specs]
        assert "MAIN" in names
        assert "MINIMAL" in names

        # Verify MAIN has correct params
        main_spec = next(s for s in specs if s.name == "MAIN")
        assert main_spec.params["buffer_km"] == 10
        assert main_spec.params["include_ntl"] is True

    def test_get_spec_named(self):
        """Test getting a named spec directly."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
            },
            "named": {
                "MAIN": {"buffer_km": 10},
            },
        })

        spec = dsl.get_spec("MAIN")
        assert spec is not None
        assert spec.name == "MAIN"
        assert spec.params["buffer_km"] == 10

    def test_get_spec_not_found(self):
        """Test getting a non-existent spec returns None."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
            },
        })

        spec = dsl.get_spec("NONEXISTENT")
        assert spec is None

    def test_auto_naming_with_bool(self):
        """Test automatic name generation includes booleans correctly."""
        dsl = RobustnessDSL({
            "dimensions": {
                "include_ntl": [True],
            },
        })
        specs = dsl.expand()
        assert len(specs) == 1

        # Boolean True should include the dimension name
        assert "include_ntl" in specs[0].name

    def test_auto_naming_with_values(self):
        """Test automatic name generation includes values."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [10],
            },
        })
        specs = dsl.expand()
        assert len(specs) == 1

        # Should include dimension and value
        assert "buffer_km" in specs[0].name
        assert "10" in specs[0].name

    def test_naming_pattern(self):
        """Test custom naming pattern."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
            },
            "naming_pattern": "spec_buf{buffer_km}",
        })
        specs = dsl.expand()

        names = [s.name for s in specs]
        assert "spec_buf5" in names
        assert "spec_buf10" in names


class TestRobustnessDSLTemplateSubstitution:
    """Tests for template substitution in schema configs."""

    def test_substitute_dollar_brace(self):
        """Test ${param} substitution."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [10],
            },
        })
        spec = dsl.expand()[0]

        schema_dict = {
            "output": "results/${buffer_km}km/data.parquet",
        }

        result = dsl.apply_to_schema(schema_dict, spec)
        assert result["output"] == "results/10km/data.parquet"

    def test_substitute_brace(self):
        """Test {param} substitution."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [25],
            },
        })
        spec = dsl.expand()[0]

        schema_dict = {
            "output": "results/{buffer_km}km/data.parquet",
        }

        result = dsl.apply_to_schema(schema_dict, spec)
        assert result["output"] == "results/25km/data.parquet"

    def test_substitute_spec_name(self):
        """Test {spec_name} substitution."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [10],
            },
            "named": {
                "MAIN": {"buffer_km": 10},
            },
        })
        spec = dsl.expand()[0]

        schema_dict = {
            "output": "results/{spec_name}/data.parquet",
        }

        result = dsl.apply_to_schema(schema_dict, spec)
        assert "MAIN" in result["output"]

    def test_substitute_nested(self):
        """Test substitution in nested structures."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [10],
            },
        })
        spec = dsl.expand()[0]

        schema_dict = {
            "sources": [
                {
                    "name": "conflict",
                    "spatial_join": "buffer_${buffer_km}km",
                },
            ],
        }

        result = dsl.apply_to_schema(schema_dict, spec)
        assert result["sources"][0]["spatial_join"] == "buffer_10km"

    def test_substitute_preserves_non_template(self):
        """Test that non-template strings are preserved."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [10],
            },
        })
        spec = dsl.expand()[0]

        schema_dict = {
            "name": "my_fusion",
            "resolution": "5km",
        }

        result = dsl.apply_to_schema(schema_dict, spec)
        assert result["name"] == "my_fusion"
        assert result["resolution"] == "5km"

    def test_substitute_multiple_in_string(self):
        """Test multiple substitutions in one string."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [10],
                "temporal_lag": [2],
            },
        })
        spec = dsl.expand()[0]

        schema_dict = {
            "output": "buf${buffer_km}_lag${temporal_lag}.parquet",
        }

        result = dsl.apply_to_schema(schema_dict, spec)
        assert result["output"] == "buf10_lag2.parquet"


class TestExpandRobustnessSpecs:
    """Tests for expand_robustness_specs helper function."""

    def test_no_robustness_block(self):
        """Test schema with no robustness block returns MAIN spec."""
        schema_dict = {
            "name": "test",
            "output": "output.parquet",
        }

        specs, schemas = expand_robustness_specs(schema_dict)

        assert len(specs) == 1
        assert specs[0].name == "MAIN"
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test"

    def test_with_robustness_block(self):
        """Test schema with robustness block expands correctly."""
        schema_dict = {
            "name": "test",
            "output": "results/{spec_name}/data.parquet",
            "robustness": {
                "dimensions": {
                    "buffer_km": [5, 10],
                },
            },
        }

        specs, schemas = expand_robustness_specs(schema_dict)

        assert len(specs) == 2
        assert len(schemas) == 2

        # Robustness block should be removed from output schemas
        for sd in schemas:
            assert "robustness" not in sd

    def test_template_substitution_applied(self):
        """Test that template substitution is applied to expanded schemas."""
        schema_dict = {
            "name": "test",
            "output": "results/${buffer_km}km/data.parquet",
            "robustness": {
                "dimensions": {
                    "buffer_km": [5, 10],
                },
            },
        }

        specs, schemas = expand_robustness_specs(schema_dict)

        outputs = [sd["output"] for sd in schemas]
        assert "results/5km/data.parquet" in outputs
        assert "results/10km/data.parquet" in outputs


class TestSpecificationCurve:
    """Tests for SpecificationCurve analysis."""

    def test_create_curve(self):
        """Test creating a specification curve."""
        specs = [Spec("spec1"), Spec("spec2")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="results/{spec_name}/estimates.csv",
        )

        assert len(curve.specs) == 2
        assert curve.estimate_col == "estimate"
        assert curve.se_col == "std_error"

    def test_load_results_from_csv(self, tmp_path):
        """Test loading results from CSV files."""
        # Create test result files
        spec1_dir = tmp_path / "spec1"
        spec1_dir.mkdir()
        df1 = pd.DataFrame({"estimate": [0.5], "std_error": [0.1]})
        df1.to_csv(spec1_dir / "estimates.csv", index=False)

        spec2_dir = tmp_path / "spec2"
        spec2_dir.mkdir()
        df2 = pd.DataFrame({"estimate": [0.8], "std_error": [0.15]})
        df2.to_csv(spec2_dir / "estimates.csv", index=False)

        specs = [Spec("spec1"), Spec("spec2")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="{spec_name}/estimates.csv",
            results_dir=str(tmp_path),
        )

        results = curve.load_results()

        assert len(results) == 2
        assert "estimate" in results.columns
        assert "std_error" in results.columns

    def test_compute_summary(self, tmp_path):
        """Test computing summary statistics."""
        # Create test results
        spec_dir = tmp_path / "spec1"
        spec_dir.mkdir()
        df = pd.DataFrame({
            "estimate": [0.5, 0.6, 0.7],
            "std_error": [0.1, 0.1, 0.1],
        })
        df.to_csv(spec_dir / "estimates.csv", index=False)

        specs = [Spec("spec1")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="{spec_name}/estimates.csv",
            results_dir=str(tmp_path),
        )

        curve.load_results()
        summary = curve.compute_summary()

        assert "n_specs" in summary
        assert "mean_estimate" in summary
        assert "median_estimate" in summary
        assert "std_estimate" in summary
        assert "pct_positive" in summary

    def test_compute_summary_with_significance(self, tmp_path):
        """Test summary includes significance stats when SE available."""
        spec_dir = tmp_path / "spec1"
        spec_dir.mkdir()
        df = pd.DataFrame({
            "estimate": [0.5],
            "std_error": [0.1],  # z = 5.0, significant
        })
        df.to_csv(spec_dir / "estimates.csv", index=False)

        specs = [Spec("spec1")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="{spec_name}/estimates.csv",
            results_dir=str(tmp_path),
        )

        curve.load_results()
        summary = curve.compute_summary()

        assert "pct_significant_05" in summary
        assert "pct_significant_10" in summary

    def test_rank_by_influence_empty(self):
        """Test rank_by_influence with no results."""
        specs = [Spec("spec1")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="*.csv",
        )

        # Empty results
        curve._results = pd.DataFrame()
        influence = curve.rank_by_influence()
        assert influence.empty

    def test_rank_by_influence(self, tmp_path):
        """Test ranking dimensions by influence."""
        # Create results with varying buffer_km
        for buffer in [5, 10]:
            spec_name = f"buffer_km_{buffer}"
            spec_dir = tmp_path / spec_name
            spec_dir.mkdir()
            # Larger buffer = larger estimate to show influence
            df = pd.DataFrame({
                "estimate": [buffer * 0.1],
                "std_error": [0.1],
            })
            df.to_csv(spec_dir / "estimates.csv", index=False)

        specs = [
            Spec("buffer_km_5", buffer_km=5),
            Spec("buffer_km_10", buffer_km=10),
        ]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="{spec_name}/estimates.csv",
            results_dir=str(tmp_path),
        )

        curve.load_results()
        influence = curve.rank_by_influence()

        assert not influence.empty
        assert "dimension" in influence.columns
        assert "variance_ratio" in influence.columns
        assert "estimate_range" in influence.columns

    def test_to_markdown(self, tmp_path):
        """Test markdown export."""
        spec_dir = tmp_path / "spec1"
        spec_dir.mkdir()
        df = pd.DataFrame({"estimate": [0.5], "std_error": [0.1]})
        df.to_csv(spec_dir / "estimates.csv", index=False)

        specs = [Spec("spec1")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="{spec_name}/estimates.csv",
            results_dir=str(tmp_path),
        )
        curve.load_results()

        md = curve.to_markdown()

        assert "# Specification Curve Summary" in md
        assert "Mean:" in md
        assert "Median:" in md

    def test_to_latex(self, tmp_path):
        """Test LaTeX export."""
        spec_dir = tmp_path / "spec1"
        spec_dir.mkdir()
        df = pd.DataFrame({"estimate": [0.5], "std_error": [0.1]})
        df.to_csv(spec_dir / "estimates.csv", index=False)

        specs = [Spec("spec1")]
        curve = SpecificationCurve(
            specs=specs,
            results_pattern="{spec_name}/estimates.csv",
            results_dir=str(tmp_path),
        )
        curve.load_results()

        latex = curve.to_latex()

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "specifications" in latex.lower()

    def test_repr(self):
        """Test string representation."""
        specs = [Spec("spec1"), Spec("spec2")]
        curve = SpecificationCurve(specs=specs, results_pattern="*.csv")

        repr_str = repr(curve)
        assert "SpecificationCurve" in repr_str
        assert "specs=2" in repr_str


class TestFusionSchemaExpandSpecs:
    """Tests for FusionSchema.expand_specs() integration."""

    def test_expand_specs_no_robustness(self):
        """Test expand_specs with no robustness block."""
        from geopipe.fusion.schema import FusionSchema

        schema = FusionSchema(
            name="test",
            sources=[],
            output="output.parquet",
        )

        result = schema.expand_specs()

        assert len(result) == 1
        spec, configured_schema = result[0]
        assert spec.name == "MAIN"
        assert configured_schema.name == "test"

    def test_expand_specs_with_robustness(self):
        """Test expand_specs with robustness block."""
        from geopipe.fusion.schema import FusionSchema

        schema = FusionSchema(
            name="test",
            sources=[],
            output="results/${buffer_km}km/output.parquet",
            robustness={
                "dimensions": {
                    "buffer_km": [5, 10],
                },
            },
        )

        result = schema.expand_specs()

        assert len(result) == 2

        # Check outputs are substituted
        outputs = [sch.output for _, sch in result]
        assert "results/5km/output.parquet" in outputs
        assert "results/10km/output.parquet" in outputs


class TestRobustnessDSLSummary:
    """Tests for DSL summary and string representation."""

    def test_summary(self):
        """Test summary generation."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10, 25],
                "include_ntl": [True, False],
            },
            "exclude": [
                {"buffer_km": 25, "include_ntl": False},
            ],
            "named": {
                "MAIN": {"buffer_km": 10, "include_ntl": True},
            },
        })

        summary = dsl.summary()

        assert "RobustnessDSL" in summary
        assert "buffer_km" in summary
        assert "include_ntl" in summary
        assert "Exclusions: 1" in summary
        assert "MAIN" in summary
        assert "5" in summary  # Total specs

    def test_repr(self):
        """Test string representation."""
        dsl = RobustnessDSL({
            "dimensions": {
                "buffer_km": [5, 10],
            },
        })

        repr_str = repr(dsl)
        assert "RobustnessDSL" in repr_str
        assert "buffer_km" in repr_str
        assert "count=2" in repr_str


class TestRobustnessDSLFromYamlBlock:
    """Tests for from_yaml_block class method."""

    def test_from_yaml_block(self):
        """Test creating DSL from YAML-parsed block."""
        yaml_block = {
            "dimensions": {
                "buffer_km": [5, 10],
            },
            "exclude": [],
            "named": {},
        }

        dsl = RobustnessDSL.from_yaml_block(yaml_block)

        assert dsl.count() == 2
        assert "buffer_km" in dsl.dimensions
