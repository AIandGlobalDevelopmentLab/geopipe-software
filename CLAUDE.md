# CLAUDE.md

This file provides guidance to Claude Code when working with the geopipe codebase.

## Project Overview

geopipe is a Python package for managing large-scale geospatial data pipelines, designed for causal inference workflows that integrate satellite imagery with heterogeneous tabular data sources.

## Architecture

```
geopipe/
├── sources/     # Data source connectors (raster, tabular, remote)
├── fusion/      # Declarative fusion schemas
├── pipeline/    # Task orchestration with checkpointing
├── cluster/     # HPC integration (SLURM)
├── specs/       # Robustness specs, DSL, and curve analysis
├── quality/     # Data quality checks (preflight + post-load)
├── discovery/   # Dataset discovery across catalogs
└── cli.py       # Command-line interface
```

## Key Components

### Data Sources (`geopipe/sources/`)
- `base.py` - Abstract `DataSource` class with `load()`, `validate()`, `get_schema()`
- `raster.py` - `RasterSource` for GeoTIFF/COG with zonal statistics
- `tabular.py` - `TabularSource` for CSV/Parquet with spatial joins
- `remote_base.py` - `RemoteSourceMixin` base class for cloud sources
- `earthengine.py` - `EarthEngineSource` for Google Earth Engine collections
- `planetary.py` - `PlanetaryComputerSource` for Microsoft Planetary Computer
- `stac.py` - `STACSource` for generic STAC catalog access

### Fusion (`geopipe/fusion/`)
- `schema.py` - `FusionSchema` for declarative multi-source data fusion
- Supports YAML configuration files
- Handles resolution alignment, temporal filtering, spatial joins

### Pipeline (`geopipe/pipeline/`)
- `tasks.py` - `@task` decorator with caching and checkpointing
- `dag.py` - `Pipeline` class for DAG-based execution
- Supports resume from failure

### Cluster (`geopipe/cluster/`)
- `slurm.py` - `SLURMExecutor` for distributed pipeline execution
- Auto-generates job scripts with dependencies
- Job monitoring and cancellation

### Specs (`geopipe/specs/`)
- `variants.py` - `Spec` and `SpecRegistry` for robustness specifications
- `dsl.py` - `RobustnessDSL` for declarative YAML specs with template substitution
- `curve.py` - `SpecificationCurve` for analysis, plotting, and influence ranking
- LaTeX table generation for papers
- Cross-spec result comparison

### Quality Checks (`geopipe/quality/`)
- `checks.py` - Quality check classes: `TemporalOverlapCheck`, `CRSAlignmentCheck`, `BoundsOverlapCheck`, `MissingValueCheck`, etc.
- `preflight.py` - Pre-load validation: `PathAccessibilityCheck`, `RequiredConfigCheck`, `TabularFormatCheck`, `RasterFormatCheck`, `AuthenticationCheck`
- `report.py` - `QualityReport` with export to Markdown, LaTeX, JSON

### Data Discovery (`geopipe/discovery/`)
- `catalog.py` - `CatalogRegistry`, `DatasetInfo`, `discover()` function
- `results.py` - `DiscoveryResult`, `CategoryType` enum
- Categories: nightlights, optical, sar, elevation, climate, land_cover, vegetation, population, infrastructure

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=geopipe

# Type checking
mypy geopipe

# Linting
ruff check geopipe
```

## CLI Commands

```bash
# Core commands
geopipe init --output schema.yaml      # Initialize template schema
geopipe validate schema.yaml           # Validate schema
geopipe fuse schema.yaml               # Execute fusion
geopipe status                         # Check pipeline status
geopipe clean                          # Clean checkpoints

# Quality commands
geopipe preflight schema.yaml          # Run pre-load validation checks
geopipe audit schema.yaml              # Data quality audit with --fix, --output, --strict

# Source management
geopipe sources list schema.yaml       # List sources in schema

# Specification commands
geopipe specs list schema.yaml         # Preview specifications
geopipe specs expand schema.yaml       # Write individual schema files
geopipe specs run schema.yaml          # Execute all specifications
geopipe specs curve "results/*.csv"    # Generate specification curve

# Discovery commands
geopipe discover search                # Search datasets (--category, --provider)
geopipe discover list-datasets         # List all known datasets
geopipe discover info <dataset_id>     # Get dataset details
geopipe discover categories            # List available categories
```

## Design Principles

1. **Declarative over imperative** - YAML schemas define what, not how
2. **Checkpoint everything** - Support resume from any failure point
3. **HPC-native** - First-class SLURM/PBS support
4. **Robustness-focused** - Built-in specification management for sensitivity analysis

## Dependencies

Core: geopandas, xarray, rioxarray, dask, pyarrow, pyyaml, click, rich, pydantic

Optional:
- `prefect` - For advanced orchestration
- `submitit` - Alternative SLURM submission
- `planetary-computer`, `earthengine-api`, `pystac-client`, `stackstac`, `tenacity`, `requests` - Remote data sources

## R Integration

R users can access geopipe via the `reticulate` package. See README.md "Using geopipe from R" section for setup and usage examples with conda environments.

## Testing

Tests are in `tests/` using pytest. Each module has corresponding test file:
- `test_sources.py` - Data source tests
- `test_fusion.py` - Fusion schema tests
- `test_pipeline.py` - Pipeline and task tests
- `test_specs.py` - Specification management tests
- `test_robustness_dsl.py` - Robustness DSL tests
- `test_preflight.py` - Preflight validation tests
- `test_quality.py` - Quality audit tests
- `test_discovery.py` - Data discovery tests
- `test_remote_sources.py` - Remote source tests
