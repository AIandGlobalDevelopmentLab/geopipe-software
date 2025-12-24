# CLAUDE.md

This file provides guidance to Claude Code when working with the geopipe codebase.

## Project Overview

geopipe is a Python package for managing large-scale geospatial data pipelines, designed for causal inference workflows that integrate satellite imagery with heterogeneous tabular data sources.

## Architecture

```
geopipe/
├── sources/     # Data source connectors (raster, tabular, vector)
├── fusion/      # Declarative fusion schemas
├── pipeline/    # Task orchestration with checkpointing
├── cluster/     # HPC integration (SLURM, PBS)
├── specs/       # Robustness specification management
└── cli.py       # Command-line interface
```

## Key Components

### Data Sources (`geopipe/sources/`)
- `base.py` - Abstract `DataSource` class with `load()`, `validate()`, `get_schema()`
- `raster.py` - `RasterSource` for GeoTIFF/COG with zonal statistics
- `tabular.py` - `TabularSource` for CSV/Parquet with spatial joins

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
- LaTeX table generation for papers
- Cross-spec result comparison

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
# Initialize template schema
geopipe init --output schema.yaml

# Validate schema
geopipe validate schema.yaml

# Execute fusion
geopipe fuse schema.yaml

# Check pipeline status
geopipe status

# Clean checkpoints
geopipe clean
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
- `planetary-computer`, `earthengine-api` - Remote data sources (not yet implemented)

## R Integration

R users can access geopipe via the `reticulate` package. See README.md "Using geopipe from R" section for setup and usage examples with conda environments.

## Testing

Tests are in `tests/` using pytest. Each module has corresponding test file:
- `test_sources.py` - Data source tests
- `test_fusion.py` - Fusion schema tests
- `test_pipeline.py` - Pipeline and task tests
- `test_specs.py` - Specification management tests
