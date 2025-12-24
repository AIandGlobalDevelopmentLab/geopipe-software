# geopipe

**Geospatial data pipeline framework for causal inference workflows**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

geopipe simplifies large-scale geospatial data pipelines that integrate satellite imagery with heterogeneous tabular data sources. It's designed for researchers running causal inference workflows with complex data fusion requirements.

## Features

- **Declarative Data Fusion**: YAML/Python schemas for joining 27+ heterogeneous sources
- **Pipeline Orchestration**: DAG-based workflows with checkpointing and resume
- **Cluster Computing**: Native SLURM/PBS integration with job monitoring
- **Robustness Specifications**: Manage parallel analysis variants (MAIN, ROBUST_*, etc.)

## Installation

```bash
pip install geopipe

# With optional dependencies
pip install geopipe[prefect,cluster]  # Pipeline orchestration + HPC
pip install geopipe[remote]           # Earth Engine, Planetary Computer
pip install geopipe[all]              # Everything
```

### System Dependencies

geopipe requires GDAL and PROJ for geospatial operations. Install these first:

**macOS (Homebrew):**
```bash
brew install gdal proj
```

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev libproj-dev
```

**Conda (recommended):**
```bash
conda install -c conda-forge gdal proj
pip install geopipe
```

## Quick Start

### 1. Define a Data Fusion Schema

```python
from geopipe import FusionSchema, RasterSource, TabularSource

schema = FusionSchema(
    name="aid_effects",
    resolution="5km",
    temporal_range=("2000-01-01", "2020-12-31"),
    sources=[
        RasterSource("nightlights", path="data/viirs/*.tif", aggregation="mean"),
        RasterSource("landcover", path="data/modis/*.tif", aggregation="mode"),
        TabularSource("conflict", path="data/acled.csv",
                      spatial_join="buffer_10km", temporal_align="yearly_sum"),
        TabularSource("aid_projects", path="data/aiddata.csv",
                      spatial_join="nearest"),
    ],
    output="data/interim/consolidated.parquet"
)

# Execute fusion
result = schema.execute()
```

### 2. Or use YAML configuration

```yaml
# sources.yaml
name: aid_effects
resolution: 5km
temporal_range: [2000-01-01, 2020-12-31]

sources:
  - type: raster
    name: nightlights
    path: data/viirs/*.tif
    aggregation: mean

  - type: tabular
    name: conflict
    path: data/acled.csv
    spatial_join: buffer_10km
    temporal_align: yearly_sum

output: data/interim/consolidated.parquet
```

```bash
geopipe fuse sources.yaml
```

#### Alignment Options Reference

**Spatial Join Methods** (`spatial_join`):
| Method | Description |
|--------|-------------|
| `nearest` | Nearest neighbor join (default) |
| `intersects` | Join on geometric intersection |
| `within` | Source geometries within target |
| `contains` | Target contains source geometries |
| `buffer_5km` | Buffer target by 5km, then intersect |
| `buffer_10km` | Buffer target by 10km, then intersect |
| `buffer_25km` | Buffer target by 25km, then intersect |
| `buffer_50km` | Buffer target by 50km, then intersect |

**Temporal Alignment** (`temporal_align`):
| Method | Description |
|--------|-------------|
| `exact` | No aggregation (default) |
| `yearly_sum` | Sum numeric values by year |
| `yearly_mean` | Average numeric values by year |
| `monthly_sum` | Sum numeric values by month |
| `monthly_mean` | Average numeric values by month |
| `latest` | Keep most recent record per geometry |

**Raster Aggregation** (`aggregation`):
| Method | Description |
|--------|-------------|
| `mean` | Arithmetic mean (default) |
| `sum` | Sum of pixel values |
| `min` / `max` | Minimum / maximum value |
| `std` | Standard deviation |
| `median` | Median value |
| `mode` | Most frequent value |
| `count` | Count of valid pixels |

**Resolution Formats**:
- `"5km"` - Kilometers
- `"1000m"` - Meters
- `"0.5deg"` - Decimal degrees

### 3. Build Pipelines with Checkpointing

```python
from geopipe import Pipeline, task

@task(cache=True, checkpoint=True)
def download_imagery(region, year):
    ...

@task(cache=True, resources={"memory": "16GB"})
def extract_features(imagery_path, model="resnet50"):
    ...

pipeline = Pipeline([download_imagery, extract_features])
pipeline.run(resume=True)  # Resume from last checkpoint
```

**Checkpoint Behavior:**
- Task checkpoints stored in `.geopipe/checkpoints/` (configurable via `checkpoint_dir`)
- Pipeline state stored in `.geopipe/pipeline/{name}_state.json`
- Resume skips completed stages and re-executes from failure point
- Clear checkpoints with `geopipe clean` or delete `.geopipe/` directory

```python
# Configure checkpoint location
pipeline = Pipeline(
    [download_imagery, extract_features],
    checkpoint_dir="my_checkpoints/",
)

# Force fresh run (ignore existing checkpoints)
pipeline.run(resume=False)

# Resume from last successful stage
pipeline.run(resume=True)
```

### 4. Run on HPC Clusters

```python
from geopipe.cluster import SLURMExecutor

executor = SLURMExecutor(
    partition="gpu",
    nodes=10,
    time_limit="24:00:00",
    conda_env="geopipe",
)

pipeline.run(executor=executor)
executor.status()  # Monitor progress
```

### 5. Manage Robustness Specifications

Run parallel analysis variants to assess sensitivity of results:

```python
from geopipe.specs import Spec, SpecRegistry

# Define analysis variants
specs = SpecRegistry([
    Spec("MAIN", buffer_km=5, include_ntl=True),
    Spec("ROBUST_BUFFER", buffer_km=10, include_ntl=True),
    Spec("ROBUST_NO_NTL", buffer_km=5, include_ntl=False),
])

# Run each specification
for spec in specs:
    schema.output = f"results/{spec.name}/estimates.csv"
    pipeline = Pipeline.from_schema(schema)
    pipeline.run()

# Compare results across specifications
specs.compare_results(
    pattern="{spec}/estimates.csv",
    estimate_col="estimate",
    se_col="std_error",
)

# Generate LaTeX table for publication
latex = specs.to_latex(
    pattern="{spec}/estimates.csv",
    estimate_col="estimate",
    se_col="std_error",
    caption="Robustness Specifications",
)
```

## Use Case: Satellite Imagery Causal Inference

geopipe was designed for workflows like estimating causal effects from satellite imagery across multiple countries, integrating:

- **Satellite data**: VIIRS nightlights, MODIS land cover, Sentinel-2 imagery
- **Conflict data**: ACLED, UCDP
- **Climate data**: CHIRPS precipitation, ERA5 temperature
- **Development data**: World Bank indicators, DHS surveys
- **Treatment data**: Aid project locations (AidData, IATI)

Instead of coordinating 25+ scripts manually:

```
Analysis/
├── 01_setup.R
├── 02_get_conflict.R
├── 03_get_climate.R
├── ... (11 data prep scripts)
├── call_CI_analysis.R
└── consolidate_results.R
```

Use a single declarative pipeline:

```python
schema = FusionSchema.from_yaml("sources.yaml")
pipeline = Pipeline.from_schema(schema)
pipeline.add_stage(run_cnn_inference)
pipeline.add_stage(estimate_causal_effects)
pipeline.run(executor=SLURMExecutor(), specs=["MAIN", "ROBUST_BUFFER"])
```

## Data Format Requirements

### Coordinate Reference System
All sources default to **EPSG:4326 (WGS84)**. Buffer operations automatically convert to EPSG:3857 for accurate distance calculations.

### Raster Sources
- **Formats**: GeoTIFF, Cloud-Optimized GeoTIFF (COG)
- **Bands**: 1-indexed (e.g., `band=1` for first band)
- **Nodata**: Auto-detected from file metadata or explicit parameter

### Tabular Sources
- **Formats**: CSV, Parquet, Excel (.xlsx), JSON
- **Required columns** (one of):
  - `latitude` + `longitude` columns (configurable via `lat_col`, `lon_col`)
  - `geometry` column with WKT strings

### Temporal Format
ISO 8601 strings: `"2020-01-01"` or `"2020-01-01T00:00:00"`

### Bounds Format
Tuple of `(minx, miny, maxx, maxy)` in WGS84 coordinates.

## Troubleshooting

### Console Output
geopipe uses colored terminal output via `rich`:
- **Blue**: Progress information
- **Green**: Success messages
- **Yellow**: Warnings and retries
- **Red**: Errors

### Common Issues

**File not found errors:**
```python
# Validate sources before execution
issues = schema.validate_sources()
if issues:
    print("\n".join(issues))
```

**Task failures with retry:**
```python
@task(retries=3, retry_delay=5.0)  # Retry 3x with exponential backoff
def flaky_download(url):
    ...
```

**Checkpoint corruption:**
```bash
# Clear all checkpoints and restart
geopipe clean
# Or manually: rm -rf .geopipe/
```

## Using geopipe from R

R users can access geopipe via the `reticulate` package. Results are returned as GeoDataFrames that convert directly to `sf` objects.

### Setup

```r
install.packages(c("reticulate", "sf"))
library(reticulate)

# Create conda environment with dependencies
conda_create("geopipe-env")
conda_install("geopipe-env", c("gdal", "proj"), channel = "conda-forge")
py_install("geopipe", envname = "geopipe-env")
```

### Basic Usage

```r
library(reticulate)
library(sf)
use_condaenv("geopipe-env")

# Import geopipe
geopipe <- import("geopipe")

# Define fusion schema
schema <- geopipe$FusionSchema(
  name = "analysis",
  resolution = "5km",
  sources = list(
    geopipe$RasterSource("nightlights", path = "data/viirs/*.tif", aggregation = "mean"),
    geopipe$TabularSource("conflict", path = "data/acled.csv", spatial_join = "buffer_10km")
  ),
  output = "results/fused.parquet"
)

# Execute and convert to sf
result <- schema$execute()
result_sf <- st_as_sf(result)
```

### YAML-based Workflow

Load a pre-configured YAML schema to minimize Python syntax:

```r
schema <- geopipe$FusionSchema$from_yaml("sources.yaml")
result_sf <- st_as_sf(schema$execute())
```

## Documentation

- [Data Sources](docs/sources.md)
- [Fusion Schemas](docs/fusion.md)
- [Pipeline Orchestration](docs/pipeline.md)
- [Cluster Computing](docs/cluster.md)
- [Robustness Specifications](docs/specs.md)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use geopipe in your research, please cite:

```bibtex
@software{geopipe,
  author = {Jerzak, Connor T.},
  title = {geopipe: Geospatial Data Pipeline Framework},
  year = {2025},
  url = {https://github.com/cjerzak/geopipe-software}
}
```
