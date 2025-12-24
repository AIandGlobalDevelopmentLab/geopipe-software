# ImageDeconfoundAid Example

This example demonstrates how to use geopipe to replicate a satellite imagery
causal inference pipeline similar to the ImageDeconfoundAid project.

## Overview

The pipeline integrates:
- Satellite imagery (VIIRS nightlights, MODIS land cover)
- Conflict data (ACLED events)
- Climate data (CHIRPS precipitation)
- Development data (World Bank indicators)
- Treatment data (Aid project locations)

## Usage

1. Create the sources configuration:

```yaml
# sources.yaml
name: aid_effects
resolution: 5km
temporal_range:
  - "2000-01-01"
  - "2020-12-31"

sources:
  - type: raster
    name: nightlights
    path: data/viirs/*.tif
    aggregation: mean

  - type: raster
    name: landcover
    path: data/modis/*.tif
    aggregation: mode

  - type: tabular
    name: conflict
    path: data/acled.csv
    lat_col: latitude
    lon_col: longitude
    time_col: event_date
    spatial_join: buffer_10km
    temporal_align: yearly_sum

  - type: tabular
    name: aid_wb
    path: data/worldbank_aid.csv
    spatial_join: buffer_5km

  - type: tabular
    name: aid_china
    path: data/china_aid.csv
    spatial_join: buffer_5km

output: data/fused/consolidated.parquet
```

2. Run the fusion:

```bash
geopipe fuse sources.yaml
```

3. Or use the Python API:

```python
from geopipe import FusionSchema, Pipeline
from geopipe.cluster import SLURMExecutor
from geopipe.specs import SpecRegistry, Spec

# Load schema
schema = FusionSchema.from_yaml("sources.yaml")

# Define robustness specifications
specs = SpecRegistry([
    Spec("MAIN", buffer_km=5, include_ntl=True),
    Spec("ROBUST_BUFFER", buffer_km=10, include_ntl=True),
    Spec("ROBUST_NO_NTL", buffer_km=5, include_ntl=False),
    Spec("ROBUST_STRICT", buffer_km=5, include_ntl=True, strict_matching=True),
])

# Run on cluster
executor = SLURMExecutor(
    partition="gpu",
    nodes=10,
    time_limit="24:00:00",
    conda_env="geopipe",
)

for spec in specs:
    schema.output = f"results/{spec.name}/consolidated.parquet"
    pipeline = Pipeline.from_schema(schema)
    pipeline.run(executor=executor)

# Compare results
specs.compare_results("{spec}/estimates.csv")
```
