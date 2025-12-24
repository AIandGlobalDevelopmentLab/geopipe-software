#!/usr/bin/env bash
# run_specs_parallel.sh - Run robustness specifications in parallel
# Usage: ./run_specs_parallel.sh <NameTag> <schema_file>
# Example: ./run_specs_parallel.sh LocalRun configs/schema.yaml
set -ex
cd "$(dirname "$0")/.."

# Check parallel is available
parallel --version || { echo "GNU parallel not installed. Install with: brew install parallel"; exit 1; }

ulimit -a
parallel --number-of-cpus
parallel --number-of-cores

export NameTag=${1:-"LocalRun"}
export SCHEMA_FILE=${2:-"configs/schema.yaml"}

# Define specifications to run
export SPECS="MAIN ROBUST_BUFFER ROBUST_NO_NTL ROBUST_STRICT"

# Configure parallelism based on machine
if [ "$NameTag" = "Studio" ]; then
    export nParallelJobs=1
elif [ "$NameTag" = "M4" ]; then
    export nParallelJobs=4
elif [ "$NameTag" = "Cluster" ]; then
    export nParallelJobs=16
else
    export nParallelJobs=$(echo "scale=0; $(nproc 2>/dev/null || sysctl -n hw.ncpu) / 2 + 1" | bc)
fi

echo "Running specifications in parallel: ${SPECS}"
echo "nParallelJobs=${nParallelJobs}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run each specification in parallel
nohup parallel \
    --jobs ${nParallelJobs} \
    --joblog ./bash_scripts/logs/specs_${NameTag}_log.txt \
    --load 90% \
    --delay 1 \
    'python -c "
import sys
from geopipe import FusionSchema
from geopipe.specs import Spec

spec_name = \"{}\"
print(f\"Running specification: {spec_name}\")

# Load schema and modify for this spec
schema = FusionSchema.from_yaml(\"'${SCHEMA_FILE}'\")
schema.output = f\"results/{spec_name}/consolidated.parquet\"

# Apply spec-specific parameters
if spec_name == \"ROBUST_BUFFER\":
    # Modify sources for larger buffer
    for source in schema.sources:
        if hasattr(source, \"spatial_join\") and \"buffer\" in str(source.spatial_join):
            source.spatial_join = \"buffer_10km\"
elif spec_name == \"ROBUST_NO_NTL\":
    # Remove nightlights source
    schema.sources = [s for s in schema.sources if s.name != \"nightlights\"]

schema.execute()
print(f\"Completed: {spec_name}\")
" {}' \
    ::: ${SPECS} \
    > ./bash_scripts/logs/specs_${NameTag}_out.out \
    2> ./bash_scripts/logs/specs_${NameTag}_err.err &

echo "Specification jobs submitted. Check logs in bash_scripts/logs/"
echo "  Job log: bash_scripts/logs/specs_${NameTag}_log.txt"
echo "  Stdout:  bash_scripts/logs/specs_${NameTag}_out.out"
echo "  Stderr:  bash_scripts/logs/specs_${NameTag}_err.err"
