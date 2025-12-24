#!/usr/bin/env bash
# run_fusion_parallel.sh - Run multiple fusion jobs in parallel using GNU parallel
# Usage: ./run_fusion_parallel.sh <NameTag> [StartAt] [StopAt]
# Example: ./run_fusion_parallel.sh LocalRun 1 10
set -ex
cd "$(dirname "$0")/.."

# Check parallel is available
parallel --version || { echo "GNU parallel not installed. Install with: brew install parallel"; exit 1; }

ulimit -a
parallel --number-of-cpus
parallel --number-of-cores
parallel --number-of-threads

export NameTag=${1:-"LocalRun"}
export StartAt=${2:-1}
export StopAt=${3:-1}

# Configure parallelism based on machine
if [ "$NameTag" = "Studio" ]; then
    export nParallelJobs=1
elif [ "$NameTag" = "M4" ]; then
    export nParallelJobs=4
elif [ "$NameTag" = "Cluster" ]; then
    export nParallelJobs=16
else
    # Default: use half the CPUs
    export nParallelJobs=$(echo "scale=0; $(nproc 2>/dev/null || sysctl -n hw.ncpu) / 2 + 1" | bc)
fi

echo "Running with nParallelJobs=${nParallelJobs}, jobs ${StartAt} to ${StopAt}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run fusion jobs in parallel
# Each job processes a different schema file or batch index
nohup parallel \
    --jobs ${nParallelJobs} \
    --joblog ./bash_scripts/logs/fusion_${NameTag}_log.txt \
    --load 90% \
    --delay 1 \
    'python -c "
from geopipe import FusionSchema
schema = FusionSchema.from_yaml(\"configs/schema_{}.yaml\")
schema.execute()
" {}' \
    ::: $(seq ${StartAt} ${StopAt}) \
    > ./bash_scripts/logs/fusion_${NameTag}_out.out \
    2> ./bash_scripts/logs/fusion_${NameTag}_err.err &

echo "Jobs submitted. Check logs in bash_scripts/logs/"
echo "  Job log: bash_scripts/logs/fusion_${NameTag}_log.txt"
echo "  Stdout:  bash_scripts/logs/fusion_${NameTag}_out.out"
echo "  Stderr:  bash_scripts/logs/fusion_${NameTag}_err.err"
