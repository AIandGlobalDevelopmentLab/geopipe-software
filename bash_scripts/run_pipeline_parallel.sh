#!/usr/bin/env bash
# run_pipeline_parallel.sh - Run pipeline stages with GNU parallel
# Usage: ./run_pipeline_parallel.sh <NameTag> <pipeline_script> [StartBatch] [StopBatch]
# Example: ./run_pipeline_parallel.sh LocalRun scripts/my_pipeline.py 1 100
set -ex
cd "$(dirname "$0")/.."

# Check parallel is available
parallel --version || { echo "GNU parallel not installed. Install with: brew install parallel"; exit 1; }

ulimit -a
parallel --number-of-cpus
parallel --number-of-cores
parallel --number-of-threads
parallel --number-of-sockets

export NameTag=${1:-"LocalRun"}
export PIPELINE_SCRIPT=${2:-"scripts/run_batch.py"}
export StartAt=${3:-1}
export StopAt=${4:-1}

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

echo "Running pipeline batches ${StartAt} to ${StopAt}"
echo "nParallelJobs=${nParallelJobs}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run pipeline batches in parallel
nohup parallel \
    --jobs ${nParallelJobs} \
    --joblog ./bash_scripts/logs/pipeline_${NameTag}_log.txt \
    --load 90% \
    --delay 1 \
    --resume-failed \
    'python '${PIPELINE_SCRIPT}' --batch {} --output results/batch_{}' \
    ::: $(seq ${StartAt} ${StopAt}) \
    > ./bash_scripts/logs/pipeline_${NameTag}_out.out \
    2> ./bash_scripts/logs/pipeline_${NameTag}_err.err &

echo "Pipeline jobs submitted. Check logs in bash_scripts/logs/"
echo "  Job log: bash_scripts/logs/pipeline_${NameTag}_log.txt"
echo "  Stdout:  bash_scripts/logs/pipeline_${NameTag}_out.out"
echo "  Stderr:  bash_scripts/logs/pipeline_${NameTag}_err.err"
