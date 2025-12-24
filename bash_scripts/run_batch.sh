#!/usr/bin/env bash
# run_batch.sh - Generic batch runner for geopipe tasks
# Modeled after your asa-software batch runner pattern
# Usage: ./run_batch.sh <NameTag> <TaskType> [StartAt] [StopAt]
# Example: ./run_batch.sh M4 fusion 1 50
set -ex
cd "$(dirname "$0")/.."

# Check parallel is available
parallel --version || { echo "GNU parallel not installed. Install with: brew install parallel"; exit 1; }

ulimit -a
parallel --number-of-cpus
parallel --number-of-cores
parallel --number-of-threads

export NameTag=${1:-"LocalRun"}
export TaskType=${2:-"fusion"}
export StartAt=${3:-1}
export StopAt=${4:-1}

# Configure parallelism based on machine tag
if [ "$NameTag" = "Studio" ]; then
    export nParallelJobs=1
elif [ "$NameTag" = "M4" ]; then
    export nParallelJobs=4
elif [ "$NameTag" = "TACC" ]; then
    # For TACC/Stampede
    export nParallelJobs=48
elif [ "$NameTag" = "Cluster" ]; then
    export nParallelJobs=16
else
    export nParallelJobs=$(echo "scale=0; $(nproc 2>/dev/null || sysctl -n hw.ncpu) / 2 + 1" | bc)
fi

echo "========================================"
echo "geopipe Batch Runner"
echo "========================================"
echo "NameTag:       ${NameTag}"
echo "TaskType:      ${TaskType}"
echo "Jobs:          ${StartAt} to ${StopAt}"
echo "Parallel Jobs: ${nParallelJobs}"
echo "========================================"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Select command based on task type
case ${TaskType} in
    "fusion")
        COMMAND='geopipe fuse configs/batch_{}.yaml --output results/batch_{}.parquet'
        ;;
    "pipeline")
        COMMAND='python scripts/run_pipeline.py --batch {}'
        ;;
    "download")
        COMMAND='python scripts/download_data.py --region {}'
        ;;
    "extract")
        COMMAND='python scripts/extract_features.py --batch {}'
        ;;
    *)
        echo "Unknown task type: ${TaskType}"
        echo "Valid types: fusion, pipeline, download, extract"
        exit 1
        ;;
esac

# Create log directory
mkdir -p ./bash_scripts/logs

# Run tasks in parallel with nohup for background execution
nohup parallel \
    --jobs ${nParallelJobs} \
    --joblog ./bash_scripts/logs/${TaskType}_${NameTag}_log.txt \
    --load 90% \
    --delay 1 \
    --resume-failed \
    "${COMMAND}" \
    ::: $(seq ${StartAt} ${StopAt}) \
    > ./bash_scripts/logs/${TaskType}_${NameTag}_out.out \
    2> ./bash_scripts/logs/${TaskType}_${NameTag}_err.err &

echo ""
echo "Jobs submitted in background."
echo "Logs:"
echo "  Job log: bash_scripts/logs/${TaskType}_${NameTag}_log.txt"
echo "  Stdout:  bash_scripts/logs/${TaskType}_${NameTag}_out.out"
echo "  Stderr:  bash_scripts/logs/${TaskType}_${NameTag}_err.err"
echo ""
echo "Monitor with: tail -f bash_scripts/logs/${TaskType}_${NameTag}_out.out"
echo "Check status: parallel --joblog bash_scripts/logs/${TaskType}_${NameTag}_log.txt --resume-failed echo ::: $(seq ${StartAt} ${StopAt})"
