# geopipe Bash Scripts

Shell scripts for running geopipe tasks locally and on clusters using GNU parallel.

## Prerequisites

```bash
# Install GNU parallel
brew install parallel   # macOS
apt install parallel    # Ubuntu/Debian

# First-time setup
parallel --citation     # Agree to citation notice
```

## Scripts

### `run_tests.sh`
Run the pytest test suite with coverage.

```bash
./bash_scripts/run_tests.sh
./bash_scripts/run_tests.sh -k "test_fusion"  # Run specific tests
```

### `run_batch.sh`
Generic batch runner for any geopipe task type.

```bash
# Usage: ./run_batch.sh <NameTag> <TaskType> [StartAt] [StopAt]

# Run fusion tasks 1-50 on M4 Mac
./bash_scripts/run_batch.sh M4 fusion 1 50

# Run pipeline tasks 1-100 on cluster
./bash_scripts/run_batch.sh Cluster pipeline 1 100

# Task types: fusion, pipeline, download, extract
```

### `run_fusion_parallel.sh`
Run multiple fusion schema files in parallel.

```bash
# Usage: ./run_fusion_parallel.sh <NameTag> [StartAt] [StopAt]
./bash_scripts/run_fusion_parallel.sh LocalRun 1 10
```

### `run_specs_parallel.sh`
Run robustness specifications (MAIN, ROBUST_*) in parallel.

```bash
# Usage: ./run_specs_parallel.sh <NameTag> <schema_file>
./bash_scripts/run_specs_parallel.sh M4 configs/schema.yaml
```

### `run_pipeline_parallel.sh`
Run pipeline batches in parallel.

```bash
# Usage: ./run_pipeline_parallel.sh <NameTag> <pipeline_script> [StartBatch] [StopBatch]
./bash_scripts/run_pipeline_parallel.sh M4 scripts/my_pipeline.py 1 100
```

### `check_status.sh`
Check status of running/completed parallel jobs.

```bash
./bash_scripts/check_status.sh           # All jobs
./bash_scripts/check_status.sh M4        # Jobs with NameTag=M4
./bash_scripts/check_status.sh M4 fusion # Specific task type
```

## Machine Tags (NameTag)

| Tag | Description | Parallel Jobs |
|-----|-------------|---------------|
| `Studio` | Mac Studio (conservative) | 1 |
| `M4` | M4 Mac | 4 |
| `Cluster` | Generic cluster | 16 |
| `TACC` | TACC/Stampede | 48 |
| Other | Auto-detect (half CPUs) | varies |

## Logs

All logs are saved to `bash_scripts/logs/`:
- `*_log.txt` - GNU parallel job log (tracks success/failure)
- `*_out.out` - Stdout from all jobs
- `*_err.err` - Stderr from all jobs

## Tips

### Monitor running jobs
```bash
tail -f bash_scripts/logs/fusion_M4_out.out
```

### Resume failed jobs
GNU parallel's `--resume-failed` flag automatically skips successful jobs.

### Check job counts
```bash
wc -l bash_scripts/logs/*_log.txt
```

### Kill all parallel jobs
```bash
pkill -f "parallel.*geopipe"
```
