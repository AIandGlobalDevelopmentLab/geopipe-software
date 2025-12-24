#!/usr/bin/env bash
# check_status.sh - Check status of parallel jobs
# Usage: ./check_status.sh [NameTag] [TaskType]
set -e
cd "$(dirname "$0")"

export NameTag=${1:-"*"}
export TaskType=${2:-"*"}

echo "========================================"
echo "Job Status Check"
echo "========================================"

# Find and display job logs
for logfile in logs/${TaskType}_${NameTag}_log.txt; do
    if [ -f "$logfile" ]; then
        echo ""
        echo "Log: $logfile"
        echo "----------------------------------------"

        # Parse joblog format: Seq Host Starttime Jobruntime Send Receive Exitval Signal Command
        total=$(tail -n +2 "$logfile" | wc -l | xargs)
        success=$(tail -n +2 "$logfile" | awk '$7 == 0' | wc -l | xargs)
        failed=$(tail -n +2 "$logfile" | awk '$7 != 0' | wc -l | xargs)
        running=$(ps aux | grep -c "parallel" || echo "0")

        echo "Total jobs:     $total"
        echo "Successful:     $success"
        echo "Failed:         $failed"

        if [ "$failed" -gt 0 ]; then
            echo ""
            echo "Failed commands:"
            tail -n +2 "$logfile" | awk '$7 != 0 {print "  Exit " $7 ": " $NF}'
        fi
    fi
done

# Show recent stderr output if any
for errfile in logs/${TaskType}_${NameTag}_err.err; do
    if [ -f "$errfile" ] && [ -s "$errfile" ]; then
        echo ""
        echo "Recent errors from $errfile:"
        echo "----------------------------------------"
        tail -20 "$errfile"
    fi
done

echo ""
echo "========================================"
