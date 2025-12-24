#!/usr/bin/env bash
# run_tests.sh - Run geopipe test suite
set -ex
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run pytest with coverage
pytest tests/ \
    --verbose \
    --cov=geopipe \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    "$@"

echo "Tests complete. Coverage report: htmlcov/index.html"
