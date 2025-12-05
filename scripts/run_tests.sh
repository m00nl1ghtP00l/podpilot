#!/bin/bash
# Simple test runner script

# Check if pytest is installed
if ! python3 -m pytest --version &> /dev/null; then
    echo "pytest is not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run tests
python3 -m pytest "$@"

