#!/bin/bash
# Script to run the Universal Accessibility Reader using the virtual environment

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if venv exists
if [ -d "$DIR/venv" ]; then
    echo "Starting Universal Accessibility Reader..."
    "$DIR/venv/bin/python" "$DIR/app.py"
else
    echo "Error: Virtual environment not found. Please run installation steps first."
    exit 1
fi
