#!/bin/bash
# Add src directory to PYTHONPATH when conda environment is activated

# Get the project root directory (parent of .exercisetracker)
# Script is at .exercisetracker/etc/conda/activate.d/setup_path.sh
# So we need to go up 4 levels: activate.d -> conda -> etc -> .exercisetracker -> project root
export PROJECT_ROOT="$(pwd)"
SRC_DIR="${PROJECT_ROOT}/src"

# Add src to PYTHONPATH if not already present
if [[ ":$PYTHONPATH:" != *":${SRC_DIR}:"* ]]; then
    export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"
    echo "Added ${SRC_DIR} to PYTHONPATH"
fi

