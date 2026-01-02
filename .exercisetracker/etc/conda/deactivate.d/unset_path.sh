#!/bin/bash
# Remove src directory from PYTHONPATH when conda environment is deactivated

# Get the project root directory (parent of .exercisetracker)
# Script is at .exercisetracker/etc/conda/deactivate.d/unset_path.sh
# So we need to go up 4 levels: deactivate.d -> conda -> etc -> .exercisetracker -> project root
SRC_DIR="${PROJECT_ROOT}/src"

# Remove src from PYTHONPATH
if [[ -n "$PYTHONPATH" ]]; then
    # Remove the src directory from PYTHONPATH
    PYTHONPATH=$(echo "$PYTHONPATH" | sed "s|:${SRC_DIR}||g" | sed "s|${SRC_DIR}:||g" | sed "s|${SRC_DIR}||g")
    export PYTHONPATH
    echo "Removed ${SRC_DIR} from PYTHONPATH"
    unset PROJECT_ROOT
fi

