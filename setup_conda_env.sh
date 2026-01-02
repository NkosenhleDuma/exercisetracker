#!/bin/bash
# Setup script for conda environment with activation scripts

set -e

ENV_DIR=".exercisetracker"
SCRIPT_DIR=".exercisetracker/etc/conda"

echo "Setting up conda environment..."

# Create conda environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment from conda.env.yaml..."
    conda env create -f conda.env.yaml -p "$ENV_DIR"
else
    echo "Conda environment already exists at $ENV_DIR"
fi

# Ensure activation script directories exist
mkdir -p "$SCRIPT_DIR/activate.d"
mkdir -p "$SCRIPT_DIR/deactivate.d"

# Copy activation scripts
echo "Installing activation scripts..."
cp scripts/conda_activate.sh "$SCRIPT_DIR/activate.d/setup_path.sh"
chmod +x "$SCRIPT_DIR/activate.d/setup_path.sh"

cp scripts/conda_deactivate.sh "$SCRIPT_DIR/deactivate.d/unset_path.sh"
chmod +x "$SCRIPT_DIR/deactivate.d/unset_path.sh"

# Windows scripts
cp scripts/conda_activate.bat "$SCRIPT_DIR/activate.d/setup_path.bat"
cp scripts/conda_deactivate.bat "$SCRIPT_DIR/deactivate.d/unset_path.bat"

echo "Conda environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ./.exercisetracker"
echo ""
echo "The activation script will automatically add src/ to PYTHONPATH"

