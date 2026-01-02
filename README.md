# Exercise Rep Counter and Form Checker

A pose-trajectory based system for automatically counting exercise reps and assessing form from video without sensors or wearables.

## Overview

This system models exercises as loops in pose-space with continuous phase labels. It:
- Extracts 2D poses from video using MediaPipe
- Learns canonical exercise manifolds from reference videos
- Estimates phase progression through exercise reps
- Counts valid reps by tracking phase wraps
- Assesses form quality by comparing trajectories to canonical patterns

## Architecture

The system is structured as a modular Python package:

- **Pose Extraction**: MediaPipe integration for 2D keypoint extraction
- **Pose Processing**: Normalization and embedding to latent space
- **Reference Learning**: Build canonical manifolds from reference videos
- **Phase Estimation**: Map pose embeddings to phase (sin/cos representation)
- **Rep Counting**: Track phase wraps and count valid reps
- **Form Assessment**: Compare user trajectory to canonical manifold
- **Exercise Framework**: Configurable exercise definitions
- **CLI Interface**: Fire CLI for video processing and analysis

## Installation

### Using Conda (Recommended)

```bash
# Run setup script to create environment and install activation scripts
./setup_conda_env.sh

# Or manually:
# Create conda environment in .exercisetracker directory
conda env create -f conda.env.yaml -p .exercisetracker

# Copy activation scripts to environment
mkdir -p .exercisetracker/etc/conda/activate.d
mkdir -p .exercisetracker/etc/conda/deactivate.d
cp scripts/conda_activate.sh .exercisetracker/etc/conda/activate.d/setup_path.sh
cp scripts/conda_deactivate.sh .exercisetracker/etc/conda/deactivate.d/unset_path.sh
cp scripts/conda_activate.bat .exercisetracker/etc/conda/activate.d/setup_path.bat
cp scripts/conda_deactivate.bat .exercisetracker/etc/conda/deactivate.d/unset_path.bat
chmod +x .exercisetracker/etc/conda/activate.d/setup_path.sh
chmod +x .exercisetracker/etc/conda/deactivate.d/unset_path.sh

# Activate the environment
conda activate ./.exercisetracker

# The activation script will automatically add src to PYTHONPATH
# Install the package in development mode
pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Usage

### CLI Commands

```bash
# Process reference videos and build manifolds
python -m exercise_tracker.cli ingest --exercise squats --video-dir data/raw/squats

# Analyze a video for rep count and form
python -m exercise_tracker.cli analyze --exercise squats --video path/to/video.mp4

# Train phase estimation model
python -m exercise_tracker.cli train --exercise squats

# List registered exercises
python -m exercise_tracker.cli list-exercises
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/exercise_tracker
```

## Project Structure

```
ExerciseTracker2/
├── src/exercise_tracker/    # Main package
├── tests/                    # Test suite
├── data/                     # Data directories
│   ├── raw/                  # Reference videos
│   └── processed/            # Cached pose data
└── pyproject.toml            # Project configuration
```

