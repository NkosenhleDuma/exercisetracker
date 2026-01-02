"""Tests for reference learning module."""

import pytest
import numpy as np
from exercise_tracker.reference_learning import (
    TrajectoryBuilder,
    ManifoldBuilder,
    ToleranceBand,
)


def test_trajectory_builder():
    """Test trajectory builder."""
    builder = TrajectoryBuilder(num_phase_points=50)
    
    # Create synthetic rep trajectory
    rep_trajectory = np.random.randn(100, 16)  # 100 frames, 16-dim embeddings
    
    normalized, phases = builder.time_normalize_rep(rep_trajectory)
    
    assert normalized.shape[0] == 50
    assert normalized.shape[1] == 16
    assert len(phases) == 50
    assert all(0 <= p < 1.0 for p in phases)


def test_trajectory_builder_multiple_reps():
    """Test building trajectories from multiple reps."""
    builder = TrajectoryBuilder(num_phase_points=50)
    
    rep_trajectories = [
        np.random.randn(100, 16),
        np.random.randn(120, 16),
        np.random.randn(90, 16),
    ]
    
    all_trajectories, phases = builder.build_trajectories(rep_trajectories)
    
    assert all_trajectories.shape[0] == 3  # 3 reps
    assert all_trajectories.shape[1] == 50  # phase points
    assert all_trajectories.shape[2] == 16  # embed dim
    assert len(phases) == 50


def test_manifold_builder():
    """Test manifold builder."""
    builder = ManifoldBuilder(num_phase_bins=20)
    
    # Create synthetic labeled data
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    
    mean_manifold, std_manifold = builder.build_manifold(labeled_data)
    
    assert mean_manifold.shape[0] == 20
    assert mean_manifold.shape[1] == 16
    assert std_manifold.shape == mean_manifold.shape


def test_manifold_builder_get_canonical_pose():
    """Test getting canonical pose at phase."""
    builder = ManifoldBuilder(num_phase_bins=20)
    
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    mean_pose, std_pose = builder.get_canonical_pose(0.5)
    
    assert mean_pose.shape == (16,)
    assert std_pose.shape == (16,)


def test_manifold_builder_compute_deviation():
    """Test computing deviation from canonical pose."""
    builder = ManifoldBuilder(num_phase_bins=20)
    
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    embedding = np.random.randn(16)
    phase = 0.5
    
    total_dev, per_dim = builder.compute_deviation(embedding, phase)
    
    assert total_dev >= 0
    assert len(per_dim) == 16


def test_tolerance_band():
    """Test tolerance band."""
    builder = ManifoldBuilder(num_phase_bins=20)
    
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    tolerance_band = ToleranceBand(builder, tolerance_multiplier=2.0)
    
    lower, upper = tolerance_band.get_tolerance_band(0.5)
    assert lower.shape == (16,)
    assert upper.shape == (16,)
    
    embedding = np.random.randn(16)
    is_within, deviation_score = tolerance_band.is_within_tolerance(embedding, 0.5)
    
    assert isinstance(is_within, bool)
    assert deviation_score >= 0

