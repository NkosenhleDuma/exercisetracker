"""Tests for form assessment module."""

import pytest
import numpy as np
from exercise_tracker.form_assessment import FormScorer, TrajectoryComparator
from exercise_tracker.reference_learning import ManifoldBuilder


def test_form_scorer():
    """Test form scorer."""
    # Build a simple manifold
    builder = ManifoldBuilder(num_phase_bins=20)
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    scorer = FormScorer(builder)
    
    # Score a frame
    embedding = np.random.randn(16)
    phase = 0.5
    
    score = scorer.score_frame(embedding, phase)
    
    assert score is not None
    assert 0 <= score <= 1.0


def test_form_scorer_sequence():
    """Test scoring a sequence."""
    builder = ManifoldBuilder(num_phase_bins=20)
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    scorer = FormScorer(builder)
    
    embeddings = [np.random.randn(16) for _ in range(10)]
    phases = [i / 10.0 for i in range(10)]
    
    frame_scores, summary = scorer.score_sequence(embeddings, phases)
    
    assert len(frame_scores) == 10
    assert "mean_score" in summary
    assert "valid_frames" in summary


def test_trajectory_comparator():
    """Test trajectory comparator."""
    builder = ManifoldBuilder(num_phase_bins=20)
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    comparator = TrajectoryComparator(builder)
    
    embeddings = [np.random.randn(16) for _ in range(10)]
    phases = [i / 10.0 for i in range(10)]
    
    comparison = comparator.compare_trajectory(embeddings, phases)
    
    assert "mean_deviation" in comparison
    assert "trajectory_similarity" in comparison
    assert 0 <= comparison["trajectory_similarity"] <= 1.0


def test_trajectory_comparator_rep():
    """Test comparing a single rep."""
    builder = ManifoldBuilder(num_phase_bins=20)
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    comparator = TrajectoryComparator(builder)
    
    embeddings = [np.random.randn(16) for _ in range(20)]
    phases = [i / 20.0 for i in range(20)]
    
    comparison = comparator.compare_rep(0, 19, embeddings, phases)
    
    assert "rep_start_idx" in comparison
    assert "rep_end_idx" in comparison


def test_phase_coverage():
    """Test phase coverage computation."""
    builder = ManifoldBuilder(num_phase_bins=20)
    labeled_data = [
        (np.random.randn(16), i / 100.0) for i in range(100)
    ]
    builder.build_manifold(labeled_data)
    
    comparator = TrajectoryComparator(builder)
    
    phases = [i / 50.0 % 1.0 for i in range(50)]
    
    coverage = comparator.compute_phase_coverage(phases)
    
    assert "coverage" in coverage
    assert 0 <= coverage["coverage"] <= 1.0

