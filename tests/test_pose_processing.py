"""Tests for pose processing module."""

import pytest
import numpy as np
from exercise_tracker.pose_processing import PoseNormalizer, PoseEmbedder
from exercise_tracker.pose_processing.pose_utils import (
    compute_centroid,
    compute_scale,
    compute_orientation,
    rotate_points,
)


def test_compute_centroid():
    """Test centroid computation."""
    keypoints = np.array([
        [0.5, 0.5, 1.0],
        [0.6, 0.6, 1.0],
        [0.4, 0.4, 0.5],
    ])
    centroid = compute_centroid(keypoints)
    assert centroid.shape == (2,)
    assert 0.4 < centroid[0] < 0.6
    assert 0.4 < centroid[1] < 0.6


def test_compute_scale():
    """Test scale computation."""
    keypoints = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    scale = compute_scale(keypoints)
    assert scale > 0


def test_rotate_points():
    """Test point rotation."""
    points = np.array([[1.0, 0.0], [0.0, 1.0]])
    rotated = rotate_points(points, np.pi / 2)
    assert rotated.shape == points.shape


def test_pose_normalizer():
    """Test pose normalization."""
    normalizer = PoseNormalizer()
    
    # Create synthetic keypoints
    keypoints = np.array([
        [0.5, 0.5, 1.0],
        [0.6, 0.6, 1.0],
        [0.4, 0.4, 1.0],
    ])
    
    normalized = normalizer.normalize(keypoints)
    assert normalized is not None
    assert normalized.shape == keypoints.shape
    
    # Check that position normalization centers the pose
    pose_vector = normalizer.get_pose_vector(normalized)
    assert pose_vector is not None
    assert len(pose_vector) == keypoints.shape[0] * 2


def test_pose_normalizer_sequence():
    """Test normalizing a sequence of poses."""
    normalizer = PoseNormalizer()
    
    keypoints1 = np.array([[0.5, 0.5, 1.0], [0.6, 0.6, 1.0]])
    keypoints2 = np.array([[0.4, 0.4, 1.0], [0.5, 0.5, 1.0]])
    
    sequence = [keypoints1, keypoints2]
    normalized = normalizer.normalize_sequence(sequence)
    
    assert len(normalized) == 2
    assert all(n is not None for n in normalized)


def test_pose_embedder():
    """Test pose embedding."""
    embedder = PoseEmbedder(n_components=16)
    
    # Create synthetic pose vectors
    pose_vectors = [
        np.random.randn(66),  # 33 keypoints * 2 (x, y)
        np.random.randn(66),
        np.random.randn(66),
    ]
    
    # Fit embedder
    embedder.fit(pose_vectors)
    assert embedder.is_fitted
    
    # Embed a pose
    embedding = embedder.embed(pose_vectors[0])
    assert embedding is not None
    assert len(embedding) == 16


def test_pose_embedder_sequence():
    """Test embedding a sequence."""
    embedder = PoseEmbedder(n_components=16)
    
    pose_vectors = [np.random.randn(66) for _ in range(5)]
    embedder.fit(pose_vectors)
    
    embeddings = embedder.embed_sequence(pose_vectors)
    assert len(embeddings) == 5
    assert all(e is not None for e in embeddings)


def test_pose_embedder_save_load(tmp_path):
    """Test saving and loading embedder."""
    embedder = PoseEmbedder(n_components=16)
    
    pose_vectors = [np.random.randn(66) for _ in range(5)]
    embedder.fit(pose_vectors)
    
    # Save
    filepath = tmp_path / "embedder.pkl"
    embedder.save(str(filepath))
    
    # Load
    loaded = PoseEmbedder.load(str(filepath))
    assert loaded.is_fitted
    assert loaded.n_components == 16
    
    # Test embedding with loaded model
    embedding = loaded.embed(pose_vectors[0])
    assert embedding is not None

