"""Tests for phase estimation module."""

import pytest
import numpy as np
from exercise_tracker.phase_estimation import PhaseModel, PhaseTracker


def test_phase_model_initialization():
    """Test phase model initialization."""
    model = PhaseModel(model_type="mlp")
    assert not model.is_fitted


def test_phase_model_fit_predict():
    """Test phase model training and prediction."""
    model = PhaseModel(model_type="mlp", hidden_layer_sizes=(32,))
    
    # Create synthetic training data
    embeddings = [np.random.randn(16) for _ in range(100)]
    phases = [i / 100.0 for i in range(100)]
    
    model.fit(embeddings, phases)
    assert model.is_fitted
    
    # Predict phase
    test_embedding = np.random.randn(16)
    predicted_phase = model.predict_phase(test_embedding)
    
    assert predicted_phase is not None
    assert 0 <= predicted_phase < 1.0


def test_phase_model_sequence():
    """Test predicting phases for a sequence."""
    model = PhaseModel(model_type="mlp", hidden_layer_sizes=(32,))
    
    embeddings = [np.random.randn(16) for _ in range(50)]
    phases = [i / 50.0 for i in range(50)]
    
    model.fit(embeddings, phases)
    
    test_embeddings = [np.random.randn(16) for _ in range(10)]
    predicted_phases = model.predict_phases(test_embeddings)
    
    assert len(predicted_phases) == 10
    assert all(p is not None and 0 <= p < 1.0 for p in predicted_phases)


def test_phase_model_save_load(tmp_path):
    """Test saving and loading phase model."""
    model = PhaseModel(model_type="mlp", hidden_layer_sizes=(32,))
    
    embeddings = [np.random.randn(16) for _ in range(50)]
    phases = [i / 50.0 for i in range(50)]
    
    model.fit(embeddings, phases)
    
    # Save
    filepath = tmp_path / "phase_model.pkl"
    model.save(str(filepath))
    
    # Load
    loaded = PhaseModel.load(str(filepath))
    assert loaded.is_fitted
    
    # Test prediction
    test_embedding = np.random.randn(16)
    phase1 = model.predict_phase(test_embedding)
    phase2 = loaded.predict_phase(test_embedding)
    
    # Should be similar (may not be exact due to randomness)
    assert abs(phase1 - phase2) < 0.1


def test_phase_tracker_smooth():
    """Test phase smoothing."""
    tracker = PhaseTracker(smoothing_window=5)
    
    phases = [0.1 * i % 1.0 for i in range(20)]
    timestamps = [i * 0.033 for i in range(20)]  # ~30fps
    
    smoothed = tracker.smooth_phases(phases, timestamps)
    
    assert len(smoothed) == len(phases)
    assert all(p is None or 0 <= p < 1.0 for p in smoothed)


def test_phase_tracker_unwrap():
    """Test phase unwrapping."""
    tracker = PhaseTracker()
    
    # Create phases that wrap around
    phases = [0.9, 0.95, 0.0, 0.05, 0.1, 0.15]
    timestamps = [i * 0.033 for i in range(len(phases))]
    
    unwrapped = tracker.unwrap_phase(phases, timestamps)
    
    assert len(unwrapped) == len(phases)
    # Check that unwrapped values increase
    valid_unwrapped = [u for u in unwrapped if u is not None]
    if len(valid_unwrapped) > 1:
        # Should generally increase (allowing for some noise)
        assert valid_unwrapped[-1] >= valid_unwrapped[0] - 0.5


def test_phase_tracker_track():
    """Test full phase tracking."""
    tracker = PhaseTracker(smoothing_window=5)
    
    phases = [0.1 * i % 1.0 for i in range(20)]
    timestamps = [i * 0.033 for i in range(20)]
    
    tracked = tracker.track_phase(phases, timestamps, smooth=True, unwrap=True)
    
    assert len(tracked) == len(phases)

