"""Tests for exercise framework."""

import pytest
import numpy as np
from pathlib import Path
from exercise_tracker.exercise_framework import (
    ExerciseConfig,
    Exercise,
    ExerciseRegistry,
)


def test_exercise_config():
    """Test exercise configuration."""
    config = ExerciseConfig(name="squats", min_rep_duration=0.5)
    
    assert config.name == "squats"
    assert config.min_rep_duration == 0.5
    assert config.embed_dim == 32


def test_exercise_config_save_load(tmp_path):
    """Test saving and loading config."""
    config = ExerciseConfig(name="squats", min_rep_duration=0.5)
    
    filepath = tmp_path / "config.yaml"
    config.save(str(filepath))
    
    loaded = ExerciseConfig.load(str(filepath))
    
    assert loaded.name == config.name
    assert loaded.min_rep_duration == config.min_rep_duration


def test_exercise_registry():
    """Test exercise registry."""
    registry = ExerciseRegistry()
    
    config = ExerciseConfig(name="squats")
    exercise = Exercise(config, data_root="data")
    
    registry.register(exercise)
    
    assert "squats" in registry.list_exercises()
    
    retrieved = registry.get("squats")
    assert retrieved is not None
    assert retrieved.config.name == "squats"


def test_exercise_registry_overwrite():
    """Test exercise registry overwrite."""
    registry = ExerciseRegistry()
    
    config1 = ExerciseConfig(name="squats", min_rep_duration=0.5)
    exercise1 = Exercise(config1, data_root="data")
    registry.register(exercise1)
    
    config2 = ExerciseConfig(name="squats", min_rep_duration=1.0)
    exercise2 = Exercise(config2, data_root="data")
    
    # Should raise error without overwrite
    with pytest.raises(ValueError):
        registry.register(exercise2)
    
    # Should work with overwrite
    registry.register(exercise2, overwrite=True)
    
    retrieved = registry.get("squats")
    assert retrieved.config.min_rep_duration == 1.0


def test_exercise_save_load(tmp_path):
    """Test exercise save and load."""
    config = ExerciseConfig(name="test_exercise")
    exercise = Exercise(config, data_root=str(tmp_path))
    
    # Save
    exercise.save()
    
    # Verify files created
    assert exercise.get_config_path().exists()
    
    # Load
    loaded_exercise = Exercise(config, data_root=str(tmp_path))
    loaded_exercise.load()
    
    assert loaded_exercise.config.name == "test_exercise"

