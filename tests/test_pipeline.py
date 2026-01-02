"""Tests for pipeline integration."""

import pytest
import numpy as np
from pathlib import Path
from exercise_tracker.pipeline import ExercisePipeline
from exercise_tracker.exercise_framework import Exercise, ExerciseConfig
from exercise_tracker.pose_processing import PoseEmbedder


def test_pipeline_initialization():
    """Test pipeline initialization."""
    config = ExerciseConfig(name="test_exercise")
    exercise = Exercise(config, data_root="data")
    pipeline = ExercisePipeline(exercise)
    
    assert pipeline.exercise is not None


@pytest.mark.skip(reason="Requires actual video files and trained models")
def test_pipeline_ingest():
    """Test pipeline ingestion (requires video files)."""
    config = ExerciseConfig(name="test_exercise")
    exercise = Exercise(config, data_root="data")
    pipeline = ExercisePipeline(exercise)
    
    # This would require actual video files
    # video_paths = ["tests/fixtures/sample_videos/test.mp4"]
    # result = pipeline.ingest_reference_videos(video_paths)
    # assert result["manifold_built"]
    pass


@pytest.mark.skip(reason="Requires trained models")
def test_pipeline_train():
    """Test pipeline training (requires reference data)."""
    config = ExerciseConfig(name="test_exercise")
    exercise = Exercise(config, data_root="data")
    
    # Would need to set up embedder and manifold first
    # exercise.embedder = PoseEmbedder()
    # ...
    
    pipeline = ExercisePipeline(exercise)
    # result = pipeline.train_phase_model()
    # assert result["model_trained"]
    pass


@pytest.mark.skip(reason="Requires trained models and video file")
def test_pipeline_analyze():
    """Test pipeline analysis (requires trained models and video)."""
    config = ExerciseConfig(name="test_exercise")
    exercise = Exercise(config, data_root="data")
    pipeline = ExercisePipeline(exercise)
    
    # This would require trained models and a video file
    # result = pipeline.analyze_video("tests/fixtures/sample_videos/test.mp4")
    # assert "rep_count" in result
    pass

