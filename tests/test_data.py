"""Tests for data management module."""

import pytest
import numpy as np
from pathlib import Path
from exercise_tracker.data import DataManager, VideoProcessor
from exercise_tracker.pose_processing import PoseEmbedder


def test_data_manager_initialization(tmp_path):
    """Test data manager initialization."""
    manager = DataManager(data_root=str(tmp_path))
    
    assert manager.raw_dir.exists()
    assert manager.processed_dir.exists()


def test_data_manager_list_exercises(tmp_path):
    """Test listing exercises."""
    manager = DataManager(data_root=str(tmp_path))
    
    # Create exercise directory
    exercise_dir = manager.get_exercise_dir("squats", raw=True)
    exercise_dir.mkdir(parents=True, exist_ok=True)
    
    exercises = manager.list_exercises()
    assert "squats" in exercises


def test_video_processor_initialization():
    """Test video processor initialization."""
    processor = VideoProcessor()
    
    assert processor.pose_extractor is not None
    assert processor.normalizer is not None


@pytest.mark.skip(reason="Requires actual video file")
def test_video_processor_process_video():
    """Test video processing (requires video file)."""
    processor = VideoProcessor()
    
    # This would require an actual video file
    # video_path = "tests/fixtures/sample_videos/test.mp4"
    # cache_dir = "data/processed/test"
    # result = processor.process_video(video_path, cache_dir=cache_dir)
    # assert "keypoints" in result
    # assert "timestamps" in result
    pass


def test_video_processor_cache(tmp_path):
    """Test video processor caching."""
    processor = VideoProcessor()
    
    # Create a dummy cache file structure
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Test cache path generation
    video_path = Path("test_video.mp4")
    cache_path = processor._get_cache_path(video_path, str(cache_dir))
    
    assert cache_path.parent == cache_dir
    assert cache_path.name == "test_video.npz"

