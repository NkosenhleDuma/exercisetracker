"""Tests for pose extraction module."""

import pytest
import numpy as np
from exercise_tracker.pose_extraction import MediaPipeExtractor, BasePoseExtractor


def test_mediapipe_extractor_initialization():
    """Test MediaPipe extractor initialization."""
    extractor = MediaPipeExtractor()
    assert extractor is not None
    assert extractor.get_num_keypoints() == 33
    assert len(extractor.get_keypoint_names()) == 33


def test_mediapipe_extractor_keypoint_names():
    """Test keypoint names."""
    extractor = MediaPipeExtractor()
    names = extractor.get_keypoint_names()
    assert "nose" in names
    assert "left_shoulder" in names
    assert "right_shoulder" in names
    assert "left_hip" in names
    assert "right_hip" in names


def test_mediapipe_extract_pose_synthetic():
    """Test pose extraction from synthetic frame."""
    extractor = MediaPipeExtractor()
    
    # Create a synthetic frame (dummy test - MediaPipe may not detect pose in random image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(128)  # Gray image
    
    # This may return None if no pose detected, which is expected
    pose = extractor.extract_pose(frame)
    # Just verify it doesn't crash
    assert pose is None or pose.shape[1] == 3  # [x, y, confidence]


def test_base_extractor_interface():
    """Test that MediaPipeExtractor implements BasePoseExtractor interface."""
    extractor = MediaPipeExtractor()
    assert isinstance(extractor, BasePoseExtractor)


@pytest.mark.skip(reason="Requires actual video file")
def test_extract_poses_from_video():
    """Test extracting poses from video (requires video file)."""
    extractor = MediaPipeExtractor()
    # This would require an actual video file
    # video_path = "tests/fixtures/sample_videos/test.mp4"
    # poses, timestamps = extractor.extract_poses_from_video(video_path)
    # assert len(poses) > 0
    # assert len(timestamps) == len(poses)
    pass

