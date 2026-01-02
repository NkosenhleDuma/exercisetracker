"""Pose extraction module for extracting 2D keypoints from video."""

from .base_extractor import BasePoseExtractor
from .mediapipe_extractor import MediaPipeExtractor

__all__ = ["BasePoseExtractor", "MediaPipeExtractor"]

