"""Abstract base class for pose extractors."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class BasePoseExtractor(ABC):
    """Abstract base class for pose extraction from video frames."""

    @abstractmethod
    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from a single frame.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            Pose keypoints as numpy array of shape (num_keypoints, 3) where
            each row is [x, y, confidence], or None if no pose detected
        """
        pass

    @abstractmethod
    def extract_poses_from_video(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract poses from all frames in a video.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (pose_sequence, timestamps) where:
            - pose_sequence: List of pose arrays, one per frame
            - timestamps: List of timestamps in seconds for each frame
        """
        pass

    @abstractmethod
    def get_keypoint_names(self) -> List[str]:
        """
        Get the names of keypoints in order.

        Returns:
            List of keypoint names
        """
        pass

    @abstractmethod
    def get_num_keypoints(self) -> int:
        """
        Get the number of keypoints.

        Returns:
            Number of keypoints
        """
        pass

