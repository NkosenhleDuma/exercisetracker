"""MoveNet pose extractor implementation (optional)."""

import numpy as np
from typing import List, Optional, Tuple

from .base_extractor import BasePoseExtractor


class MoveNetExtractor(BasePoseExtractor):
    """
    MoveNet-based pose extractor (placeholder for future implementation).

    MoveNet can be integrated here if needed. For now, this is a placeholder
    that follows the same interface as MediaPipeExtractor.
    """

    KEYPOINT_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self):
        """Initialize MoveNet extractor."""
        raise NotImplementedError(
            "MoveNet extractor not yet implemented. Use MediaPipeExtractor instead."
        )

    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose keypoints from a single frame."""
        raise NotImplementedError

    def extract_poses_from_video(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract poses from all frames in a video."""
        raise NotImplementedError

    def get_keypoint_names(self) -> List[str]:
        """Get the names of keypoints in order."""
        return self.KEYPOINT_NAMES.copy()

    def get_num_keypoints(self) -> int:
        """Get the number of keypoints."""
        return len(self.KEYPOINT_NAMES)

