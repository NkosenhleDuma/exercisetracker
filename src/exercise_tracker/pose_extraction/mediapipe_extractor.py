"""MediaPipe pose extractor implementation."""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple

from .base_extractor import BasePoseExtractor


class MediaPipeExtractor(BasePoseExtractor):
    """MediaPipe-based pose extractor for 2D/3D keypoint extraction."""

    # MediaPipe Pose landmark indices
    KEYPOINT_NAMES = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    def __init__(
        self, 
        model_complexity: int = 1, 
        min_detection_confidence: float = 0.5,
        use_3d: bool = False
    ):
        """
        Initialize MediaPipe pose extractor.

        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for pose detection
            use_3d: If True, extract 3D coordinates (x, y, z, visibility);
                   if False, extract 2D (x, y, visibility)
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.use_3d = use_3d

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from a single frame.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            If use_3d=False: Pose keypoints as numpy array of shape (num_keypoints, 3)
                where each row is [x, y, visibility]
            If use_3d=True: Pose keypoints as numpy array of shape (num_keypoints, 4)
                where each row is [x, y, z, visibility]
            Returns None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process frame
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        landmarks = results.pose_landmarks.landmark
        
        if self.use_3d:
            # 3D mode: [x, y, z, visibility]
            keypoints = np.zeros((len(landmarks), 4))
            for i, landmark in enumerate(landmarks):
                keypoints[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        else:
            # 2D mode: [x, y, visibility] (backward compatible)
            keypoints = np.zeros((len(landmarks), 3))
            for i, landmark in enumerate(landmarks):
                keypoints[i] = [landmark.x, landmark.y, landmark.visibility]

        return keypoints

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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1.0 / fps if fps > 0 else 0.033  # Default to 30fps

        pose_sequence = []
        timestamps = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose = self.extract_pose(frame)
            pose_sequence.append(pose)
            timestamps.append(frame_idx * frame_time)
            frame_idx += 1

        cap.release()
        return pose_sequence, timestamps

    def get_keypoint_names(self) -> List[str]:
        """Get the names of keypoints in order."""
        return self.KEYPOINT_NAMES.copy()

    def get_num_keypoints(self) -> int:
        """Get the number of keypoints."""
        return len(self.KEYPOINT_NAMES)

