"""Pose normalization for position, scale, and orientation."""

import numpy as np
from typing import Optional

from .pose_utils import (
    compute_centroid,
    compute_scale,
    compute_orientation,
    rotate_points,
)


class PoseNormalizer:
    """Normalize poses for position, scale, and orientation invariance."""

    def __init__(
        self,
        normalize_position: bool = True,
        normalize_scale: bool = True,
        normalize_orientation: bool = True,
        reference_length: Optional[float] = None,
    ):
        """
        Initialize pose normalizer.

        Args:
            normalize_position: Whether to center poses (translate to origin)
            normalize_scale: Whether to normalize scale
            normalize_orientation: Whether to normalize orientation
            reference_length: Reference length for scale normalization
        """
        self.normalize_position = normalize_position
        self.normalize_scale = normalize_scale
        self.normalize_orientation = normalize_orientation
        self.reference_length = reference_length

    def normalize(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Normalize a single pose.

        Args:
            keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence]

        Returns:
            Normalized keypoints with same shape, or None if input is None
        """
        if keypoints is None:
            return None

        normalized = keypoints.copy()

        # Extract positions and confidences
        positions = normalized[:, :2]
        confidences = normalized[:, 2]

        # Normalize position (center)
        if self.normalize_position:
            centroid = compute_centroid(normalized)
            positions = positions - centroid

        # Normalize orientation
        if self.normalize_orientation:
            angle = compute_orientation(normalized)
            positions = rotate_points(positions, angle)

        # Normalize scale
        if self.normalize_scale:
            scale = compute_scale(normalized, self.reference_length)
            positions = positions * scale

        # Reconstruct keypoints
        normalized[:, :2] = positions
        normalized[:, 2] = confidences

        return normalized

    def normalize_sequence(
        self, pose_sequence: list[Optional[np.ndarray]]
    ) -> list[Optional[np.ndarray]]:
        """
        Normalize a sequence of poses.

        Args:
            pose_sequence: List of pose arrays

        Returns:
            List of normalized pose arrays
        """
        return [self.normalize(pose) for pose in pose_sequence]

    def get_pose_vector(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract pose vector (flattened positions) from normalized keypoints.

        Args:
            keypoints: Normalized keypoints array

        Returns:
            Flattened pose vector of shape (2 * num_keypoints,), or None
        """
        if keypoints is None:
            return None

        # Extract only positions (x, y) and flatten
        positions = keypoints[:, :2]
        return positions.flatten()

