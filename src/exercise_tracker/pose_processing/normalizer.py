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
        use_3d: bool = False,
    ):
        """
        Initialize pose normalizer.

        Args:
            normalize_position: Whether to center poses (translate to origin)
            normalize_scale: Whether to normalize scale
            normalize_orientation: Whether to normalize orientation
            reference_length: Reference length for scale normalization
            use_3d: If True, expects 3D keypoints (x, y, z, visibility);
                   if False, expects 2D (x, y, visibility)
        """
        self.normalize_position = normalize_position
        self.normalize_scale = normalize_scale
        self.normalize_orientation = normalize_orientation
        self.reference_length = reference_length
        self.use_3d = use_3d

    def normalize(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Normalize a single pose.

        Args:
            keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence] for 2D,
                      or (num_keypoints, 4) with [x, y, z, visibility] for 3D

        Returns:
            Normalized keypoints with same shape, or None if input is None
        """
        if keypoints is None:
            return None

        normalized = keypoints.copy()

        # Determine dimensionality from input
        is_3d = normalized.shape[1] == 4 if normalized.shape[1] in [3, 4] else False
        
        # Extract positions and confidences/visibility
        if is_3d:
            positions = normalized[:, :3]  # x, y, z
            confidences = normalized[:, 3]
        else:
            positions = normalized[:, :2]  # x, y
            confidences = normalized[:, 2]

        # Normalize position (center)
        if self.normalize_position:
            centroid = compute_centroid(normalized, use_3d=is_3d)
            positions = positions - centroid

        # Normalize orientation (in x-y plane, z preserved for 3D)
        if self.normalize_orientation:
            angle = compute_orientation(normalized, use_3d=is_3d)
            positions = rotate_points(positions, angle, use_3d=is_3d)

        # Normalize scale
        if self.normalize_scale:
            scale = compute_scale(normalized, self.reference_length, use_3d=is_3d)
            positions = positions * scale

        # Reconstruct keypoints
        if is_3d:
            normalized[:, :3] = positions
            normalized[:, 3] = confidences
        else:
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
            If 2D: Flattened pose vector of shape (2 * num_keypoints,)
            If 3D: Flattened pose vector of shape (3 * num_keypoints,)
            Returns None if input is None
        """
        if keypoints is None:
            return None

        # Determine dimensionality
        if keypoints.shape[1] == 4:
            # 3D: extract x, y, z
            positions = keypoints[:, :3]
        elif keypoints.shape[1] == 3:
            # 2D: extract x, y
            positions = keypoints[:, :2]
        else:
            raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")

        return positions.flatten()

