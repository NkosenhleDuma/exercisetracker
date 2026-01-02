"""Utility functions for pose manipulation."""

import numpy as np
from typing import Optional, Tuple


def compute_centroid(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of keypoints (weighted by confidence).

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence]

    Returns:
        Centroid as [x, y]
    """
    if keypoints is None or len(keypoints) == 0:
        return np.array([0.0, 0.0])

    # Weight by confidence
    confidences = keypoints[:, 2]
    valid_mask = confidences > 0.1  # Filter low confidence points

    if not np.any(valid_mask):
        return np.array([0.0, 0.0])

    positions = keypoints[valid_mask, :2]
    weights = confidences[valid_mask]

    centroid = np.average(positions, axis=0, weights=weights)
    return centroid


def compute_scale(keypoints: np.ndarray, reference_length: Optional[float] = None) -> float:
    """
    Compute scale factor based on keypoint spread.

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence]
        reference_length: Optional reference length for normalization

    Returns:
        Scale factor
    """
    if keypoints is None or len(keypoints) == 0:
        return 1.0

    confidences = keypoints[:, 2]
    valid_mask = confidences > 0.1

    if not np.any(valid_mask):
        return 1.0

    positions = keypoints[valid_mask, :2]

    # Compute bounding box diagonal as scale measure
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    bbox_size = np.linalg.norm(max_coords - min_coords)

    if bbox_size < 1e-6:
        return 1.0

    if reference_length is not None:
        return reference_length / bbox_size

    return 1.0 / bbox_size


def compute_orientation(keypoints: np.ndarray) -> float:
    """
    Compute orientation angle from shoulder line.

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence]

    Returns:
        Orientation angle in radians
    """
    if keypoints is None or len(keypoints) < 13:
        return 0.0

    # Use shoulder line (left_shoulder=11, right_shoulder=12 in MediaPipe)
    left_shoulder_idx = 11
    right_shoulder_idx = 12

    if (
        keypoints[left_shoulder_idx, 2] < 0.1
        or keypoints[right_shoulder_idx, 2] < 0.1
    ):
        return 0.0

    left_shoulder = keypoints[left_shoulder_idx, :2]
    right_shoulder = keypoints[right_shoulder_idx, :2]

    # Compute angle from horizontal
    vec = right_shoulder - left_shoulder
    angle = np.arctan2(vec[1], vec[0])

    return angle


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate points around origin by given angle.

    Args:
        points: Array of shape (N, 2) with [x, y] coordinates
        angle: Rotation angle in radians

    Returns:
        Rotated points
    """
    cos_a = np.cos(-angle)  # Negative for counter-clockwise
    sin_a = np.sin(-angle)

    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = points @ rotation_matrix.T

    return rotated


def filter_low_confidence(keypoints: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Filter out low confidence keypoints.

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence]
        threshold: Minimum confidence threshold

    Returns:
        Filtered keypoints (same shape, low confidence set to [0, 0, 0])
    """
    filtered = keypoints.copy()
    low_conf_mask = filtered[:, 2] < threshold
    filtered[low_conf_mask] = [0.0, 0.0, 0.0]
    return filtered


def flip_keypoints_horizontally(keypoints: np.ndarray) -> np.ndarray:
    """
    Flip keypoints horizontally by swapping left/right pairs and negating x coordinates.

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence]
                  MediaPipe format with 33 keypoints

    Returns:
        Horizontally flipped keypoints
    """
    if keypoints is None or len(keypoints) == 0:
        return keypoints

    flipped = keypoints.copy()

    # MediaPipe keypoint indices for left/right pairs
    # Based on MediaPipe Pose landmark structure
    left_right_pairs = [
        (1, 4),   # left_eye_inner <-> right_eye_inner
        (2, 5),   # left_eye <-> right_eye
        (3, 6),   # left_eye_outer <-> right_eye_outer
        (7, 8),   # left_ear <-> right_ear
        (9, 10),  # mouth_left <-> mouth_right
        (11, 12), # left_shoulder <-> right_shoulder
        (13, 14), # left_elbow <-> right_elbow
        (15, 16), # left_wrist <-> right_wrist
        (17, 18), # left_pinky <-> right_pinky
        (19, 20), # left_index <-> right_index
        (21, 22), # left_thumb <-> right_thumb
        (23, 24), # left_hip <-> right_hip
        (25, 26), # left_knee <-> right_knee
        (27, 28), # left_ankle <-> right_ankle
        (29, 30), # left_heel <-> right_heel
        (31, 32), # left_foot_index <-> right_foot_index
    ]

    # Swap left/right pairs
    for left_idx, right_idx in left_right_pairs:
        if left_idx < len(flipped) and right_idx < len(flipped):
            # Swap positions and confidences
            flipped[[left_idx, right_idx]] = flipped[[right_idx, left_idx]]

    # Negate x coordinates for all keypoints (mirror effect)
    flipped[:, 0] = -flipped[:, 0]

    return flipped

