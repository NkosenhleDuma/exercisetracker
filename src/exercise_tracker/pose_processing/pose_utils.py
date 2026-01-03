"""Utility functions for pose manipulation."""

import numpy as np
from typing import Optional, Tuple


def compute_centroid(keypoints: np.ndarray, use_3d: bool = False) -> np.ndarray:
    """
    Compute the centroid of keypoints (weighted by confidence).

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence] for 2D,
                  or (num_keypoints, 4) with [x, y, z, visibility] for 3D
        use_3d: If True, expects 3D keypoints

    Returns:
        Centroid as [x, y] for 2D or [x, y, z] for 3D
    """
    if keypoints is None or len(keypoints) == 0:
        return np.array([0.0, 0.0, 0.0]) if use_3d else np.array([0.0, 0.0])

    # Extract confidences/visibility
    if use_3d:
        confidences = keypoints[:, 3]
        valid_mask = confidences > 0.1
        positions = keypoints[valid_mask, :3]  # x, y, z
    else:
        confidences = keypoints[:, 2]
        valid_mask = confidences > 0.1
        positions = keypoints[valid_mask, :2]  # x, y

    if not np.any(valid_mask):
        return np.array([0.0, 0.0, 0.0]) if use_3d else np.array([0.0, 0.0])

    weights = confidences[valid_mask]
    centroid = np.average(positions, axis=0, weights=weights)
    return centroid


def compute_scale(keypoints: np.ndarray, reference_length: Optional[float] = None, use_3d: bool = False) -> float:
    """
    Compute scale factor based on keypoint spread.

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence] for 2D,
                  or (num_keypoints, 4) with [x, y, z, visibility] for 3D
        reference_length: Optional reference length for normalization
        use_3d: If True, expects 3D keypoints

    Returns:
        Scale factor
    """
    if keypoints is None or len(keypoints) == 0:
        return 1.0

    # Extract confidences/visibility
    if use_3d:
        confidences = keypoints[:, 3]
        valid_mask = confidences > 0.1
        positions = keypoints[valid_mask, :3]  # x, y, z
    else:
        confidences = keypoints[:, 2]
        valid_mask = confidences > 0.1
        positions = keypoints[valid_mask, :2]  # x, y

    if not np.any(valid_mask):
        return 1.0

    if use_3d:
        # For 3D, use characteristic length (shoulder-to-hip distance)
        # MediaPipe indices: shoulders=11,12; hips=23,24
        if len(positions) > 24:
            # Use hip distance
            hip_left = positions[23]
            hip_right = positions[24]
            char_length = np.linalg.norm(hip_right - hip_left)
        elif len(positions) > 12:
            # Use shoulder distance
            shoulder_left = positions[11]
            shoulder_right = positions[12]
            char_length = np.linalg.norm(shoulder_right - shoulder_left)
        else:
            # Fallback: use max distance between any two points
            from scipy.spatial.distance import pdist
            if len(positions) > 1:
                char_length = np.max(pdist(positions))
            else:
                return 1.0
    else:
        # 2D: Compute bounding box diagonal as scale measure
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        char_length = np.linalg.norm(max_coords - min_coords)

    if char_length < 1e-6:
        return 1.0

    if reference_length is not None:
        return reference_length / char_length

    return 1.0 / char_length


def compute_orientation(keypoints: np.ndarray, use_3d: bool = False) -> float:
    """
    Compute orientation angle from shoulder line.

    Args:
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence] for 2D,
                  or (num_keypoints, 4) with [x, y, z, visibility] for 3D
        use_3d: If True, expects 3D keypoints (but orientation is still computed in 2D plane)

    Returns:
        Orientation angle in radians (computed from x-y projection)
    """
    if keypoints is None or len(keypoints) < 13:
        return 0.0

    # Use shoulder line (left_shoulder=11, right_shoulder=12 in MediaPipe)
    left_shoulder_idx = 11
    right_shoulder_idx = 12

    # Extract confidence/visibility
    if use_3d:
        left_conf = keypoints[left_shoulder_idx, 3]
        right_conf = keypoints[right_shoulder_idx, 3]
        left_shoulder = keypoints[left_shoulder_idx, :2]  # Use x, y only
        right_shoulder = keypoints[right_shoulder_idx, :2]
    else:
        left_conf = keypoints[left_shoulder_idx, 2]
        right_conf = keypoints[right_shoulder_idx, 2]
        left_shoulder = keypoints[left_shoulder_idx, :2]
        right_shoulder = keypoints[right_shoulder_idx, :2]

    if left_conf < 0.1 or right_conf < 0.1:
        return 0.0

    # Compute angle from horizontal (in x-y plane)
    vec = right_shoulder - left_shoulder
    angle = np.arctan2(vec[1], vec[0])

    return angle


def rotate_points(points: np.ndarray, angle: float, use_3d: bool = False) -> np.ndarray:
    """
    Rotate points around origin by given angle.

    Args:
        points: Array of shape (N, 2) with [x, y] coordinates for 2D,
               or (N, 3) with [x, y, z] for 3D
        angle: Rotation angle in radians (applied in x-y plane)
        use_3d: If True, expects 3D points (z is preserved, rotation in x-y plane)

    Returns:
        Rotated points (same shape as input)
    """
    cos_a = np.cos(-angle)  # Negative for counter-clockwise
    sin_a = np.sin(-angle)

    if use_3d:
        # 3D rotation in x-y plane (z preserved)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        # 2D rotation
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
        keypoints: Array of shape (num_keypoints, 3) with [x, y, confidence] for 2D,
                  or (num_keypoints, 4) with [x, y, z, visibility] for 3D
                  MediaPipe format with 33 keypoints

    Returns:
        Horizontally flipped keypoints (same shape as input)
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
            # Swap all columns (positions and confidences/visibility)
            flipped[[left_idx, right_idx]] = flipped[[right_idx, left_idx]]

    # Negate x coordinates for all keypoints (mirror effect)
    # z coordinate is preserved (depth doesn't flip)
    flipped[:, 0] = -flipped[:, 0]

    return flipped

