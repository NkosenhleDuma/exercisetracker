"""Compute tolerance bands around canonical trajectory."""

import numpy as np
from typing import Tuple, Optional
from .manifold_builder import ManifoldBuilder


class ToleranceBand:
    """Compute and manage tolerance bands for form assessment."""

    def __init__(
        self,
        manifold_builder: ManifoldBuilder,
        tolerance_multiplier: float = 2.0,
    ):
        """
        Initialize tolerance band.

        Args:
            manifold_builder: ManifoldBuilder instance with built manifold
            tolerance_multiplier: Multiplier for standard deviation to define tolerance
        """
        self.manifold_builder = manifold_builder
        self.tolerance_multiplier = tolerance_multiplier

    def get_tolerance_band(self, phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get upper and lower bounds of tolerance band at given phase.

        Args:
            phase: Phase value in [0, 1)

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        mean_pose, std_pose = self.manifold_builder.get_canonical_pose(phase)

        lower_bound = mean_pose - self.tolerance_multiplier * std_pose
        upper_bound = mean_pose + self.tolerance_multiplier * std_pose

        return lower_bound, upper_bound

    def is_within_tolerance(
        self, embedding: np.ndarray, phase: float
    ) -> Tuple[bool, float]:
        """
        Check if embedding is within tolerance band at given phase.

        Args:
            embedding: Embedding vector
            phase: Phase value in [0, 1)

        Returns:
            Tuple of (is_within, deviation_score) where:
            - is_within: Boolean indicating if within tolerance
            - deviation_score: Normalized deviation (0 = perfect, >1 = outside tolerance)
        """
        total_deviation, _ = self.manifold_builder.compute_deviation(embedding, phase)
        mean_pose, std_pose = self.manifold_builder.get_canonical_pose(phase)

        # Normalize deviation by tolerance
        tolerance = self.tolerance_multiplier * np.mean(std_pose)
        deviation_score = total_deviation / (tolerance + 1e-6)

        is_within = deviation_score <= 1.0

        return is_within, deviation_score

