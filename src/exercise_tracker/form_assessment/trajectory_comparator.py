"""Compare user trajectory to canonical trajectory."""

import numpy as np
from typing import List, Optional, Dict, Tuple

from ..reference_learning import ManifoldBuilder


class TrajectoryComparator:
    """Compare user trajectory to canonical exercise trajectory."""

    def __init__(self, manifold_builder: ManifoldBuilder):
        """
        Initialize trajectory comparator.

        Args:
            manifold_builder: ManifoldBuilder instance with built manifold
        """
        self.manifold_builder = manifold_builder

    def compare_trajectory(
        self,
        embeddings: List[Optional[np.ndarray]],
        phases: List[Optional[float]],
    ) -> Dict[str, any]:
        """
        Compare user trajectory to canonical trajectory.

        Args:
            embeddings: List of user embedding vectors
            phases: List of phase values

        Returns:
            Dictionary with comparison metrics
        """
        deviations = []
        per_dimension_deviations = []

        for emb, phase in zip(embeddings, phases):
            if emb is None or phase is None:
                continue

            try:
                total_dev, per_dim = self.manifold_builder.compute_deviation(
                    emb, phase
                )
                deviations.append(total_dev)
                per_dimension_deviations.append(per_dim)
            except Exception:
                continue

        if len(deviations) == 0:
            return {
                "mean_deviation": 0.0,
                "max_deviation": 0.0,
                "std_deviation": 0.0,
                "trajectory_similarity": 0.0,
            }

        deviations_array = np.array(deviations)
        per_dim_array = np.array(per_dimension_deviations)

        # Compute aggregate metrics
        mean_deviation = float(np.mean(deviations_array))
        max_deviation = float(np.max(deviations_array))
        std_deviation = float(np.std(deviations_array))

        # Trajectory similarity (inverse of mean deviation, normalized)
        trajectory_similarity = float(1.0 / (1.0 + mean_deviation))

        # Per-dimension analysis
        mean_per_dim = np.mean(per_dim_array, axis=0)
        max_per_dim = np.max(per_dim_array, axis=0)

        return {
            "mean_deviation": mean_deviation,
            "max_deviation": max_deviation,
            "std_deviation": std_deviation,
            "trajectory_similarity": trajectory_similarity,
            "mean_per_dimension": mean_per_dim.tolist(),
            "max_per_dimension": max_per_dim.tolist(),
            "num_valid_frames": len(deviations),
        }

    def compare_rep(
        self,
        start_idx: int,
        end_idx: int,
        embeddings: List[Optional[np.ndarray]],
        phases: List[Optional[float]],
    ) -> Dict[str, any]:
        """
        Compare a single rep to canonical trajectory.

        Args:
            start_idx: Start frame index
            end_idx: End frame index
            embeddings: List of embedding vectors
            phases: List of phase values

        Returns:
            Dictionary with rep comparison metrics
        """
        rep_embeddings = embeddings[start_idx : end_idx + 1]
        rep_phases = phases[start_idx : end_idx + 1]

        comparison = self.compare_trajectory(rep_embeddings, rep_phases)

        # Add rep-specific info
        comparison["rep_start_idx"] = start_idx
        comparison["rep_end_idx"] = end_idx
        comparison["rep_duration_frames"] = end_idx - start_idx + 1

        return comparison

    def compute_phase_coverage(
        self, phases: List[Optional[float]]
    ) -> Dict[str, float]:
        """
        Compute phase coverage statistics.

        Args:
            phases: List of phase values

        Returns:
            Dictionary with phase coverage metrics
        """
        valid_phases = [p % 1.0 for p in phases if p is not None]

        if len(valid_phases) == 0:
            return {
                "coverage": 0.0,
                "min_phase": 0.0,
                "max_phase": 0.0,
                "phase_range": 0.0,
            }

        phases_array = np.array(valid_phases)

        # Compute coverage by binning phases
        num_bins = 20
        bins = np.linspace(0, 1, num_bins + 1)
        hist, _ = np.histogram(phases_array, bins=bins)
        coverage = float(np.sum(hist > 0) / num_bins)

        return {
            "coverage": coverage,
            "min_phase": float(np.min(phases_array)),
            "max_phase": float(np.max(phases_array)),
            "phase_range": float(np.max(phases_array) - np.min(phases_array)),
        }

