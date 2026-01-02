"""Build canonical manifolds from reference trajectories."""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


class ManifoldBuilder:
    """Build canonical manifolds from (z, θ) pairs."""

    def __init__(self, num_phase_bins: int = 100):
        """
        Initialize manifold builder.

        Args:
            num_phase_bins: Number of phase bins for discretization
        """
        self.num_phase_bins = num_phase_bins
        self.phase_bins = np.linspace(0, 1, num_phase_bins, endpoint=False)
        self.manifold_mean = None
        self.manifold_std = None
        self.embed_dim = None

    def build_manifold(
        self, labeled_data: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build canonical manifold from (embedding, phase) pairs.

        Args:
            labeled_data: List of (embedding, phase) tuples where phase ∈ [0, 1)

        Returns:
            Tuple of (mean_manifold, std_manifold) where:
            - mean_manifold: Array of shape (num_phase_bins, embed_dim) with mean pose at each phase
            - std_manifold: Array of shape (num_phase_bins, embed_dim) with std at each phase
        """
        if len(labeled_data) == 0:
            raise ValueError("No labeled data provided")

        # Extract embeddings and phases
        embeddings = np.array([emb for emb, _ in labeled_data])
        phases = np.array([phase for _, phase in labeled_data])

        self.embed_dim = embeddings.shape[1]

        # Initialize arrays
        mean_manifold = np.zeros((self.num_phase_bins, self.embed_dim))
        std_manifold = np.zeros((self.num_phase_bins, self.embed_dim))

        # Build manifold for each phase bin
        for i, phase_center in enumerate(self.phase_bins):
            # Find data points in this phase bin
            # Use circular distance for phase (wraps around)
            phase_diff = np.abs(phases - phase_center)
            phase_diff = np.minimum(phase_diff, 1.0 - phase_diff)  # Handle wrap-around

            bin_width = 1.0 / self.num_phase_bins
            in_bin = phase_diff < bin_width / 2

            if np.any(in_bin):
                bin_embeddings = embeddings[in_bin]
                mean_manifold[i] = np.mean(bin_embeddings, axis=0)
                std_manifold[i] = np.std(bin_embeddings, axis=0)
            else:
                # No data in this bin, interpolate from neighbors
                if i > 0 and i < self.num_phase_bins - 1:
                    mean_manifold[i] = (mean_manifold[i - 1] + mean_manifold[i + 1]) / 2
                    std_manifold[i] = (std_manifold[i - 1] + std_manifold[i + 1]) / 2
                elif i == 0:
                    mean_manifold[i] = mean_manifold[i + 1] if self.num_phase_bins > 1 else np.zeros(self.embed_dim)
                    std_manifold[i] = std_manifold[i + 1] if self.num_phase_bins > 1 else np.ones(self.embed_dim)
                else:
                    mean_manifold[i] = mean_manifold[i - 1]
                    std_manifold[i] = std_manifold[i - 1]

        self.manifold_mean = mean_manifold
        self.manifold_std = std_manifold

        return mean_manifold, std_manifold

    def get_canonical_pose(self, phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get canonical pose (mean and std) at a given phase.

        Args:
            phase: Phase value in [0, 1)

        Returns:
            Tuple of (mean_pose, std_pose) at the given phase
        """
        if self.manifold_mean is None:
            raise ValueError("Manifold not built. Call build_manifold first.")

        phase = phase % 1.0  # Ensure in [0, 1)

        # Find nearest phase bin
        phase_diff = np.abs(self.phase_bins - phase)
        phase_diff = np.minimum(phase_diff, 1.0 - phase_diff)  # Handle wrap-around
        bin_idx = np.argmin(phase_diff)

        return self.manifold_mean[bin_idx], self.manifold_std[bin_idx]

    def compute_deviation(
        self, embedding: np.ndarray, phase: float
    ) -> Tuple[float, np.ndarray]:
        """
        Compute deviation of embedding from canonical pose at given phase.

        Args:
            embedding: Embedding vector
            phase: Phase value in [0, 1)

        Returns:
            Tuple of (total_deviation, per_dimension_deviations)
        """
        canonical_mean, canonical_std = self.get_canonical_pose(phase)

        # Normalized deviation (Mahalanobis-like distance)
        deviation = embedding - canonical_mean
        normalized_deviation = deviation / (canonical_std + 1e-6)  # Add epsilon to avoid division by zero

        total_deviation = np.linalg.norm(normalized_deviation)
        per_dimension = np.abs(normalized_deviation)

        return total_deviation, per_dimension

