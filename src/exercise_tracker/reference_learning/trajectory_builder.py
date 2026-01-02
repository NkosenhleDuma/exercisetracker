"""Build reference trajectories from labeled exercise reps."""

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.interpolate import interp1d


class TrajectoryBuilder:
    """Build time-normalized trajectories from exercise rep segments."""

    def __init__(self, num_phase_points: int = 100):
        """
        Initialize trajectory builder.

        Args:
            num_phase_points: Number of points to use for phase normalization
        """
        self.num_phase_points = num_phase_points

    def segment_reps(
        self,
        embeddings: List[Optional[np.ndarray]],
        timestamps: List[float],
        rep_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> List[np.ndarray]:
        """
        Segment embeddings into individual reps.

        Args:
            embeddings: List of embedding vectors
            timestamps: List of timestamps
            rep_boundaries: Optional list of (start_idx, end_idx) tuples for rep boundaries.
                          If None, treats entire sequence as one rep.

        Returns:
            List of rep trajectories, each as array of shape (num_frames_in_rep, embed_dim)
        """
        # Filter out None embeddings
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        if len(valid_indices) == 0:
            return []

        valid_embeddings = [embeddings[i] for i in valid_indices]

        if rep_boundaries is None:
            # Treat entire sequence as one rep
            return [np.array(valid_embeddings)]

        reps = []
        for start_idx, end_idx in rep_boundaries:
            # Map to valid indices
            rep_embeddings = []
            for i in range(start_idx, min(end_idx + 1, len(embeddings))):
                if i in valid_indices:
                    orig_idx = valid_indices.index(i)
                    rep_embeddings.append(valid_embeddings[orig_idx])

            if len(rep_embeddings) > 0:
                reps.append(np.array(rep_embeddings))

        return reps

    def time_normalize_rep(
        self, rep_trajectory: np.ndarray, num_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-normalize a rep trajectory to fixed number of points.

        Args:
            rep_trajectory: Array of shape (num_frames, embed_dim)
            num_points: Number of points for normalization (default: self.num_phase_points)

        Returns:
            Tuple of (normalized_trajectory, phases) where:
            - normalized_trajectory: Array of shape (num_points, embed_dim)
            - phases: Array of phase values in [0, 1) for each point
        """
        if num_points is None:
            num_points = self.num_phase_points

        num_frames, embed_dim = rep_trajectory.shape

        if num_frames == 0:
            return np.zeros((num_points, embed_dim)), np.linspace(0, 1, num_points)

        # Create phase values
        phases = np.linspace(0, 1, num_points, endpoint=False)

        # Interpolate trajectory
        original_indices = np.linspace(0, num_frames - 1, num_frames)
        normalized_trajectory = np.zeros((num_points, embed_dim))

        for dim in range(embed_dim):
            interp_func = interp1d(
                original_indices,
                rep_trajectory[:, dim],
                kind="linear",
                fill_value="extrapolate",
            )
            normalized_trajectory[:, dim] = interp_func(
                np.linspace(0, num_frames - 1, num_points)
            )

        return normalized_trajectory, phases

    def build_trajectories(
        self,
        rep_trajectories: List[np.ndarray],
        num_points: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build time-normalized trajectories from multiple reps.

        Args:
            rep_trajectories: List of rep trajectory arrays
            num_points: Number of phase points (default: self.num_phase_points)

        Returns:
            Tuple of (all_trajectories, phases) where:
            - all_trajectories: Array of shape (num_reps, num_points, embed_dim)
            - phases: Array of phase values in [0, 1)
        """
        if len(rep_trajectories) == 0:
            raise ValueError("No rep trajectories provided")

        if num_points is None:
            num_points = self.num_phase_points

        normalized_reps = []
        phases = None

        for rep in rep_trajectories:
            norm_rep, phases = self.time_normalize_rep(rep, num_points)
            normalized_reps.append(norm_rep)

        all_trajectories = np.array(normalized_reps)
        return all_trajectories, phases

    def assign_phase_labels(
        self, embeddings: List[Optional[np.ndarray]], rep_boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Assign phase labels to embeddings based on rep boundaries.

        Args:
            embeddings: List of embedding vectors
            rep_boundaries: List of (start_idx, end_idx) tuples for rep boundaries

        Returns:
            List of (embedding, phase) tuples where phase is in [0, 1)
        """
        labeled_data = []

        for start_idx, end_idx in rep_boundaries:
            rep_length = end_idx - start_idx + 1

            for i in range(start_idx, min(end_idx + 1, len(embeddings))):
                if embeddings[i] is not None:
                    # Compute phase within this rep
                    phase = (i - start_idx) / rep_length if rep_length > 0 else 0.0
                    phase = phase % 1.0  # Ensure in [0, 1)
                    labeled_data.append((embeddings[i], phase))

        return labeled_data

