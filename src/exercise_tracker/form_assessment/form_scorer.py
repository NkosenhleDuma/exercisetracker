"""Form scorer that compares user trajectory to canonical manifold."""

import numpy as np
from typing import List, Optional, Dict, Tuple

from ..reference_learning import ManifoldBuilder


class FormScorer:
    """Score exercise form by comparing to canonical manifold."""

    def __init__(self, manifold_builder: ManifoldBuilder):
        """
        Initialize form scorer.

        Args:
            manifold_builder: ManifoldBuilder instance with built manifold
        """
        self.manifold_builder = manifold_builder

    def score_frame(
        self, embedding: Optional[np.ndarray], phase: Optional[float]
    ) -> Optional[float]:
        """
        Score a single frame's form.

        Args:
            embedding: Embedding vector
            phase: Phase value in [0, 1)

        Returns:
            Form score (0-1, higher is better), or None if inputs are invalid
        """
        if embedding is None or phase is None:
            return None

        try:
            total_deviation, _ = self.manifold_builder.compute_deviation(
                embedding, phase
            )

            # Convert deviation to score (0-1, higher is better)
            # Use exponential decay: score = exp(-deviation / scale)
            scale = 1.0  # Adjust this to tune sensitivity
            score = np.exp(-total_deviation / scale)

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return None

    def score_sequence(
        self,
        embeddings: List[Optional[np.ndarray]],
        phases: List[Optional[float]],
    ) -> Tuple[List[Optional[float]], Dict[str, float]]:
        """
        Score form for a sequence of frames.

        Args:
            embeddings: List of embedding vectors
            phases: List of phase values

        Returns:
            Tuple of (frame_scores, summary_stats) where:
            - frame_scores: List of scores for each frame
            - summary_stats: Dictionary with aggregate statistics
        """
        frame_scores = []

        for emb, phase in zip(embeddings, phases):
            score = self.score_frame(emb, phase)
            frame_scores.append(score)

        # Compute summary statistics
        valid_scores = [s for s in frame_scores if s is not None]

        if len(valid_scores) == 0:
            summary_stats = {
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "std_score": 0.0,
                "valid_frames": 0,
            }
        else:
            summary_stats = {
                "mean_score": float(np.mean(valid_scores)),
                "min_score": float(np.min(valid_scores)),
                "max_score": float(np.max(valid_scores)),
                "std_score": float(np.std(valid_scores)),
                "valid_frames": len(valid_scores),
            }

        return frame_scores, summary_stats

    def score_rep(
        self,
        start_idx: int,
        end_idx: int,
        embeddings: List[Optional[np.ndarray]],
        phases: List[Optional[float]],
    ) -> Dict[str, float]:
        """
        Score form for a single rep.

        Args:
            start_idx: Start frame index
            end_idx: End frame index
            embeddings: List of embedding vectors
            phases: List of phase values

        Returns:
            Dictionary with rep form scores
        """
        rep_embeddings = embeddings[start_idx : end_idx + 1]
        rep_phases = phases[start_idx : end_idx + 1]

        frame_scores, summary_stats = self.score_sequence(
            rep_embeddings, rep_phases
        )

        # Add rep-specific metrics
        rep_scores = summary_stats.copy()
        rep_scores["rep_start_idx"] = start_idx
        rep_scores["rep_end_idx"] = end_idx
        rep_scores["num_frames"] = end_idx - start_idx + 1

        return rep_scores

