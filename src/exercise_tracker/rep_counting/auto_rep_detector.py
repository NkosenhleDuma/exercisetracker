"""Automatic rep detection for reference videos without phase model."""

import numpy as np
from typing import List, Optional, Tuple
from scipy.signal import find_peaks


class AutoRepDetector:
    """Detect reps automatically from embedding sequences without phase model."""

    def __init__(
        self,
        min_rep_duration: float = 0.5,
        max_rep_duration: float = 10.0,
        similarity_threshold: float = 0.7,
        window_size: int = 5,
    ):
        """
        Initialize automatic rep detector.

        Args:
            min_rep_duration: Minimum rep duration in seconds
            max_rep_duration: Maximum rep duration in seconds
            similarity_threshold: Threshold for detecting return to start pose (0-1)
            window_size: Window size for smoothing similarity scores
        """
        self.min_rep_duration = min_rep_duration
        self.max_rep_duration = max_rep_duration
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def compute_similarity_to_start(
        self,
        embeddings: List[Optional[np.ndarray]],
        start_window: int = 10,
    ) -> List[float]:
        """
        Compute similarity of each embedding to the starting pose.

        Args:
            embeddings: List of embedding vectors
            start_window: Number of initial frames to average for start pose

        Returns:
            List of similarity scores (0-1, higher = more similar to start)
        """
        # Get valid embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        if len(valid_embeddings) == 0:
            # Return zeros for all frames if no valid embeddings
            return [0.0] * len(embeddings)

        # Compute average start embedding
        start_frames = min(start_window, len(valid_embeddings))
        start_embeddings = valid_embeddings[:start_frames]
        start_embedding = np.mean([emb for emb in start_embeddings], axis=0)

        # Normalize start embedding
        start_norm = np.linalg.norm(start_embedding)
        if start_norm < 1e-6:
            return [0.0] * len(embeddings)

        start_embedding = start_embedding / start_norm

        # Compute similarity for each frame
        similarities = []
        for emb in embeddings:
            if emb is None:
                similarities.append(0.0)
                continue

            # Normalize embedding
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-6:
                similarities.append(0.0)
                continue

            emb_normalized = emb / emb_norm

            # Cosine similarity
            similarity = np.dot(start_embedding, emb_normalized)
            # Convert to [0, 1] range (cosine similarity is [-1, 1])
            similarity = (similarity + 1.0) / 2.0
            similarities.append(similarity)

        return similarities

    def smooth_similarities(self, similarities: List[float]) -> List[float]:
        """
        Smooth similarity scores using moving average.

        Args:
            similarities: List of similarity scores

        Returns:
            Smoothed similarity scores
        """
        if len(similarities) < self.window_size:
            return similarities

        smoothed = []
        half_window = self.window_size // 2

        for i in range(len(similarities)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(similarities), i + half_window + 1)
            window = similarities[start_idx:end_idx]
            smoothed.append(np.mean(window))

        return smoothed

    def detect_rep_boundaries(
        self,
        embeddings: List[Optional[np.ndarray]],
        timestamps: List[float],
    ) -> List[Tuple[int, int]]:
        """
        Detect rep boundaries from embedding sequence.

        Args:
            embeddings: List of embedding vectors
            timestamps: List of timestamps

        Returns:
            List of (start_idx, end_idx) tuples for detected reps
        """
        if len(embeddings) == 0 or len(timestamps) == 0:
            return []

        # Compute similarity to start
        similarities = self.compute_similarity_to_start(embeddings)
        if len(similarities) == 0:
            return []

        # Smooth similarities
        smoothed = self.smooth_similarities(similarities)

        # Find peaks in similarity (when pose returns to start)
        # Invert so peaks are valleys (low similarity = far from start)
        inverted = np.array([1.0 - s for s in smoothed])

        # Find peaks in inverted signal (valleys in original = returns to start)
        # Use minimum distance between peaks based on min rep duration
        fps = len(timestamps) / timestamps[-1] if timestamps[-1] > 0 else 30.0
        min_distance = max(1, int(self.min_rep_duration * fps))

        # Find peaks with appropriate parameters
        try:
            peaks, properties = find_peaks(
                inverted,
                height=1.0 - self.similarity_threshold,
                distance=min_distance,
            )
        except ValueError:
            # If find_peaks fails, return empty boundaries
            return []

        # Convert peaks to rep boundaries
        # Each peak represents a return to start, so reps are between peaks
        rep_boundaries = []

        if len(peaks) == 0:
            # No clear peaks, treat entire sequence as one rep
            return [(0, len(embeddings) - 1)]

        # First rep: from start to first peak
        if peaks[0] > min_distance:
            rep_boundaries.append((0, int(peaks[0])))

        # Middle reps: between consecutive peaks
        for i in range(len(peaks) - 1):
            start_idx = int(peaks[i])
            end_idx = int(peaks[i + 1])
            duration = timestamps[end_idx] - timestamps[start_idx]

            if self.min_rep_duration <= duration <= self.max_rep_duration:
                rep_boundaries.append((start_idx, end_idx))

        # Last rep: from last peak to end (if there's enough time)
        if len(peaks) > 0:
            last_peak = int(peaks[-1])
            if last_peak < len(embeddings) - min_distance:
                duration = timestamps[-1] - timestamps[last_peak]
                if duration >= self.min_rep_duration:
                    rep_boundaries.append((last_peak, len(embeddings) - 1))

        # Filter by duration
        filtered_boundaries = []
        for start_idx, end_idx in rep_boundaries:
            duration = timestamps[end_idx] - timestamps[start_idx]
            if self.min_rep_duration <= duration <= self.max_rep_duration:
                filtered_boundaries.append((start_idx, end_idx))

        return filtered_boundaries

