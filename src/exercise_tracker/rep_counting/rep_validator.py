"""Rep validator to filter out noise and validate rep completeness."""

import numpy as np
from typing import List, Tuple, Dict, Optional


class RepValidator:
    """Validate and filter exercise reps."""

    def __init__(
        self,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        min_phase_coverage: float = 0.7,
        min_velocity: float = 0.1,
    ):
        """
        Initialize rep validator.

        Args:
            min_duration: Minimum rep duration in seconds
            max_duration: Maximum rep duration in seconds
            min_phase_coverage: Minimum phase coverage (0-1)
            min_velocity: Minimum average phase velocity
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_phase_coverage = min_phase_coverage
        self.min_velocity = min_velocity

    def validate_rep(
        self,
        start_idx: int,
        end_idx: int,
        phases: List[Optional[float]],
        timestamps: List[float],
    ) -> Tuple[bool, str]:
        """
        Validate a single rep.

        Args:
            start_idx: Start frame index
            end_idx: End frame index
            phases: List of phase values
            timestamps: List of timestamps

        Returns:
            Tuple of (is_valid, reason) where reason explains why invalid
        """
        if start_idx >= end_idx:
            return False, "Invalid indices: start >= end"

        duration = timestamps[end_idx] - timestamps[start_idx]

        if duration < self.min_duration:
            return False, f"Duration too short: {duration:.2f}s < {self.min_duration:.2f}s"

        if duration > self.max_duration:
            return False, f"Duration too long: {duration:.2f}s > {self.max_duration:.2f}s"

        # Check phase coverage
        rep_phases = [
            p % 1.0 for p in phases[start_idx : end_idx + 1] if p is not None
        ]

        if len(rep_phases) == 0:
            return False, "No valid phases in rep"

        phase_range = max(rep_phases) - min(rep_phases)
        if phase_range < self.min_phase_coverage:
            return False, f"Phase coverage too low: {phase_range:.2f} < {self.min_phase_coverage:.2f}"

        # Check phase velocity
        phase_velocity = phase_range / duration if duration > 0 else 0.0
        if phase_velocity < self.min_velocity:
            return False, f"Phase velocity too low: {phase_velocity:.2f} < {self.min_velocity:.2f}"

        return True, "Valid rep"

    def filter_reps(
        self,
        rep_boundaries: List[Tuple[int, int]],
        phases: List[Optional[float]],
        timestamps: List[float],
    ) -> List[Tuple[int, int]]:
        """
        Filter rep boundaries to keep only valid reps.

        Args:
            rep_boundaries: List of (start_idx, end_idx) tuples
            phases: List of phase values
            timestamps: List of timestamps

        Returns:
            List of valid rep boundaries
        """
        valid_reps = []

        for start_idx, end_idx in rep_boundaries:
            is_valid, reason = self.validate_rep(
                start_idx, end_idx, phases, timestamps
            )

            if is_valid:
                valid_reps.append((start_idx, end_idx))

        return valid_reps

    def compute_rep_quality(
        self,
        start_idx: int,
        end_idx: int,
        phases: List[Optional[float]],
        timestamps: List[float],
    ) -> Dict[str, float]:
        """
        Compute quality metrics for a rep.

        Args:
            start_idx: Start frame index
            end_idx: End frame index
            phases: List of phase values
            timestamps: List of timestamps

        Returns:
            Dictionary of quality metrics
        """
        duration = timestamps[end_idx] - timestamps[start_idx]

        rep_phases = [
            p % 1.0 for p in phases[start_idx : end_idx + 1] if p is not None
        ]

        phase_range = max(rep_phases) - min(rep_phases) if rep_phases else 0.0
        phase_velocity = phase_range / duration if duration > 0 else 0.0

        # Compute phase smoothness (lower std = smoother)
        phase_std = np.std(rep_phases) if len(rep_phases) > 1 else 0.0
        smoothness = 1.0 / (1.0 + phase_std)  # Normalize to [0, 1]

        return {
            "duration": duration,
            "phase_range": phase_range,
            "phase_velocity": phase_velocity,
            "smoothness": smoothness,
        }

