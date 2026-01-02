"""Phase tracking with smoothing and unwrapping."""

import numpy as np
from typing import List, Optional
from scipy.signal import savgol_filter


class PhaseTracker:
    """Track continuous phase progression with smoothing and unwrapping."""

    def __init__(
        self,
        smoothing_window: int = 5,
        min_rep_duration: float = 0.5,
        min_phase_coverage: float = 0.7,
    ):
        """
        Initialize phase tracker.

        Args:
            smoothing_window: Window size for phase smoothing (must be odd)
            min_rep_duration: Minimum duration in seconds for a valid rep
            min_phase_coverage: Minimum phase coverage (0-1) for a valid rep
        """
        self.smoothing_window = smoothing_window
        self.min_rep_duration = min_rep_duration
        self.min_phase_coverage = min_phase_coverage

    def smooth_phases(
        self,
        phases: List[Optional[float]],
        timestamps: List[float],
    ) -> List[Optional[float]]:
        """
        Smooth phase sequence to reduce noise.

        Args:
            phases: List of phase values
            timestamps: List of timestamps

        Returns:
            List of smoothed phase values
        """
        # Filter out None values for smoothing
        valid_indices = [i for i, phase in enumerate(phases) if phase is not None]
        if len(valid_indices) < self.smoothing_window:
            return phases  # Not enough data for smoothing

        valid_phases = np.array([phases[i] for i in valid_indices])

        # Apply Savitzky-Golay filter for smoothing
        if len(valid_phases) >= self.smoothing_window:
            # Ensure window is odd
            window = self.smoothing_window
            if window % 2 == 0:
                window += 1

            try:
                smoothed = savgol_filter(valid_phases, window, 3)
            except ValueError:
                # Fallback if smoothing fails
                smoothed = valid_phases
        else:
            smoothed = valid_phases

        # Reconstruct full sequence
        smoothed_phases = phases.copy()
        for i, orig_idx in enumerate(valid_indices):
            smoothed_phases[orig_idx] = smoothed[i] % 1.0  # Ensure in [0, 1)

        return smoothed_phases

    def unwrap_phase(
        self,
        phases: List[Optional[float]],
        timestamps: List[float],
    ) -> List[Optional[float]]:
        """
        Unwrap phase to continuous angle that increases through reps.

        Args:
            phases: List of phase values in [0, 1)
            timestamps: List of timestamps

        Returns:
            List of unwrapped phase values (can exceed 1.0)
        """
        if len(phases) == 0:
            return []

        unwrapped = []
        last_phase = None
        total_wraps = 0.0

        for phase in phases:
            if phase is None:
                unwrapped.append(None)
                continue

            if last_phase is not None:
                # Check for phase wrap (forward or backward)
                phase_diff = phase - last_phase

                # Handle wrap-around: if phase jumps from ~1.0 to ~0.0, it's a forward wrap
                if phase_diff < -0.5:
                    total_wraps += 1.0
                # If phase jumps from ~0.0 to ~1.0, it's a backward wrap (unlikely but possible)
                elif phase_diff > 0.5:
                    total_wraps -= 1.0

            unwrapped_phase = phase + total_wraps
            unwrapped.append(unwrapped_phase)
            last_phase = phase

        return unwrapped

    def track_phase(
        self,
        phases: List[Optional[float]],
        timestamps: List[float],
        smooth: bool = True,
        unwrap: bool = True,
    ) -> List[Optional[float]]:
        """
        Track phase with optional smoothing and unwrapping.

        Args:
            phases: List of phase values
            timestamps: List of timestamps
            smooth: Whether to smooth phases
            unwrap: Whether to unwrap phases

        Returns:
            List of tracked phase values
        """
        tracked = phases

        if smooth:
            tracked = self.smooth_phases(tracked, timestamps)

        if unwrap:
            tracked = self.unwrap_phase(tracked, timestamps)

        return tracked

