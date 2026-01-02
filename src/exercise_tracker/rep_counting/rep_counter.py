"""Rep counter that detects phase wraps and counts valid reps."""

import numpy as np
from typing import List, Optional, Tuple, Dict


class RepCounter:
    """Count exercise reps by detecting phase wraps."""

    def __init__(
        self,
        min_rep_duration: float = 0.5,
        min_phase_coverage: float = 0.7,
        phase_wrap_threshold: float = 0.8,
    ):
        """
        Initialize rep counter.

        Args:
            min_rep_duration: Minimum duration in seconds for a valid rep
            min_phase_coverage: Minimum phase coverage (0-1) for a valid rep
            phase_wrap_threshold: Minimum phase increase to count as a wrap
        """
        self.min_rep_duration = min_rep_duration
        self.min_phase_coverage = min_phase_coverage
        self.phase_wrap_threshold = phase_wrap_threshold

    def detect_phase_wraps(
        self,
        unwrapped_phases: List[Optional[float]],
        timestamps: List[float],
    ) -> List[Tuple[int, int, float]]:
        """
        Detect phase wraps (complete rep cycles).

        Args:
            unwrapped_phases: List of unwrapped phase values
            timestamps: List of timestamps

        Returns:
            List of (start_idx, end_idx, duration) tuples for each detected wrap
        """
        if len(unwrapped_phases) == 0:
            return []

        wraps = []
        last_phase = None
        wrap_start_idx = None
        wrap_start_phase = None

        for i, phase in enumerate(unwrapped_phases):
            if phase is None:
                continue

            if last_phase is not None:
                phase_increase = phase - last_phase

                # Detect wrap start: phase increases significantly
                if wrap_start_idx is None and phase_increase > 0.1:
                    wrap_start_idx = i - 1
                    wrap_start_phase = last_phase

                # Detect wrap end: phase has increased by at least threshold
                if (
                    wrap_start_idx is not None
                    and wrap_start_phase is not None
                    and phase - wrap_start_phase >= self.phase_wrap_threshold
                ):
                    duration = timestamps[i] - timestamps[wrap_start_idx]
                    wraps.append((wrap_start_idx, i, duration))
                    wrap_start_idx = None
                    wrap_start_phase = None

            last_phase = phase

        return wraps

    def count_reps(
        self,
        unwrapped_phases: List[Optional[float]],
        timestamps: List[float],
    ) -> Tuple[int, List[Dict[str, any]]]:
        """
        Count valid reps from phase progression.

        Args:
            unwrapped_phases: List of unwrapped phase values
            timestamps: List of timestamps

        Returns:
            Tuple of (rep_count, rep_details) where rep_details is a list of
            dictionaries with rep information
        """
        wraps = self.detect_phase_wraps(unwrapped_phases, timestamps)

        valid_reps = []
        for start_idx, end_idx, duration in wraps:
            # Validate rep
            if duration < self.min_rep_duration:
                continue

            # Check phase coverage
            rep_phases = [
                p % 1.0
                for p in unwrapped_phases[start_idx : end_idx + 1]
                if p is not None
            ]

            if len(rep_phases) == 0:
                continue

            phase_range = max(rep_phases) - min(rep_phases)
            if phase_range < self.min_phase_coverage:
                continue

            valid_reps.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": timestamps[start_idx],
                    "end_time": timestamps[end_idx],
                    "duration": duration,
                    "phase_range": phase_range,
                }
            )

        return len(valid_reps), valid_reps

    def get_rep_boundaries(
        self,
        unwrapped_phases: List[Optional[float]],
        timestamps: List[float],
    ) -> List[Tuple[int, int]]:
        """
        Get rep boundaries as (start_idx, end_idx) tuples.

        Args:
            unwrapped_phases: List of unwrapped phase values
            timestamps: List of timestamps

        Returns:
            List of (start_idx, end_idx) tuples
        """
        _, rep_details = self.count_reps(unwrapped_phases, timestamps)
        return [(rep["start_idx"], rep["end_idx"]) for rep in rep_details]

