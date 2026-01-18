"""Factory for creating phase models based on configuration."""

from typing import Optional
from .base_phase_model import BasePhaseModel
from .phase_model import FrameBasedPhaseModel
from .time_based_phase_model import TimeBasedPhaseModel


def create_phase_model(
    model_type: str = "frame_based",
    phase_window_size: int = 10,
    phase_window_duration: Optional[float] = None,
    phase_window_mode: str = "frames",
    phase_buffer_size: int = 5,
    **kwargs,
) -> BasePhaseModel:
    """
    Create a phase model based on configuration.

    Args:
        model_type: Type of model - 'frame_based' or 'time_based'
        phase_window_size: Window size in frames (for frame_based)
        phase_window_duration: Window duration in seconds (for time_based)
        phase_window_mode: Window mode - 'frames' or 'time' (determines model type if model_type not specified)
        phase_buffer_size: Number of previous phases to include as context
        **kwargs: Additional model parameters

    Returns:
        Phase model instance
    """
    # Determine model type from mode if not explicitly set
    if model_type == "frame_based" or (model_type is None and phase_window_mode == "frames"):
        return FrameBasedPhaseModel(
            window_size=phase_window_size,
            phase_buffer_size=phase_buffer_size,
            **kwargs,
        )
    elif model_type == "time_based" or phase_window_mode == "time":
        if phase_window_duration is None:
            # Default: estimate from window_size assuming 30fps
            phase_window_duration = phase_window_size / 30.0
        return TimeBasedPhaseModel(
            window_duration=phase_window_duration,
            phase_buffer_size=phase_buffer_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
