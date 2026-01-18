"""Phase estimation module for mapping poses to exercise phase."""

from .phase_model import FrameBasedPhaseModel
from .time_based_phase_model import TimeBasedPhaseModel
from .phase_tracker import PhaseTracker
from .model_factory import create_phase_model
from .base_phase_model import BasePhaseModel

# Backward compatibility alias
PhaseModel = FrameBasedPhaseModel

__all__ = [
    "PhaseModel",
    "FrameBasedPhaseModel",
    "TimeBasedPhaseModel",
    "PhaseTracker",
    "create_phase_model",
    "BasePhaseModel",
]

