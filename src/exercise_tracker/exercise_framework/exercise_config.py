"""Configuration system for exercise-specific parameters."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ExerciseConfig:
    """Configuration for an exercise."""

    def __init__(
        self,
        name: str,
        min_rep_duration: float = 0.5,
        max_rep_duration: float = 10.0,
        min_phase_coverage: float = 0.7,
        phase_wrap_threshold: float = 0.8,
        smoothing_window: int = 12,
        smoothing_duration: Optional[float] = None,
        smoothing_mode: str = "frames",
        tolerance_multiplier: float = 2.0,
        num_phase_bins: int = 100,
        embed_dim: int = 32,
        phase_window_size: int = 10,
        phase_window_duration: Optional[float] = None,
        phase_window_mode: str = "frames",
        phase_buffer_size: int = 5,
        phase_model_type: str = "frame_based",
        use_3d: bool = False,
        **kwargs,
    ):
        """
        Initialize exercise configuration.

        Args:
            name: Exercise name
            min_rep_duration: Minimum rep duration in seconds
            max_rep_duration: Maximum rep duration in seconds
            min_phase_coverage: Minimum phase coverage for valid rep
            phase_wrap_threshold: Minimum phase increase for wrap detection
            smoothing_window: Phase smoothing window size (frames, used if smoothing_mode='frames')
            smoothing_duration: Phase smoothing duration in seconds (used if smoothing_mode='time')
            smoothing_mode: Smoothing mode - 'frames' or 'time' (default: 'frames')
            tolerance_multiplier: Multiplier for tolerance band
            num_phase_bins: Number of phase bins for manifold
            embed_dim: Embedding dimension
            phase_window_size: Number of embeddings in sliding window (frames, used if phase_window_mode='frames')
            phase_window_duration: Duration of sliding window in seconds (used if phase_window_mode='time')
            phase_window_mode: Window mode - 'frames' or 'time' (default: 'frames')
            phase_buffer_size: Number of previous phases to include as context (default: 1)
            phase_model_type: Type of phase model - 'frame_based' or 'time_based' (default: 'frame_based')
            use_3d: If True, use 3D pose coordinates (x, y, z); if False, use 2D (x, y)
            **kwargs: Additional custom parameters
        """
        self.name = name
        self.min_rep_duration = min_rep_duration
        self.max_rep_duration = max_rep_duration
        self.min_phase_coverage = min_phase_coverage
        self.phase_wrap_threshold = phase_wrap_threshold
        self.smoothing_window = smoothing_window
        self.smoothing_duration = smoothing_duration
        self.smoothing_mode = smoothing_mode
        self.tolerance_multiplier = tolerance_multiplier
        self.num_phase_bins = num_phase_bins
        self.embed_dim = embed_dim
        self.phase_window_size = phase_window_size
        self.phase_window_duration = phase_window_duration
        self.phase_window_mode = phase_window_mode
        self.phase_buffer_size = phase_buffer_size
        self.phase_model_type = phase_model_type
        self.use_3d = use_3d
        self.custom_params = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "min_rep_duration": self.min_rep_duration,
            "max_rep_duration": self.max_rep_duration,
            "min_phase_coverage": self.min_phase_coverage,
            "phase_wrap_threshold": self.phase_wrap_threshold,
            "smoothing_window": self.smoothing_window,
            "smoothing_duration": self.smoothing_duration,
            "smoothing_mode": self.smoothing_mode,
            "tolerance_multiplier": self.tolerance_multiplier,
            "num_phase_bins": self.num_phase_bins,
            "embed_dim": self.embed_dim,
            "phase_window_size": self.phase_window_size,
            "phase_window_duration": self.phase_window_duration,
            "phase_window_mode": self.phase_window_mode,
            "phase_buffer_size": self.phase_buffer_size,
            "phase_model_type": self.phase_model_type,
            "use_3d": self.use_3d,
            **self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExerciseConfig":
        """Create config from dictionary."""
        name = config_dict.pop("name")
        return cls(name=name, **config_dict)

    def save(self, filepath: str) -> None:
        """Save config to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> "ExerciseConfig":
        """Load config from YAML file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

