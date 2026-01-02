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
        smoothing_window: int = 5,
        tolerance_multiplier: float = 2.0,
        num_phase_bins: int = 100,
        embed_dim: int = 32,
        phase_window_size: int = 10,
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
            smoothing_window: Phase smoothing window size
            tolerance_multiplier: Multiplier for tolerance band
            num_phase_bins: Number of phase bins for manifold
            embed_dim: Embedding dimension
            phase_window_size: Number of embeddings in sliding window for phase prediction
            **kwargs: Additional custom parameters
        """
        self.name = name
        self.min_rep_duration = min_rep_duration
        self.max_rep_duration = max_rep_duration
        self.min_phase_coverage = min_phase_coverage
        self.phase_wrap_threshold = phase_wrap_threshold
        self.smoothing_window = smoothing_window
        self.tolerance_multiplier = tolerance_multiplier
        self.num_phase_bins = num_phase_bins
        self.embed_dim = embed_dim
        self.phase_window_size = phase_window_size
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
            "tolerance_multiplier": self.tolerance_multiplier,
            "num_phase_bins": self.num_phase_bins,
            "embed_dim": self.embed_dim,
            "phase_window_size": self.phase_window_size,
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

