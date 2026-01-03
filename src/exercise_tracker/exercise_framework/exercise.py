"""Base class for exercise definitions."""

from typing import Dict, Any, Optional, List
from pathlib import Path

from .exercise_config import ExerciseConfig
from ..data import DataManager
from ..pose_processing import PoseEmbedder
from ..reference_learning import ManifoldBuilder, TrajectoryBuilder
from ..phase_estimation import PhaseModel, PhaseTracker
from ..rep_counting import RepCounter, RepValidator
from ..form_assessment import FormScorer, TrajectoryComparator


class Exercise:
    """Base class for exercise definitions."""

    def __init__(
        self,
        config: ExerciseConfig,
        data_root: str = "data",
    ):
        """
        Initialize exercise.

        Args:
            config: Exercise configuration
            data_root: Root directory for data
        """
        self.config = config
        self.data_root = Path(data_root)
        self.exercise_dir = self.data_root / "processed" / config.name
        self.exercise_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_manager = DataManager(data_root=data_root)
        self.embedder: Optional[PoseEmbedder] = None
        self.manifold_builder: Optional[ManifoldBuilder] = None
        self.phase_model: Optional[PhaseModel] = None
        self.phase_tracker = PhaseTracker(
            smoothing_window=config.smoothing_window,
            min_rep_duration=config.min_rep_duration,
            min_phase_coverage=config.min_phase_coverage,
        )
        self.rep_counter = RepCounter(
            min_rep_duration=config.min_rep_duration,
            min_phase_coverage=config.min_phase_coverage,
            phase_wrap_threshold=config.phase_wrap_threshold,
        )
        self.rep_validator = RepValidator(
            min_duration=config.min_rep_duration,
            max_duration=config.max_rep_duration,
            min_phase_coverage=config.min_phase_coverage,
        )

        self.form_scorer: Optional[FormScorer] = None
        self.trajectory_comparator: Optional[TrajectoryComparator] = None

    def get_embedder_path(self) -> Path:
        """Get path to saved embedder."""
        return self.exercise_dir / "embedder.pkl"

    def get_manifold_path(self) -> Path:
        """Get path to saved manifold."""
        return self.exercise_dir / "manifold.pkl"

    def get_phase_model_path(self) -> Path:
        """Get path to saved phase model."""
        return self.exercise_dir / "phase_model.pkl"

    def get_config_path(self) -> Path:
        """Get path to saved config."""
        return self.exercise_dir / "config.yaml"

    def save(self) -> None:
        """Save exercise components to disk."""
        # Save config
        self.config.save(str(self.get_config_path()))

        # Save embedder if available
        if self.embedder and self.embedder.is_fitted:
            self.embedder.save(str(self.get_embedder_path()))

        # Save phase model if available
        if self.phase_model and self.phase_model.is_fitted:
            self.phase_model.save(str(self.get_phase_model_path()))

        # Save manifold if available
        if self.manifold_builder and self.manifold_builder.manifold_mean is not None:
            import joblib

            manifold_data = {
                "manifold_mean": self.manifold_builder.manifold_mean,
                "manifold_std": self.manifold_builder.manifold_std,
                "phase_bins": self.manifold_builder.phase_bins,
                "embed_dim": self.manifold_builder.embed_dim,
            }
            joblib.dump(manifold_data, str(self.get_manifold_path()))

    def load(self) -> None:
        """Load exercise components from disk."""
        # Load config
        if self.get_config_path().exists():
            self.config = ExerciseConfig.load(str(self.get_config_path()))

        # Load embedder if available
        if self.get_embedder_path().exists():
            self.embedder = PoseEmbedder.load(str(self.get_embedder_path()))
            
            # Validate embedder dimension matches config use_3d setting
            if self.embedder and self.embedder.is_fitted:
                expected_dim = 99 if self.config.use_3d else 66  # 33 keypoints * 3 or * 2
                actual_dim = self.embedder.pca.n_features_in_
                if actual_dim != expected_dim:
                    import warnings
                    warnings.warn(
                        f"Dimension mismatch detected: Embedder was trained with {actual_dim} features "
                        f"({'3D' if actual_dim == 99 else '2D'} data), but config has use_3d={self.config.use_3d} "
                        f"(expects {expected_dim} features). "
                        f"This may cause errors during processing. Consider re-ingesting and re-training with "
                        f"use_3d={'True' if actual_dim == 99 else 'False'} to match the embedder."
                    )

        # Load phase model if available
        if self.get_phase_model_path().exists():
            self.phase_model = PhaseModel.load(str(self.get_phase_model_path()))

        # Load manifold if available
        if self.get_manifold_path().exists():
            import joblib

            manifold_data = joblib.load(str(self.get_manifold_path()))
            self.manifold_builder = ManifoldBuilder(
                num_phase_bins=len(manifold_data["phase_bins"])
            )
            self.manifold_builder.manifold_mean = manifold_data["manifold_mean"]
            self.manifold_builder.manifold_std = manifold_data["manifold_std"]
            self.manifold_builder.phase_bins = manifold_data["phase_bins"]
            self.manifold_builder.embed_dim = manifold_data["embed_dim"]

            # Initialize form assessment components
            from ..reference_learning import ToleranceBand

            tolerance_band = ToleranceBand(
                self.manifold_builder,
                tolerance_multiplier=self.config.tolerance_multiplier,
            )
            self.form_scorer = FormScorer(self.manifold_builder)
            self.trajectory_comparator = TrajectoryComparator(self.manifold_builder)

