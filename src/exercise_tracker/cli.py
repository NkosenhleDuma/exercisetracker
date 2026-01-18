"""Command-line interface using Fire."""

import fire
import json
from pathlib import Path
from typing import Optional, List

from .exercise_framework import ExerciseRegistry, ExerciseConfig, Exercise
from .pipeline import ExercisePipeline
from .data import DataManager


class ExerciseTrackerCLI:
    """Command-line interface for exercise tracker."""

    def __init__(self, data_root: str = "data"):
        """
        Initialize CLI.

        Args:
            data_root: Root directory for data
        """
        self.data_root = data_root
        self.registry = ExerciseRegistry()
        self.data_manager = DataManager(data_root=data_root)

    def ingest(
        self,
        exercise: str,
        video_dir: Optional[str] = None,
        video_paths: Optional[List[str]] = None,
        force_reprocess: bool = False,
        auto_rep: bool = False,
        number_of_reps: Optional[int] = None,
        augment_flip: bool = True,
        use_3d: bool = False,
        phase_model_type: Optional[str] = None,
        phase_window_duration: Optional[float] = None,
        smoothing_mode: Optional[str] = None,
        smoothing_duration: Optional[float] = None,
    ):
        """
        Process reference videos and build manifolds.

        Args:
            exercise: Exercise name
            video_dir: Directory containing reference videos (alternative to video_paths)
            video_paths: List of video file paths (alternative to video_dir)
            force_reprocess: If True, reprocess even if cached
            auto_rep: If True, automatically detect reps (default: False, treats each video as one rep)
            number_of_reps: If specified, divide each video into this many equal segments
            augment_flip: If True, augment data with horizontally flipped poses (default: True)
            use_3d: If True, use 3D pose coordinates (x, y, z); if False, use 2D (x, y) (default: False)
            phase_model_type: Phase model type - 'frame_based' or 'time_based' (default: 'frame_based')
            phase_window_duration: Window duration in seconds for time-based model (default: 0.5)
            smoothing_mode: Smoothing mode - 'frames' or 'time' (default: 'frames')
            smoothing_duration: Smoothing duration in seconds for time-based smoothing (default: 0.3)
        """
        # Get or create exercise
        ex = self.registry.get(exercise)
        if ex is None:
            # Create config with provided parameters
            config_kwargs = {"name": exercise, "use_3d": use_3d}
            
            # Set phase model type and related parameters
            if phase_model_type is not None:
                config_kwargs["phase_model_type"] = phase_model_type
                if phase_model_type == "time_based":
                    config_kwargs["phase_window_mode"] = "time"
                    if phase_window_duration is not None:
                        config_kwargs["phase_window_duration"] = phase_window_duration
                    else:
                        config_kwargs["phase_window_duration"] = 0.5  # Default
                else:
                    config_kwargs["phase_window_mode"] = "frames"
            
            # Set smoothing mode
            if smoothing_mode is not None:
                config_kwargs["smoothing_mode"] = smoothing_mode
                if smoothing_mode == "time" and smoothing_duration is not None:
                    config_kwargs["smoothing_duration"] = smoothing_duration
                elif smoothing_mode == "time" and smoothing_duration is None:
                    config_kwargs["smoothing_duration"] = 0.3  # Default
            
            config = ExerciseConfig(**config_kwargs)
            ex = Exercise(config, data_root=self.data_root)
            self.registry.register(ex)
        else:
            # Update config if provided
            updated = False
            if use_3d != ex.config.use_3d:
                ex.config.use_3d = use_3d
                updated = True
                print(f"Updated use_3d setting to {use_3d} for exercise '{exercise}'")
            
            if phase_model_type is not None and phase_model_type != ex.config.phase_model_type:
                ex.config.phase_model_type = phase_model_type
                if phase_model_type == "time_based":
                    ex.config.phase_window_mode = "time"
                    if phase_window_duration is not None:
                        ex.config.phase_window_duration = phase_window_duration
                    elif ex.config.phase_window_duration is None:
                        ex.config.phase_window_duration = 0.5  # Default
                else:
                    ex.config.phase_window_mode = "frames"
                updated = True
                print(f"Updated phase_model_type to {phase_model_type} for exercise '{exercise}'")
            
            if smoothing_mode is not None and smoothing_mode != ex.config.smoothing_mode:
                ex.config.smoothing_mode = smoothing_mode
                if smoothing_mode == "time":
                    if smoothing_duration is not None:
                        ex.config.smoothing_duration = smoothing_duration
                    elif ex.config.smoothing_duration is None:
                        ex.config.smoothing_duration = 0.3  # Default
                updated = True
                print(f"Updated smoothing_mode to {smoothing_mode} for exercise '{exercise}'")
            
            if updated:
                ex.save()

        # Get video paths
        if video_paths is None:
            if video_dir is None:
                video_dir = str(self.data_manager.get_exercise_dir(exercise, raw=True))

            video_dir = Path(video_dir)
            if not video_dir.exists():
                print(f"Error: Video directory not found: {video_dir}")
                return

            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            video_paths = [
                str(f)
                for f in video_dir.iterdir()
                if f.is_file() and f.suffix.lower() in video_extensions
            ]

        if len(video_paths) == 0:
            print(f"Error: No videos found in {video_dir}")
            return

        print(f"Ingesting {len(video_paths)} reference videos for exercise '{exercise}'...")
        if ex.config.phase_model_type:
            print(f"  Phase model type: {ex.config.phase_model_type}")
            if ex.config.phase_window_mode == "time":
                print(f"  Window duration: {ex.config.phase_window_duration} seconds")

        # Run ingestion pipeline
        pipeline = ExercisePipeline(ex)
        try:
            result = pipeline.ingest_reference_videos(
                video_paths,
                force_reprocess=force_reprocess,
                auto_rep=auto_rep,
                number_of_reps=number_of_reps,
                augment_flip=augment_flip,
            )
            print(f"Successfully ingested {result['num_videos']} videos")
            print(f"Built manifold from {result['num_samples']} samples")
            if "rep_counts_per_video" in result:
                print(f"Detected {result['total_reps']} total rep(s) across all videos")
                for i, count in enumerate(result["rep_counts_per_video"]):
                    print(f"  Video {i + 1}: {count} rep(s)")
        except Exception as e:
            print(f"Error during ingestion: {e}")
            raise

    def train(
        self,
        exercise: str,
        video_paths: Optional[List[str]] = None,
        debug: bool = False,
        augment_flip: bool = True,
        use_3d: bool = False,
        phase_model_type: Optional[str] = None,
        phase_window_duration: Optional[float] = None,
        smoothing_mode: Optional[str] = None,
        smoothing_duration: Optional[float] = None,
    ):
        """
        Train phase estimation model.

        Args:
            exercise: Exercise name
            video_paths: Optional list of video paths (uses reference videos if None)
            debug: If True, generate similarity_to_start plots for debugging
            augment_flip: If True, augment data with horizontally flipped poses (default: True)
            use_3d: If True, use 3D pose coordinates (x, y, z); if False, use 2D (x, y) (default: False)
                   Note: Must match the use_3d setting used during ingestion
            phase_model_type: Phase model type - 'frame_based' or 'time_based' (overrides config if provided)
            phase_window_duration: Window duration in seconds for time-based model (default: 0.5)
            smoothing_mode: Smoothing mode - 'frames' or 'time' (overrides config if provided)
            smoothing_duration: Smoothing duration in seconds for time-based smoothing (default: 0.3)
        """
        # Load exercise and check use_3d consistency
        ex = self.registry.get(exercise)
        if ex is None:
            try:
                ex = self.registry.load_exercise(exercise, data_root=self.data_root)
            except Exception as e:
                print(f"Error loading exercise: {e}")
                return
        
        # Update config if phase model parameters provided
        updated = False
        if phase_model_type is not None and phase_model_type != ex.config.phase_model_type:
            ex.config.phase_model_type = phase_model_type
            if phase_model_type == "time_based":
                ex.config.phase_window_mode = "time"
                if phase_window_duration is not None:
                    ex.config.phase_window_duration = phase_window_duration
                elif ex.config.phase_window_duration is None:
                    ex.config.phase_window_duration = 0.5  # Default
            else:
                ex.config.phase_window_mode = "frames"
            updated = True
            print(f"Updated phase_model_type to {phase_model_type} for training")
        
        if smoothing_mode is not None and smoothing_mode != ex.config.smoothing_mode:
            ex.config.smoothing_mode = smoothing_mode
            if smoothing_mode == "time":
                if smoothing_duration is not None:
                    ex.config.smoothing_duration = smoothing_duration
                elif ex.config.smoothing_duration is None:
                    ex.config.smoothing_duration = 0.3  # Default
            updated = True
            print(f"Updated smoothing_mode to {smoothing_mode} for training")
        
        if updated:
            ex.save()
        
        # Warn if use_3d doesn't match config
        if use_3d != ex.config.use_3d:
            print(f"Warning: use_3d={use_3d} doesn't match exercise config (use_3d={ex.config.use_3d})")
            print(f"Using exercise config setting: use_3d={ex.config.use_3d}")

        print(f"Training phase model for exercise '{exercise}'...")
        print(f"  Model type: {ex.config.phase_model_type}")
        if ex.config.phase_window_mode == "time":
            print(f"  Window duration: {ex.config.phase_window_duration} seconds")
        else:
            print(f"  Window size: {ex.config.phase_window_size} frames")

        # Run training pipeline
        pipeline = ExercisePipeline(ex)
        try:
            result = pipeline.train_phase_model(
                video_paths=video_paths, debug=debug, augment_flip=augment_flip
            )
            print(f"Successfully trained model on {result['num_samples']} samples")
            if debug and "debug_plots" in result:
                print(f"Debug plots saved to: {result['debug_plots']}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def analyze(
        self,
        exercise: str,
        video: str,
        output: Optional[str] = None,
        force_reprocess: bool = False,
    ):
        """
        Analyze a video for rep count and form.

        Args:
            exercise: Exercise name
            video: Path to video file
            output: Optional path to save results JSON
            force_reprocess: If True, reprocess even if cached
        """
        # Load exercise
        ex = self.registry.get(exercise)
        if ex is None:
            try:
                ex = self.registry.load_exercise(exercise, data_root=self.data_root)
            except Exception as e:
                print(f"Error loading exercise: {e}")
                return

        if not Path(video).exists():
            print(f"Error: Video file not found: {video}")
            return

        print(f"Analyzing video '{video}' for exercise '{exercise}'...")

        # Run analysis pipeline
        pipeline = ExercisePipeline(ex)
        try:
            result = pipeline.analyze_video(video, force_reprocess=force_reprocess)

            # Print results
            print(f"\nResults:")
            print(f"  Rep count: {result['rep_count']}")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Frames: {result['num_frames']}")

            if result.get("rep_scores"):
                avg_score = sum(
                    s.get("mean_score", 0.0) for s in result["rep_scores"]
                ) / len(result["rep_scores"])
                print(f"  Average form score: {avg_score:.3f}")

            if result.get("overall_comparison"):
                similarity = result["overall_comparison"].get(
                    "trajectory_similarity", 0.0
                )
                print(f"  Trajectory similarity: {similarity:.3f}")

            # Save results if output specified
            if output:
                with open(output, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nResults saved to: {output}")

        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

    def list_exercises(self):
        """List all registered exercises."""
        exercises = self.registry.list_exercises()

        # Also check data directory for exercises
        data_exercises = self.data_manager.list_exercises()
        all_exercises = sorted(set(exercises + data_exercises))

        if len(all_exercises) == 0:
            print("No exercises found.")
        else:
            print(f"Found {len(all_exercises)} exercise(s):")
            for ex_name in all_exercises:
                print(f"  - {ex_name}")


def main():
    """Main entry point for CLI."""
    fire.Fire(ExerciseTrackerCLI)


if __name__ == "__main__":
    main()

