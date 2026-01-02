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
        """
        # Get or create exercise
        ex = self.registry.get(exercise)
        if ex is None:
            config = ExerciseConfig(name=exercise)
            ex = Exercise(config, data_root=self.data_root)
            self.registry.register(ex)

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
    ):
        """
        Train phase estimation model.

        Args:
            exercise: Exercise name
            video_paths: Optional list of video paths (uses reference videos if None)
            debug: If True, generate similarity_to_start plots for debugging
            augment_flip: If True, augment data with horizontally flipped poses (default: True)
        """
        # Load exercise
        ex = self.registry.get(exercise)
        if ex is None:
            try:
                ex = self.registry.load_exercise(exercise, data_root=self.data_root)
            except Exception as e:
                print(f"Error loading exercise: {e}")
                return

        print(f"Training phase model for exercise '{exercise}'...")

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

