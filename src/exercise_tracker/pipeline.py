"""End-to-end pipeline orchestrator."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path

from .data import VideoProcessor, DataManager
from .pose_extraction import MediaPipeExtractor
from .pose_processing import PoseNormalizer, PoseEmbedder
from .pose_processing.pose_utils import flip_keypoints_horizontally
from .reference_learning import TrajectoryBuilder, ManifoldBuilder
from .phase_estimation import PhaseModel, PhaseTracker
from .rep_counting import RepCounter, RepValidator, AutoRepDetector
from .form_assessment import FormScorer, TrajectoryComparator
from .exercise_framework import Exercise, ExerciseConfig


class ExercisePipeline:
    """End-to-end pipeline for exercise analysis."""

    def __init__(self, exercise: Exercise):
        """
        Initialize pipeline with an exercise.

        Args:
            exercise: Exercise instance
        """
        self.exercise = exercise

    def ingest_reference_videos(
        self,
        video_paths: List[str],
        rep_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        force_reprocess: bool = False,
        auto_rep: bool = False,
        number_of_reps: Optional[int] = None,
        augment_flip: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest reference videos and build canonical manifold.

        Args:
            video_paths: List of paths to reference videos
            rep_boundaries: Optional list of rep boundaries per video
            force_reprocess: If True, reprocess even if cached
            auto_rep: If True, automatically detect reps (default: False)
            number_of_reps: If specified, divide each video into this many equal segments
            augment_flip: If True, augment data with horizontally flipped poses

        Returns:
            Dictionary with ingestion results
        """
        # Create extractor and normalizer with use_3d from config
        use_3d = self.exercise.config.use_3d
        pose_extractor = MediaPipeExtractor(use_3d=use_3d)
        normalizer = PoseNormalizer(use_3d=use_3d)
        video_processor = VideoProcessor(
            pose_extractor=pose_extractor,
            normalizer=normalizer
        )
        trajectory_builder = TrajectoryBuilder(
            num_phase_points=self.exercise.config.num_phase_bins
        )

        all_embeddings = []
        all_phases = []
        all_labeled_data = []
        video_rep_counts = []  # Track rep counts per video for validation

        # Process each video
        for video_idx, video_path in enumerate(video_paths):
            print(f"Processing reference video {video_idx + 1}/{len(video_paths)}: {video_path}")

            # Process video
            cache_dir = str(self.exercise.exercise_dir)
            video_data = video_processor.process_video(
                video_path, cache_dir=cache_dir, force_reprocess=force_reprocess
            )

            # Fit embedder if not already fitted
            if self.exercise.embedder is None or not self.exercise.embedder.is_fitted:
                # Collect all pose vectors for fitting
                pose_vectors = video_data["pose_vectors"]
                if len(pose_vectors) > 0:
                    if self.exercise.embedder is None:
                        self.exercise.embedder = PoseEmbedder(
                            n_components=self.exercise.config.embed_dim
                        )
                    # Fit on this video's data
                    valid_vectors = [v for v in pose_vectors if v is not None]
                    if len(valid_vectors) > 0:
                        self.exercise.embedder.fit(valid_vectors)

            # Generate embeddings
            if self.exercise.embedder and self.exercise.embedder.is_fitted:
                embeddings = self.exercise.embedder.embed_sequence(
                    video_data["pose_vectors"]
                )
            else:
                raise ValueError("Embedder must be fitted before generating embeddings")

            # Create augmented (flipped) data if requested
            flipped_embeddings = None
            if augment_flip:
                # Check if keypoints are available (may not be in cached data)
                if "keypoints" not in video_data or video_data["keypoints"] is None:
                    print(f"  Warning: Keypoints not available for augmentation, skipping flip augmentation for this video")
                else:
                    # Flip keypoints horizontally
                    flipped_keypoints = [
                        flip_keypoints_horizontally(kp) if kp is not None else None
                        for kp in video_data["keypoints"]
                    ]
                    
                    # Normalize flipped keypoints (use same use_3d setting)
                    flip_normalizer = PoseNormalizer(use_3d=self.exercise.config.use_3d)
                    flipped_normalized = flip_normalizer.normalize_sequence(flipped_keypoints)
                    
                    # Get pose vectors from flipped normalized keypoints
                    flipped_pose_vectors = [
                        flip_normalizer.get_pose_vector(kp) for kp in flipped_normalized
                    ]
                    
                    # Embed flipped poses
                    if self.exercise.embedder and self.exercise.embedder.is_fitted:
                        flipped_embeddings = self.exercise.embedder.embed_sequence(
                            flipped_pose_vectors
                        )
                        print(f"  Created flipped augmentation ({len([e for e in flipped_embeddings if e is not None])} valid embeddings)")

            # Get rep boundaries for this video
            video_rep_boundaries = None
            
            if rep_boundaries and video_idx < len(rep_boundaries):
                # Use provided rep boundaries
                video_rep_boundaries = rep_boundaries[video_idx]
            elif number_of_reps is not None:
                # Divide video into specified number of equal segments
                num_frames = len(embeddings)
                if num_frames > 0:
                    frames_per_rep = num_frames // number_of_reps
                    video_rep_boundaries = []
                    for rep_idx in range(number_of_reps):
                        start_idx = rep_idx * frames_per_rep
                        end_idx = (rep_idx + 1) * frames_per_rep - 1 if rep_idx < number_of_reps - 1 else num_frames - 1
                        video_rep_boundaries.append((start_idx, end_idx))
                    print(f"  Divided video into {len(video_rep_boundaries)} rep(s) (requested: {number_of_reps})")
            elif auto_rep:
                # Automatically detect reps
                print(f"  Automatically detecting reps in video {video_idx + 1}...")
                auto_detector = AutoRepDetector(
                    min_rep_duration=self.exercise.config.min_rep_duration,
                    max_rep_duration=self.exercise.config.max_rep_duration,
                )
                video_rep_boundaries = auto_detector.detect_rep_boundaries(
                    embeddings, video_data["timestamps"]
                )
                print(f"  Detected {len(video_rep_boundaries)} rep(s)")
            # else: treat as one rep (default behavior)

            # Segment and label reps
            if video_rep_boundaries and len(video_rep_boundaries) > 0:
                labeled_data = trajectory_builder.assign_phase_labels(
                    embeddings, video_rep_boundaries
                )
                all_labeled_data.extend(labeled_data)
                
                # Add flipped augmentation with same phase labels
                if flipped_embeddings is not None:
                    flipped_labeled_data = trajectory_builder.assign_phase_labels(
                        flipped_embeddings, video_rep_boundaries
                    )
                    all_labeled_data.extend(flipped_labeled_data)
                
                video_rep_counts.append(len(video_rep_boundaries))
            else:
                # Default: treat entire video as one rep, assign phases linearly
                num_frames = len(embeddings)
                for i, emb in enumerate(embeddings):
                    if emb is not None:
                        phase = i / num_frames if num_frames > 0 else 0.0
                        all_labeled_data.append((emb, phase))
                
                # Add flipped augmentation
                if flipped_embeddings is not None:
                    for i, emb in enumerate(flipped_embeddings):
                        if emb is not None:
                            phase = i / num_frames if num_frames > 0 else 0.0
                            all_labeled_data.append((emb, phase))
                
                video_rep_counts.append(1)  # Count as 1 rep

            all_embeddings.extend(embeddings)
            if flipped_embeddings is not None:
                all_embeddings.extend([e for e in flipped_embeddings if e is not None])

        # Build manifold
        if len(all_labeled_data) == 0:
            raise ValueError("No labeled data collected from reference videos")

        print(f"Building manifold from {len(all_labeled_data)} labeled samples...")
        self.exercise.manifold_builder = ManifoldBuilder(
            num_phase_bins=self.exercise.config.num_phase_bins
        )
        self.exercise.manifold_builder.build_manifold(all_labeled_data)

        # Initialize form assessment components
        from .reference_learning import ToleranceBand

        tolerance_band = ToleranceBand(
            self.exercise.manifold_builder,
            tolerance_multiplier=self.exercise.config.tolerance_multiplier,
        )
        self.exercise.form_scorer = FormScorer(self.exercise.manifold_builder)
        self.exercise.trajectory_comparator = TrajectoryComparator(
            self.exercise.manifold_builder
        )

        # Save exercise
        self.exercise.save()

        return {
            "num_videos": len(video_paths),
            "num_samples": len(all_labeled_data),
            "manifold_built": True,
            "rep_counts_per_video": video_rep_counts,
            "total_reps": sum(video_rep_counts),
        }

    def train_phase_model(
        self,
        video_paths: Optional[List[str]] = None,
        rep_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        debug: bool = False,
        augment_flip: bool = True,
    ) -> Dict[str, Any]:
        """
        Train phase estimation model.

        Args:
            video_paths: Optional list of video paths (uses reference videos if None)
            rep_boundaries: Optional rep boundaries per video
            debug: If True, generate similarity_to_start plots for debugging
            augment_flip: If True, augment data with horizontally flipped poses

        Returns:
            Dictionary with training results
        """
        if self.exercise.embedder is None or not self.exercise.embedder.is_fitted:
            raise ValueError("Embedder must be fitted before training phase model")

        if self.exercise.manifold_builder is None:
            raise ValueError("Manifold must be built before training phase model")

        video_processor = VideoProcessor(embedder=self.exercise.embedder)
        trajectory_builder = TrajectoryBuilder()

        all_embeddings = []
        all_phases = []

        # Use provided videos or load from data manager
        if video_paths is None:
            video_paths = self.exercise.data_manager.list_videos(
                self.exercise.config.name
            )

        # Create video processor with use_3d from config
        use_3d = self.exercise.config.use_3d
        pose_extractor = MediaPipeExtractor(use_3d=use_3d)
        normalizer = PoseNormalizer(use_3d=use_3d)
        video_processor = VideoProcessor(
            pose_extractor=pose_extractor,
            normalizer=normalizer
        )

        # Setup debug directory if needed
        debug_dir = None
        if debug:
            debug_dir = self.exercise.exercise_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")  # Use non-interactive backend
            import matplotlib.pyplot as plt

        # Process videos
        for video_idx, video_path in enumerate(video_paths):
            print(f"Processing training video {video_idx + 1}/{len(video_paths)}: {video_path}")

            cache_dir = str(self.exercise.exercise_dir)
            video_data = video_processor.process_video(
                video_path, cache_dir=cache_dir, force_reprocess=False
            )

            embeddings = video_data["embeddings"]
            if embeddings is None:
                embeddings = self.exercise.embedder.embed_sequence(
                    video_data["pose_vectors"]
                )

            # Create augmented (flipped) data if requested
            flipped_embeddings = None
            if augment_flip:
                # Check if keypoints are available (may not be in cached data)
                if "keypoints" not in video_data or video_data["keypoints"] is None:
                    print(f"  Warning: Keypoints not available for augmentation, skipping flip augmentation for this video")
                else:
                    # Flip keypoints horizontally
                    flipped_keypoints = [
                        flip_keypoints_horizontally(kp) if kp is not None else None
                        for kp in video_data["keypoints"]
                    ]
                    
                    # Normalize flipped keypoints (use same use_3d setting)
                    flip_normalizer = PoseNormalizer(use_3d=self.exercise.config.use_3d)
                    flipped_normalized = flip_normalizer.normalize_sequence(flipped_keypoints)
                    
                    # Get pose vectors from flipped normalized keypoints
                    flipped_pose_vectors = [
                        flip_normalizer.get_pose_vector(kp) for kp in flipped_normalized
                    ]
                    
                    # Embed flipped poses
                    if self.exercise.embedder and self.exercise.embedder.is_fitted:
                        flipped_embeddings = self.exercise.embedder.embed_sequence(
                            flipped_pose_vectors
                        )
                        print(f"  Created flipped augmentation ({len([e for e in flipped_embeddings if e is not None])} valid embeddings)")

            # Get rep boundaries
            video_rep_boundaries = None
            auto_detector = None
            if rep_boundaries and video_idx < len(rep_boundaries):
                video_rep_boundaries = rep_boundaries[video_idx]
            else:
                # Automatically detect reps
                print(f"  Automatically detecting reps in training video {video_idx + 1}...")
                auto_detector = AutoRepDetector(
                    min_rep_duration=self.exercise.config.min_rep_duration,
                    max_rep_duration=self.exercise.config.max_rep_duration,
                )
                video_rep_boundaries = auto_detector.detect_rep_boundaries(
                    embeddings, video_data["timestamps"]
                )
                print(f"  Detected {len(video_rep_boundaries)} rep(s)")

            # Generate debug plot if requested (always compute similarity for debugging)
            if debug and debug_dir is not None:
                # Create detector if not already created
                if auto_detector is None:
                    auto_detector = AutoRepDetector(
                        min_rep_duration=self.exercise.config.min_rep_duration,
                        max_rep_duration=self.exercise.config.max_rep_duration,
                    )

                similarities = auto_detector.compute_similarity_to_start(embeddings)
                smoothed = auto_detector.smooth_similarities(similarities)
                timestamps = video_data["timestamps"]

                # Validate data lengths match
                if len(similarities) != len(timestamps):
                    print(f"  Warning: Mismatch in data lengths (similarities: {len(similarities)}, timestamps: {len(timestamps)}). Skipping plot.")
                    continue

                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timestamps, similarities, label="Similarity to Start", alpha=0.5, linewidth=1)
                ax.plot(timestamps, smoothed, label="Smoothed", linewidth=2)
                ax.axhline(
                    y=auto_detector.similarity_threshold,
                    color="r",
                    linestyle="--",
                    label=f"Threshold ({auto_detector.similarity_threshold})",
                )

                # Mark rep boundaries if available
                if video_rep_boundaries and len(video_rep_boundaries) > 0:
                    for i, (start_idx, end_idx) in enumerate(video_rep_boundaries):
                        ax.axvspan(
                            timestamps[start_idx],
                            timestamps[end_idx],
                            alpha=0.2,
                            color="green",
                            label="Rep Boundary" if i == 0 else "",
                        )

                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Similarity to Start (0-1)")
                ax.set_title(f"Similarity to Start: {Path(video_path).name}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Save plot
                video_name = Path(video_path).stem
                plot_path = debug_dir / f"similarity_to_start_{video_name}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved similarity plot to: {plot_path}")

            # Assign phases
            if video_rep_boundaries and len(video_rep_boundaries) > 0:
                labeled_data = trajectory_builder.assign_phase_labels(
                    embeddings, video_rep_boundaries
                )
                for emb, phase in labeled_data:
                    all_embeddings.append(emb)
                    all_phases.append(phase)
                
                # Add flipped augmentation with same phase labels
                if flipped_embeddings is not None:
                    flipped_labeled_data = trajectory_builder.assign_phase_labels(
                        flipped_embeddings, video_rep_boundaries
                    )
                    for emb, phase in flipped_labeled_data:
                        all_embeddings.append(emb)
                        all_phases.append(phase)
            else:
                # Fallback: linear phase assignment
                print(f"  Warning: No reps detected, using linear phase assignment")
                num_frames = len(embeddings)
                for i, emb in enumerate(embeddings):
                    if emb is not None:
                        phase = i / num_frames if num_frames > 0 else 0.0
                        all_embeddings.append(emb)
                        all_phases.append(phase)
                
                # Add flipped augmentation
                if flipped_embeddings is not None:
                    for i, emb in enumerate(flipped_embeddings):
                        if emb is not None:
                            phase = i / num_frames if num_frames > 0 else 0.0
                            all_embeddings.append(emb)
                            all_phases.append(phase)

        if len(all_embeddings) == 0:
            raise ValueError("No training data collected")

        print(f"Training phase model on {len(all_embeddings)} samples...")
        print(f"  Using sliding window size: {self.exercise.config.phase_window_size}")
        self.exercise.phase_model = PhaseModel(
            window_size=self.exercise.config.phase_window_size
        )
        self.exercise.phase_model.fit(all_embeddings, all_phases)

        # Save exercise
        self.exercise.save()

        result = {
            "num_samples": len(all_embeddings),
            "model_trained": True,
        }

        if debug and debug_dir is not None:
            result["debug_plots"] = str(debug_dir)

        return result

    def analyze_video(
        self, video_path: str, force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a video for rep count and form.

        Args:
            video_path: Path to video file
            force_reprocess: If True, reprocess even if cached

        Returns:
            Dictionary with analysis results
        """
        if self.exercise.embedder is None or not self.exercise.embedder.is_fitted:
            raise ValueError("Embedder must be fitted before analysis")

        if self.exercise.phase_model is None or not self.exercise.phase_model.is_fitted:
            raise ValueError("Phase model must be trained before analysis")

        # Process video (use same use_3d setting as training)
        use_3d = self.exercise.config.use_3d
        pose_extractor = MediaPipeExtractor(use_3d=use_3d)
        normalizer = PoseNormalizer(use_3d=use_3d)
        video_processor = VideoProcessor(
            pose_extractor=pose_extractor,
            normalizer=normalizer,
            embedder=self.exercise.embedder
        )
        cache_dir = str(self.exercise.exercise_dir)
        video_data = video_processor.process_video(
            video_path, cache_dir=cache_dir, force_reprocess=force_reprocess
        )

        embeddings = video_data["embeddings"]
        if embeddings is None:
            embeddings = self.exercise.embedder.embed_sequence(
                video_data["pose_vectors"], log_timing=True
            )

        timestamps = video_data["timestamps"]

        # Reset phase model buffer for new video
        if hasattr(self.exercise.phase_model, 'reset_buffer'):
            self.exercise.phase_model.reset_buffer()

        # Estimate phases
        phases = self.exercise.phase_model.predict_phases(embeddings, log_timing=True)

        # Track phases
        unwrapped_phases = self.exercise.phase_tracker.track_phase(
            phases, timestamps, smooth=True, unwrap=True
        )

        # Count reps
        rep_count, rep_details = self.exercise.rep_counter.count_reps(
            unwrapped_phases, timestamps
        )

        # Validate reps
        rep_boundaries = [
            (rep["start_idx"], rep["end_idx"]) for rep in rep_details
        ]
        valid_rep_boundaries = self.exercise.rep_validator.filter_reps(
            rep_boundaries, unwrapped_phases, timestamps
        )

        # Score form for each rep
        rep_scores = []
        if self.exercise.form_scorer:
            for start_idx, end_idx in valid_rep_boundaries:
                # Get phases for this rep (wrapped back to [0, 1))
                rep_phases = [
                    p % 1.0 if p is not None else None
                    for p in phases[start_idx : end_idx + 1]
                ]
                rep_embeddings = embeddings[start_idx : end_idx + 1]

                score = self.exercise.form_scorer.score_rep(
                    start_idx, end_idx, rep_embeddings, rep_phases
                )
                rep_scores.append(score)

        # Overall trajectory comparison
        overall_comparison = None
        if self.exercise.trajectory_comparator:
            overall_comparison = self.exercise.trajectory_comparator.compare_trajectory(
                embeddings, phases
            )

        return {
            "video_path": video_path,
            "rep_count": len(valid_rep_boundaries),
            "rep_details": rep_details[: len(valid_rep_boundaries)],
            "rep_scores": rep_scores,
            "overall_comparison": overall_comparison,
            "num_frames": len(embeddings),
            "duration": timestamps[-1] if timestamps else 0.0,
        }

