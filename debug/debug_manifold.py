#!/usr/bin/env python3
"""Debug script to visualize where a video lies on the manifold."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configure logging for timing information
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from exercise_tracker.exercise_framework import ExerciseRegistry, ExerciseConfig
from exercise_tracker.data import VideoProcessor
from exercise_tracker.pose_extraction import MediaPipeExtractor
from exercise_tracker.pose_processing import PoseNormalizer
from exercise_tracker.phase_estimation import PhaseTracker


def main():
    parser = argparse.ArgumentParser(
        description="Debug manifold: visualize video trajectory on canonical manifold"
    )
    parser.add_argument(
        "--exercise",
        type=str,
        required=True,
        help="Exercise name",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: processed/<exercise>/debug)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for data",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for testing)",
    )

    args = parser.parse_args()

    # Load exercise
    registry = ExerciseRegistry()
    try:
        exercise = registry.load_exercise(args.exercise, data_root=args.data_root)
    except Exception as e:
        print(f"Error loading exercise: {e}")
        print(f"Make sure the exercise '{args.exercise}' has been ingested and trained.")
        sys.exit(1)

    # Check required components
    if exercise.embedder is None or not exercise.embedder.is_fitted:
        print("Error: Embedder not fitted. Please run ingest first.")
        sys.exit(1)

    if exercise.phase_model is None or not exercise.phase_model.is_fitted:
        print("Error: Phase model not trained. Please run train first.")
        sys.exit(1)

    if exercise.manifold_builder is None or exercise.manifold_builder.manifold_mean is None:
        print("Error: Manifold not built. Please run ingest first.")
        sys.exit(1)

    # Setup output directory
    if args.output is None:
        output_dir = exercise.exercise_dir / "debug"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing video: {args.video}")
    print(f"Output directory: {output_dir}")

    # Use same use_3d setting as exercise was trained with
    use_3d = exercise.config.use_3d
    print(f"Using 3D coordinates: {use_3d}")
    
    # Check if time-based model
    is_time_based = (
        exercise.config.phase_model_type == "time_based" or 
        exercise.config.phase_window_mode == "time" or
        hasattr(exercise.phase_model, 'window_duration')
    )
    if is_time_based:
        model_backend = "PyTorch Lightning (LSTM)"
        print(f"Using time-based model (window duration: {exercise.config.phase_window_duration}s)")
        print(f"Model backend: {model_backend}")
    else:
        # Check if MLP (PyTorch) or RandomForest (sklearn)
        if hasattr(exercise.phase_model, 'model_type'):
            if exercise.phase_model.model_type == "mlp":
                model_backend = "PyTorch (MLP)"
            else:
                model_backend = "sklearn (RandomForest)"
        else:
            model_backend = "Unknown"
        print(f"Using frame-based model (window size: {exercise.config.phase_window_size} frames)")
        print(f"Model backend: {model_backend}")

    # Process video
    import time
    print("\n=== Timing Debug Information ===")
    print("Note: Current processing is BATCH mode (entire video first).")
    print("For real-time streaming, frame-by-frame processing would be needed.\n")
    
    # Create extractor and normalizer with same use_3d setting
    pose_extractor = MediaPipeExtractor(use_3d=use_3d)
    normalizer = PoseNormalizer(use_3d=use_3d)
    video_processor = VideoProcessor(
        pose_extractor=pose_extractor,
        normalizer=normalizer,
        embedder=exercise.embedder
    )
    cache_dir = str(exercise.exercise_dir)
    
    video_start = time.perf_counter()
    video_data = video_processor.process_video(
        args.video, cache_dir=cache_dir, force_reprocess=False
    )
    video_time = time.perf_counter() - video_start
    num_frames = len(video_data["timestamps"])
    print(f"Video processing (pose extraction + normalization):")
    print(f"  Total: {video_time*1000:.1f} ms for {num_frames} frames")
    if num_frames > 0:
        print(f"  Average: {video_time*1000/num_frames:.3f} ms/frame")

    embeddings = video_data["embeddings"]
    embed_time = 0.0
    if embeddings is None:
        embed_start = time.perf_counter()
        embeddings = exercise.embedder.embed_sequence(video_data["pose_vectors"], log_timing=True)
        embed_time = time.perf_counter() - embed_start
        print(f"Embedding generation: {embed_time*1000:.1f} ms total")
    else:
        print(f"Embeddings loaded from cache ({len([e for e in embeddings if e is not None])} valid)")

    timestamps = video_data["timestamps"]

    # Limit frames if requested
    if args.max_frames and len(embeddings) > args.max_frames:
        embeddings = embeddings[: args.max_frames]
        timestamps = timestamps[: args.max_frames]
        num_frames = len(embeddings)  # Update num_frames if limited

    # Estimate phases (pass timestamps for time-based models)
    phase_start = time.perf_counter()
    if is_time_based:
        phases = exercise.phase_model.predict_phases(embeddings, timestamps=timestamps, log_timing=True)
    else:
        phases = exercise.phase_model.predict_phases(embeddings, log_timing=True)
    phase_time = time.perf_counter() - phase_start
    print(f"Phase prediction: {phase_time*1000:.1f} ms total")

    # Track phases
    track_start = time.perf_counter()
    unwrapped_phases = exercise.phase_tracker.track_phase(
        phases, timestamps, smooth=True, unwrap=True
    )
    track_time = time.perf_counter() - track_start
    print(f"Phase tracking (smoothing + unwrapping): {track_time*1000:.1f} ms")

    # Compute deviations from canonical manifold (streaming-like processing)
    deviation_start = time.perf_counter()
    deviations = []
    canonical_phases = []
    frame_times = []
    
    print("\nProcessing frames (streaming-like):")
    for i, (emb, phase) in enumerate(zip(embeddings, phases)):
        frame_start = time.perf_counter()
        
        if emb is None or phase is None:
            deviations.append(None)
            canonical_phases.append(None)
        else:
            try:
                total_dev, _ = exercise.manifold_builder.compute_deviation(emb, phase)
                deviations.append(total_dev)
                canonical_phases.append(phase)
            except Exception:
                deviations.append(None)
                canonical_phases.append(None)
        
        frame_time = time.perf_counter() - frame_start
        frame_times.append(frame_time)
        
        # Log every 30 frames or at start/end
        if i == 0 or i == len(embeddings) - 1 or (i + 1) % 30 == 0:
            print(f"  Frame {i+1}/{len(embeddings)}: {frame_time*1000:.3f} ms")
    
    deviation_time = time.perf_counter() - deviation_start
    valid_frame_times = [t for t in frame_times if t > 0]
    if valid_frame_times:
        avg_frame_time = sum(valid_frame_times) / len(valid_frame_times) * 1000
        max_frame_time = max(valid_frame_times) * 1000
        min_frame_time = min(valid_frame_times) * 1000
        print(f"\nFrame processing stats:")
        print(f"  Total: {deviation_time*1000:.1f} ms for {len(embeddings)} frames")
        print(f"  Average: {avg_frame_time:.3f} ms/frame")
        print(f"  Min: {min_frame_time:.3f} ms, Max: {max_frame_time:.3f} ms")
        print(f"  Theoretical max FPS: {1000/avg_frame_time:.1f} fps")
    
    # Summary for real-time viability
    total_processing_time = video_time + embed_time + phase_time + track_time + deviation_time
    if num_frames > 0:
        avg_total_per_frame = total_processing_time / num_frames * 1000  # ms
        theoretical_fps = 1000 / avg_total_per_frame if avg_total_per_frame > 0 else 0
        print("\n=== Real-time Viability Summary ===")
        print(f"Total processing time: {total_processing_time*1000:.1f} ms for {num_frames} frames")
        print(f"Average per frame: {avg_total_per_frame:.3f} ms")
        print(f"Theoretical max FPS: {theoretical_fps:.1f} fps")
        if avg_total_per_frame < 33.33:  # 30 fps = 33.33 ms/frame
            print("✓ Viable for real-time at 30 fps")
        elif avg_total_per_frame < 16.67:  # 60 fps = 16.67 ms/frame
            print("✓ Viable for real-time at 60 fps")
        else:
            print("⚠ May struggle with real-time (target: <33ms for 30fps)")
        print("=" * 40 + "\n")

    # Create plots
    video_name = Path(args.video).stem

    # Plot 1: Phase progression over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Phase plot
    valid_indices = [i for i, p in enumerate(phases) if p is not None]
    if len(valid_indices) > 0:
        valid_times = [timestamps[i] for i in valid_indices]
        valid_phases = [phases[i] for i in valid_indices]
        valid_unwrapped = [unwrapped_phases[i] for i in valid_indices if unwrapped_phases[i] is not None]

        ax1.plot(valid_times, valid_phases, label="Phase (wrapped)", alpha=0.7, linewidth=1)
        if len(valid_unwrapped) == len(valid_times):
            ax1_twin = ax1.twinx()
            ax1_twin.plot(valid_times, valid_unwrapped, label="Phase (unwrapped)", color="orange", linewidth=2)
            ax1_twin.set_ylabel("Unwrapped Phase", color="orange")
            ax1_twin.tick_params(axis="y", labelcolor="orange")

        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Phase (0-1)", color="blue")
        ax1.set_title(f"Phase Progression: {video_name}")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

    # Deviation plot
    valid_dev_indices = [i for i, d in enumerate(deviations) if d is not None]
    if len(valid_dev_indices) > 0:
        valid_dev_times = [timestamps[i] for i in valid_dev_indices]
        valid_deviations = [deviations[i] for i in valid_dev_indices]

        ax2.plot(valid_dev_times, valid_deviations, label="Deviation from Canonical", color="red", linewidth=2)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Deviation (lower = better form)")
        ax2.set_title(f"Form Adherence: {video_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_path = output_dir / f"debug_manifold_phase_progression_{video_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved manifold debug plot to: {plot_path}")

    # Plot 2: Phase vs Deviation (scatter)
    fig, ax = plt.subplots(figsize=(12, 8))

    valid_data = [
        (canonical_phases[i], deviations[i])
        for i in range(len(canonical_phases))
        if canonical_phases[i] is not None and deviations[i] is not None
    ]

    if len(valid_data) > 0:
        phases_plot = [p for p, d in valid_data]
        deviations_plot = [d for p, d in valid_data]

        scatter = ax.scatter(
            phases_plot,
            deviations_plot,
            c=deviations_plot,
            cmap="RdYlGn_r",
            alpha=0.6,
            s=20,
        )
        plt.colorbar(scatter, ax=ax, label="Deviation")

        # Overlay canonical manifold mean deviation (if we can compute it)
        phase_bins = np.linspace(0, 1, 20, endpoint=False)
        mean_deviations = []
        for phase_bin in phase_bins:
            try:
                canonical_mean, canonical_std = exercise.manifold_builder.get_canonical_pose(phase_bin)
                # Compute average deviation at this phase (using std as proxy)
                mean_dev = np.mean(canonical_std)
                mean_deviations.append(mean_dev)
            except Exception:
                mean_deviations.append(0.0)

        ax.plot(phase_bins, mean_deviations, "k--", linewidth=2, label="Canonical (mean std)", alpha=0.5)

        ax.set_xlabel("Phase (θ)")
        ax.set_ylabel("Deviation from Canonical")
        ax.set_title(f"Phase vs Deviation: {video_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plot_path2 = output_dir / f"debug_manifold_phase_deviation_{video_name}_{timestamp}.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phase-deviation plot to: {plot_path2}")

    # Print summary statistics
    if len(valid_data) > 0:
        avg_deviation = np.mean(deviations_plot)
        min_deviation = np.min(deviations_plot)
        max_deviation = np.max(deviations_plot)
        std_deviation = np.std(deviations_plot)

        print("\nSummary Statistics:")
        print(f"  Average deviation: {avg_deviation:.4f}")
        print(f"  Min deviation: {min_deviation:.4f}")
        print(f"  Max deviation: {max_deviation:.4f}")
        print(f"  Std deviation: {std_deviation:.4f}")
        print(f"  Valid frames: {len(valid_data)}/{len(embeddings)}")


if __name__ == "__main__":
    main()

