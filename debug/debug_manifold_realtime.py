#!/usr/bin/env python3
"""Debug script to visualize real-time video processing on the manifold."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Configure logging for timing information
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from exercise_tracker.exercise_framework import ExerciseRegistry, ExerciseConfig
from exercise_tracker.data import RealtimeVideoProcessor
from exercise_tracker.pose_extraction import MediaPipeExtractor
from exercise_tracker.pose_processing import PoseNormalizer
from exercise_tracker.phase_estimation import PhaseTracker
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Debug real-time manifold: visualize video trajectory on canonical manifold (streaming mode)"
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
        help="Path to video file to analyze (processed as mock stream)",
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

    print(f"Processing video as mock stream: {args.video}")
    print(f"Output directory: {output_dir}")

    # Initialize real-time processor
    window_size = exercise.config.phase_window_size
    use_3d = exercise.config.use_3d
    print(f"\nUsing window size: {window_size}")
    print(f"Using 3D coordinates: {use_3d}")
    
    pose_extractor = MediaPipeExtractor(use_3d=use_3d)
    normalizer = PoseNormalizer(use_3d=use_3d)
    
    realtime_processor = RealtimeVideoProcessor(
        pose_extractor=pose_extractor,
        normalizer=normalizer,
        embedder=exercise.embedder,
        window_size=window_size,
    )

    # Reset phase model buffer
    if hasattr(exercise.phase_model, 'reset_buffer'):
        exercise.phase_model.reset_buffer()

    # Process video as stream
    print("\n=== Real-time Streaming Processing ===")
    print("Processing frames one-by-one (streaming mode)...\n")

    all_embeddings = []
    all_phases = []
    all_unwrapped_phases = []
    all_deviations = []
    all_timestamps = []
    all_canonical_phases = []
    
    frame_times = []
    embedding_times = []
    phase_times = []
    deviation_times = []

    stream_start = time.perf_counter()
    frame_count = 0

    # Open video capture (emulating camera feed)
    video_capture = cv2.VideoCapture(args.video)
    if not video_capture.isOpened():
        print(f"Error: Could not open video: {args.video}")
        sys.exit(1)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default fallback
    print(f"Video FPS: {fps:.2f}")

    try:
        for result in realtime_processor.process_video_stream(video_capture, fps=fps):
            frame_start = time.perf_counter()
            
            if args.max_frames and frame_count >= args.max_frames:
                break

            embedding = result["embedding"]
            window_embeddings = result["window_embeddings"]
            timestamp = result["timestamp"]
            frame_idx = result["frame_index"]

            all_embeddings.append(embedding)
            all_timestamps.append(timestamp)

            # Process embedding to get phase
            phase_start = time.perf_counter()
            # For phase prediction, we use the current embedding with the model's buffer
            # The model will handle the window internally via its buffer
            phase = exercise.phase_model.predict_phase(
                embedding, update_buffer=True, log_timing=False
            )
            phase_time = time.perf_counter() - phase_start
            phase_times.append(phase_time)

            all_phases.append(phase)

            # Track phases (need to accumulate for smoothing)
            if phase is not None:
                # For real-time, we track incrementally
                # We'll do full tracking at the end, but for now just collect
                pass

            # Compute deviation from canonical manifold
            if phase is not None:
                dev_start = time.perf_counter()
                try:
                    total_dev, _ = exercise.manifold_builder.compute_deviation(embedding, phase)
                    all_deviations.append(total_dev)
                    all_canonical_phases.append(phase)
                except Exception:
                    all_deviations.append(None)
                    all_canonical_phases.append(None)
                dev_time = time.perf_counter() - dev_start
                deviation_times.append(dev_time)
            else:
                all_deviations.append(None)
                all_canonical_phases.append(None)
                deviation_times.append(0.0)

            frame_time = time.perf_counter() - frame_start
            frame_times.append(frame_time)

            frame_count += 1

            # Log every 30 frames or at start/end
            if frame_count == 1 or frame_count % 30 == 0:
                phase_ms = phase_times[-1]*1000 if phase_times else 0.0
                dev_ms = deviation_times[-1]*1000 if deviation_times else 0.0
                print(
                    f"  Frame {frame_count}: "
                    f"phase={phase_ms:.3f}ms, "
                    f"dev={dev_ms:.3f}ms, "
                    f"total={frame_time*1000:.3f}ms"
                )

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release video capture (caller responsibility)
        video_capture.release()

    stream_time = time.perf_counter() - stream_start

    # Now do phase tracking on all collected phases
    print("\nTracking phases (smoothing + unwrapping)...")
    track_start = time.perf_counter()
    all_unwrapped_phases = exercise.phase_tracker.track_phase(
        all_phases, all_timestamps, smooth=True, unwrap=True
    )
    track_time = time.perf_counter() - track_start
    print(f"Phase tracking: {track_time*1000:.1f} ms")

    # Print timing statistics
    print("\n=== Timing Statistics ===")
    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times) * 1000
        max_frame_time = max(frame_times) * 1000
        min_frame_time = min(frame_times) * 1000
        print(f"Frame processing:")
        print(f"  Total: {stream_time*1000:.1f} ms for {frame_count} frames")
        print(f"  Average: {avg_frame_time:.3f} ms/frame")
        print(f"  Min: {min_frame_time:.3f} ms, Max: {max_frame_time:.3f} ms")

    if phase_times:
        avg_phase_time = sum(phase_times) / len(phase_times) * 1000
        print(f"Phase prediction:")
        print(f"  Average: {avg_phase_time:.3f} ms/frame")

    if deviation_times:
        valid_dev_times = [t for t in deviation_times if t > 0]
        if valid_dev_times:
            avg_dev_time = sum(valid_dev_times) / len(valid_dev_times) * 1000
            print(f"Deviation computation:")
            print(f"  Average: {avg_dev_time:.3f} ms/frame")

    # Real-time viability
    if frame_times:
        avg_total_per_frame = avg_frame_time
        theoretical_fps = 1000 / avg_total_per_frame if avg_total_per_frame > 0 else 0
        print(f"\n=== Real-time Viability Summary ===")
        print(f"Average per frame: {avg_total_per_frame:.3f} ms")
        print(f"Theoretical max FPS: {theoretical_fps:.1f} fps")
        if avg_total_per_frame < 33.33:  # 30 fps = 33.33 ms/frame
            print("✓ Viable for real-time at 30 fps")
        elif avg_total_per_frame < 16.67:  # 60 fps = 16.67 ms/frame
            print("✓ Viable for real-time at 60 fps")
        else:
            print("⚠ May struggle with real-time (target: <33ms for 30fps)")
        print("=" * 40 + "\n")

    # Create plots (same as debug_manifold.py)
    video_name = Path(args.video).stem

    # Plot 1: Phase progression over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Phase plot
    valid_indices = [i for i, p in enumerate(all_phases) if p is not None]
    if len(valid_indices) > 0:
        valid_times = [all_timestamps[i] for i in valid_indices]
        valid_phases = [all_phases[i] for i in valid_indices]
        valid_unwrapped = [all_unwrapped_phases[i] for i in valid_indices if all_unwrapped_phases[i] is not None]

        ax1.plot(valid_times, valid_phases, label="Phase (wrapped)", alpha=0.7, linewidth=1)
        if len(valid_unwrapped) == len(valid_times):
            ax1_twin = ax1.twinx()
            ax1_twin.plot(valid_times, valid_unwrapped, label="Phase (unwrapped)", color="orange", linewidth=2)
            ax1_twin.set_ylabel("Unwrapped Phase", color="orange")
            ax1_twin.tick_params(axis="y", labelcolor="orange")

        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Phase (0-1)", color="blue")
        ax1.set_title(f"Phase Progression (Real-time): {video_name}")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

    # Deviation plot
    valid_dev_indices = [i for i, d in enumerate(all_deviations) if d is not None]
    if len(valid_dev_indices) > 0:
        valid_dev_times = [all_timestamps[i] for i in valid_dev_indices]
        valid_deviations = [all_deviations[i] for i in valid_dev_indices]

        ax2.plot(valid_dev_times, valid_deviations, label="Deviation from Canonical", color="red", linewidth=2)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Deviation (lower = better form)")
        ax2.set_title(f"Form Adherence (Real-time): {video_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_path = output_dir / f"realtime_manifold_phase_progression_{video_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved real-time manifold debug plot to: {plot_path}")

    # Plot 2: Phase vs Deviation (scatter)
    fig, ax = plt.subplots(figsize=(12, 8))

    valid_data = [
        (all_canonical_phases[i], all_deviations[i])
        for i in range(len(all_canonical_phases))
        if all_canonical_phases[i] is not None and all_deviations[i] is not None
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

        # Overlay canonical manifold mean deviation
        phase_bins = np.linspace(0, 1, 20, endpoint=False)
        mean_deviations = []
        for phase_bin in phase_bins:
            try:
                canonical_mean, canonical_std = exercise.manifold_builder.get_canonical_pose(phase_bin)
                mean_dev = np.mean(canonical_std)
                mean_deviations.append(mean_dev)
            except Exception:
                mean_deviations.append(0.0)

        ax.plot(phase_bins, mean_deviations, "k--", linewidth=2, label="Canonical (mean std)", alpha=0.5)

        ax.set_xlabel("Phase (θ)")
        ax.set_ylabel("Deviation from Canonical")
        ax.set_title(f"Phase vs Deviation (Real-time): {video_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plot_path2 = output_dir / f"realtime_manifold_phase_deviation_{video_name}_{timestamp}.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved real-time phase-deviation plot to: {plot_path2}")

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
        print(f"  Valid frames: {len(valid_data)}/{len(all_embeddings)}")


if __name__ == "__main__":
    main()

