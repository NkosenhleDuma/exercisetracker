#!/usr/bin/env python3
"""Debug script to visualize real-time camera feed with phase and concordance."""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import time
from collections import deque
from typing import Optional, Tuple

from exercise_tracker.exercise_framework import ExerciseRegistry
from exercise_tracker.data import RealtimeVideoProcessor
from exercise_tracker.pose_extraction import MediaPipeExtractor
from exercise_tracker.pose_processing import PoseNormalizer
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg for interactive display
import matplotlib.pyplot as plt


def create_overlay_image(
    frame: np.ndarray,
    phase: Optional[float],
    deviation: Optional[float],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """
    Create an overlay image with phase and deviation information.
    
    Args:
        frame: Input frame
        phase: Current phase value (0-1)
        deviation: Current deviation/concordance value
        frame_idx: Current frame index
        fps: Frames per second
        
    Returns:
        Overlay image with text annotations
    """
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    
    # Create semi-transparent overlay for text
    overlay_alpha = overlay.copy()
    cv2.rectangle(overlay_alpha, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay_alpha, 0.7, overlay, 0.3, 0, overlay)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    y_offset = 35
    line_height = 30
    
    # Frame info
    cv2.putText(
        overlay,
        f"Frame: {frame_idx} | FPS: {fps:.1f}",
        (20, y_offset),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    
    # Phase
    y_offset += line_height
    phase_text = f"Phase: {phase:.3f}" if phase is not None else "Phase: N/A"
    phase_color = (0, 255, 255) if phase is not None else (128, 128, 128)
    cv2.putText(
        overlay,
        phase_text,
        (20, y_offset),
        font,
        font_scale,
        phase_color,
        thickness,
    )
    
    # Deviation/Concordance
    y_offset += line_height
    if deviation is not None:
        # Lower deviation = better form (higher concordance)
        # We'll show both deviation and a "concordance" score
        concordance = max(0.0, 1.0 - min(1.0, deviation / 5.0))  # Normalize to 0-1
        dev_text = f"Deviation: {deviation:.4f}"
        conc_text = f"Concordance: {concordance:.2%}"
        
        # Color based on deviation (green = good, red = bad)
        if deviation < 1.0:
            color = (0, 255, 0)  # Green
        elif deviation < 2.0:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
            
        cv2.putText(
            overlay,
            dev_text,
            (20, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        y_offset += line_height
        cv2.putText(
            overlay,
            conc_text,
            (20, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
    else:
        cv2.putText(
            overlay,
            "Deviation: N/A",
            (20, y_offset),
            font,
            font_scale,
            (128, 128, 128),
            thickness,
        )
    
    return overlay


def create_phase_graph(
    phase_buffer: deque,
    time_buffer: deque,
    fig: plt.Figure,
    ax: plt.Axes,
) -> None:
    """
    Update the phase graph with current buffer data.
    
    Args:
        phase_buffer: Buffer of phase values
        time_buffer: Buffer of timestamps
        fig: Matplotlib figure
        ax: Matplotlib axes
    """
    ax.clear()
    
    if len(phase_buffer) > 0 and len(time_buffer) > 0:
        phases = list(phase_buffer)
        times = list(time_buffer)
        
        # Filter out None values
        valid_data = [(t, p) for t, p in zip(times, phases) if p is not None]
        if len(valid_data) > 0:
            valid_times, valid_phases = zip(*valid_data)
            ax.plot(valid_times, valid_phases, 'b-', linewidth=2, label='Phase', alpha=0.7)
            if len(valid_times) > 0:
                ax.scatter(valid_times[-1], valid_phases[-1], color='red', s=50, zorder=5, label='Current')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Phase (0-1)')
    ax.set_title('Phase Over Time (Real-time)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if len(phase_buffer) > 0:
        ax.legend(loc='upper right')
    
    # Adjust x-axis to show recent history
    if len(time_buffer) > 0:
        max_time = max(time_buffer)
        min_time = max(0, max_time - 10)  # Show last 10 seconds
        ax.set_xlim(min_time, max_time + 1)
    else:
        ax.set_xlim(0, 10)
    
    # Use draw_idle for better performance
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main():
    parser = argparse.ArgumentParser(
        description="Debug real-time camera feed with phase and concordance visualization"
    )
    parser.add_argument(
        "--exercise",
        type=str,
        required=True,
        help="Exercise name",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for data",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300,
        help="Size of phase buffer for graph (default: 300 frames ~10s at 30fps)",
    )
    parser.add_argument(
        "--graph-update-interval",
        type=int,
        default=5,
        help="Update graph every N frames (default: 5)",
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

    print(f"Loading exercise: {args.exercise}")
    print(f"Opening camera {args.camera}...")

    # Initialize real-time processor
    # Check if time-based model (use window_size=1 for time-based as it uses time windows)
    is_time_based = (
        exercise.config.phase_model_type == "time_based" or 
        exercise.config.phase_window_mode == "time" or
        hasattr(exercise.phase_model, 'window_duration')
    )
    
    if is_time_based:
        window_size = 1  # Time-based models handle windows internally
        model_backend = "PyTorch Lightning (LSTM)"
        print(f"Using time-based model (window duration: {exercise.config.phase_window_duration}s)")
        print(f"Model backend: {model_backend}")
    else:
        window_size = exercise.config.phase_window_size
        # Check if MLP (PyTorch) or RandomForest (sklearn)
        if hasattr(exercise.phase_model, 'model_type'):
            if exercise.phase_model.model_type == "mlp":
                model_backend = "PyTorch (MLP)"
            else:
                model_backend = "sklearn (RandomForest)"
        else:
            model_backend = "Unknown"
        print(f"Using frame-based model (window size: {window_size} frames)")
        print(f"Model backend: {model_backend}")
    
    use_3d = exercise.config.use_3d
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

    # Open camera
    video_capture = cv2.VideoCapture(args.camera)
    if not video_capture.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)

    # Set camera properties for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    # Get actual camera properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default fallback

    print(f"Camera resolution: {width}x{height}")
    print(f"Camera FPS: {fps:.2f}")
    print("\nStarting real-time processing...")
    print("Press 'q' to quit, 'r' to reset buffers")

    # Buffers for graph
    phase_buffer = deque(maxlen=args.buffer_size)
    time_buffer = deque(maxlen=args.buffer_size)
    deviation_buffer = deque(maxlen=args.buffer_size)

    # Initialize matplotlib figure for phase graph
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.canvas.manager.set_window_title(f"Phase Over Time - {args.exercise}")
    plt.ion()  # Turn on interactive mode
    plt.tight_layout()
    fig.show()
    fig.canvas.draw()

    frame_count = 0
    start_time = time.perf_counter()
    last_graph_update = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            frame_start = time.perf_counter()
            timestamp = (frame_start - start_time)

            # Process frame
            result = realtime_processor.process_frame(frame, timestamp)
            
            current_phase = None
            current_deviation = None

            if result is not None:
                embedding = result["embedding"]
                frame_idx = result["frame_index"]
                result_timestamp = result.get("timestamp", timestamp)

                # Predict phase (pass timestamp for time-based models)
                if is_time_based:
                    phase = exercise.phase_model.predict_phase(
                        embedding, timestamp=result_timestamp, update_buffer=True, log_timing=False
                    )
                else:
                    phase = exercise.phase_model.predict_phase(
                        embedding, update_buffer=True, log_timing=False
                    )
                current_phase = phase

                # Compute deviation
                if phase is not None:
                    try:
                        total_dev, _ = exercise.manifold_builder.compute_deviation(
                            embedding, phase
                        )
                        current_deviation = total_dev
                    except Exception:
                        current_deviation = None

            # Update buffers
            phase_buffer.append(current_phase)
            time_buffer.append(timestamp)
            deviation_buffer.append(current_deviation)

            # Create overlay
            overlay = create_overlay_image(
                frame, current_phase, current_deviation, frame_count, fps
            )

            # Display frame
            cv2.imshow("Exercise Tracker - Real-time", overlay)

            # Update graph periodically
            if frame_count - last_graph_update >= args.graph_update_interval:
                create_phase_graph(phase_buffer, time_buffer, fig, ax)
                last_graph_update = frame_count

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\nResetting buffers...")
                phase_buffer.clear()
                time_buffer.clear()
                deviation_buffer.clear()
                if hasattr(exercise.phase_model, 'reset_buffer'):
                    exercise.phase_model.reset_buffer()
                realtime_processor.reset()
                start_time = time.perf_counter()

            frame_count += 1

            # Log FPS periodically
            if frame_count % 30 == 0:
                elapsed = time.perf_counter() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames | FPS: {current_fps:.1f} | "
                      f"Phase: {round(current_phase, 3) if current_phase is not None else 'N/A'} | "
                      f"Deviation: {round(current_deviation, 4) if current_deviation is not None else 'N/A'}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        plt.close(fig)
        print("Camera released and windows closed.")


if __name__ == "__main__":
    main()
