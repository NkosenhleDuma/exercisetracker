"""Generate a debug overlay video for rep analysis.

Overlays:
- Phase (wrapped and unwrapped)
- Rep index and validity
- Similarity-to-start (auto rep detector signal)
- Canonical deviation (if manifold is available)
- Pose skeleton

Usage:
    python debug/debug_overlay.py --exercise squats --video path/to.mp4 --output debug_overlay.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from exercise_tracker.exercise_framework import ExerciseRegistry
from exercise_tracker.data import VideoProcessor
from exercise_tracker.rep_counting import AutoRepDetector


# Basic MediaPipe Pose connections for drawing a skeleton
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 13),  # left shoulder -> left elbow
    (13, 15),  # left elbow -> left wrist
    (12, 14),  # right shoulder -> right elbow
    (14, 16),  # right elbow -> right wrist
    (11, 12),  # left shoulder -> right shoulder
    (11, 23),  # left shoulder -> left hip
    (12, 24),  # right shoulder -> right hip
    (23, 24),  # left hip -> right hip
    (23, 25),  # left hip -> left knee
    (25, 27),  # left knee -> left ankle
    (24, 26),  # right hip -> right knee
    (26, 28),  # right knee -> right ankle
    (27, 31),  # left ankle -> left foot index
    (28, 32),  # right ankle -> right foot index
]


def current_rep_index(idx: int, boundaries: List[Tuple[int, int]]) -> Optional[int]:
    """Return the (0-based) rep index for a frame, or None if not inside a valid rep."""
    for rep_idx, (start, end) in enumerate(boundaries):
        if start <= idx <= end:
            return rep_idx
    return None


def draw_skeleton(frame: np.ndarray, keypoints: Optional[np.ndarray]) -> None:
    """Draw keypoints and skeleton on the frame."""
    if keypoints is None:
        return
    h, w = frame.shape[:2]

    # keypoints are normalized [x, y, confidence]
    pts = []
    for x, y, c in keypoints:
        if c < 0.1:
            pts.append(None)
            continue
        pts.append((int(x * w), int(y * h)))

    # draw connections
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts) and pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    # draw points
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)


def overlay_text(
    frame: np.ndarray,
    lines: List[str],
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw multi-line text on the frame."""
    x, y = 10, 25
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 22


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rep debug overlay video.")
    parser.add_argument("--exercise", required=True, help="Exercise name")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output debug video")
    parser.add_argument("--data-root", default="data", help="Data root (default: data)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to process")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load exercise (registry will pull config/manifold/model if saved)
    registry = ExerciseRegistry()
    exercise = registry.load_exercise(args.exercise, data_root=args.data_root)

    # Use same use_3d setting as exercise was trained with
    use_3d = exercise.config.use_3d
    from exercise_tracker.pose_extraction import MediaPipeExtractor
    from exercise_tracker.pose_processing import PoseNormalizer
    
    # Create extractor and normalizer with same use_3d setting
    pose_extractor = MediaPipeExtractor(use_3d=use_3d)
    normalizer = PoseNormalizer(use_3d=use_3d)
    vp = VideoProcessor(
        pose_extractor=pose_extractor,
        normalizer=normalizer,
        embedder=exercise.embedder
    )
    video_data = vp.process_video(str(video_path), cache_dir=str(exercise.exercise_dir), force_reprocess=True)

    embeddings = video_data["embeddings"]
    if embeddings is None:
        embeddings = exercise.embedder.embed_sequence(video_data["pose_vectors"])

    timestamps = video_data["timestamps"]
    keypoints_seq = video_data["keypoints"]

    # Ensure phase model exists
    if exercise.phase_model is None or not exercise.phase_model.is_fitted:
        raise RuntimeError(
            "Phase model is not available. Please train the model first:\n"
            "  python -m exercise_tracker.cli train --exercise {exercise}"
        )

    # Phase estimation
    phases = exercise.phase_model.predict_phases(embeddings)
    unwrapped = exercise.phase_tracker.track_phase(phases, timestamps, smooth=True, unwrap=True)

    # Rep counting + validation
    rep_count, rep_details = exercise.rep_counter.count_reps(unwrapped, timestamps)
    rep_boundaries = [(rep["start_idx"], rep["end_idx"]) for rep in rep_details]
    valid_boundaries = exercise.rep_validator.filter_reps(rep_boundaries, unwrapped, timestamps)

    # Similarity to start (auto rep detector signal)
    similarity = AutoRepDetector(min_rep_duration=exercise.config.min_rep_duration).compute_similarity_to_start(embeddings)

    # Canonical deviation (if manifold available)
    canonical_dev: List[Optional[float]] = []
    if exercise.manifold_builder and exercise.manifold_builder.manifold_mean is not None:
        for emb, phase in zip(embeddings, phases):
            if emb is None or phase is None:
                canonical_dev.append(None)
                continue
            total_dev, _ = exercise.manifold_builder.compute_deviation(emb, phase)
            canonical_dev.append(float(total_dev))
    else:
        canonical_dev = [None] * len(embeddings)

    # Open video for frame-wise overlay
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        # Draw skeleton
        kp = keypoints_seq[frame_idx] if frame_idx < len(keypoints_seq) else None
        draw_skeleton(frame, kp)

        # Collect data for overlays
        phase = phases[frame_idx] if frame_idx < len(phases) else None
        unwrap = unwrapped[frame_idx] if frame_idx < len(unwrapped) else None
        sim = similarity[frame_idx] if frame_idx < len(similarity) else None
        dev = canonical_dev[frame_idx] if frame_idx < len(canonical_dev) else None
        t = timestamps[frame_idx] if frame_idx < len(timestamps) else 0.0

        rep_idx = current_rep_index(frame_idx, rep_boundaries)
        valid_rep_idx = current_rep_index(frame_idx, valid_boundaries)

        # Text overlays
        lines = [
            f"frame: {frame_idx}",
            f"time: {t:.2f}s",
            f"phase: {phase:.3f}" if phase is not None else "phase: -",
            f"unwrapped: {unwrap:.3f}" if unwrap is not None else "unwrapped: -",
            f"rep (detected): {rep_idx if rep_idx is not None else '-'}",
            f"rep (valid): {valid_rep_idx if valid_rep_idx is not None else '-'}",
            f"similarity-to-start: {sim:.3f}" if sim is not None else "similarity-to-start: -",
            f"canonical deviation: {dev:.3f}" if dev is not None else "canonical deviation: -",
        ]

        # Highlight if inside a valid rep
        color = (0, 255, 0) if valid_rep_idx is not None else (0, 0, 255)
        overlay_text(frame, lines, color=color)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved debug overlay to: {output_path}")


if __name__ == "__main__":
    main()


