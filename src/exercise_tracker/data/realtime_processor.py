"""Real-time video processing pipeline for streaming video feeds."""

import numpy as np
from typing import Optional, Iterator, Tuple, Dict, Any
import time

from ..pose_extraction import BasePoseExtractor
from ..pose_processing import PoseNormalizer, PoseEmbedder


class RealtimeVideoProcessor:
    """Process video frames in real-time for streaming feeds."""

    def __init__(
        self,
        pose_extractor: BasePoseExtractor,
        normalizer: PoseNormalizer,
        embedder: PoseEmbedder,
        window_size: int = 1,
    ):
        """
        Initialize real-time video processor.

        Args:
            pose_extractor: Pose extractor instance
            normalizer: Pose normalizer instance
            embedder: Pose embedder instance (must be fitted)
            window_size: Size of sliding window for embeddings (default: 1)
        """
        if embedder is None or not embedder.is_fitted:
            raise ValueError("Embedder must be fitted before real-time processing")

        self.pose_extractor = pose_extractor
        self.normalizer = normalizer
        self.embedder = embedder
        self.window_size = window_size

        # Buffers for sliding window
        self.embedding_buffer = []
        self.frame_count = 0
        self.start_time = None

    def reset(self) -> None:
        """Reset internal buffers (useful for new video/stream)."""
        self.embedding_buffer = []
        self.frame_count = 0
        self.start_time = None

    def process_frame(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single frame and return embedding if window is ready.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format
            timestamp: Optional timestamp for this frame (in seconds)

        Returns:
            Dictionary with:
            - 'embedding': Embedding vector (when window is ready), None otherwise
            - 'timestamp': Timestamp for this frame
            - 'frame_index': Frame index
            - 'keypoints': Raw keypoints (for debugging)
            - 'normalized_keypoints': Normalized keypoints (for debugging)
            Or None if pose extraction failed
        """
        if self.start_time is None:
            self.start_time = time.perf_counter()
            if timestamp is None:
                timestamp = 0.0

        # Extract pose from frame
        keypoints = self.pose_extractor.extract_pose(frame)
        if keypoints is None:
            return None

        # Normalize pose
        normalized_keypoints = self.normalizer.normalize(keypoints)
        if normalized_keypoints is None:
            return None

        # Get pose vector
        pose_vector = self.normalizer.get_pose_vector(normalized_keypoints)
        if pose_vector is None:
            return None

        # Embed pose
        embedding = self.embedder.embed(pose_vector, log_timing=False)
        if embedding is None:
            return None

        # Add to buffer
        self.embedding_buffer.append(embedding)
        if len(self.embedding_buffer) > self.window_size:
            self.embedding_buffer.pop(0)

        self.frame_count += 1

        # Return result when window is ready
        if len(self.embedding_buffer) >= self.window_size:
            # Use the most recent window_size embeddings
            window_embeddings = self.embedding_buffer[-self.window_size:]
            
            return {
                "embedding": embedding,  # Current embedding
                "window_embeddings": window_embeddings,  # Full window for phase prediction
                "timestamp": timestamp if timestamp is not None else self.frame_count / 30.0,  # Assume 30fps if not provided
                "frame_index": self.frame_count - 1,
                "keypoints": keypoints,
                "normalized_keypoints": normalized_keypoints,
            }
        else:
            # Window not ready yet, return None to indicate waiting
            return None

    def process_video_stream(
        self, video_capture, fps: Optional[float] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Process frames from a video capture object (e.g., cv2.VideoCapture) as a stream,
        yielding embeddings when window is ready.

        Args:
            video_capture: Video capture object (e.g., cv2.VideoCapture) or iterator of (frame, timestamp) tuples
            fps: Optional FPS for timestamp calculation (if not provided, will try to get from capture or default to 30.0)

        Yields:
            Dictionary with embedding and metadata (same format as process_frame)
        """
        import cv2

        # Handle cv2.VideoCapture object
        if hasattr(video_capture, 'read') and hasattr(video_capture, 'get'):
            # It's a cv2.VideoCapture object
            if fps is None:
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0  # Default fallback

            frame_index = 0
            self.reset()

            try:
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    timestamp = frame_index / fps
                    result = self.process_frame(frame, timestamp)

                    if result is not None:
                        yield result

                    frame_index += 1
            finally:
                # Don't release - caller is responsible for managing the capture
                pass
        else:
            # Assume it's an iterator of (frame, timestamp) tuples
            self.reset()
            for frame, timestamp in video_capture:
                result = self.process_frame(frame, timestamp)
                if result is not None:
                    yield result

    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Get current buffer status.

        Returns:
            Dictionary with buffer information
        """
        return {
            "buffer_size": len(self.embedding_buffer),
            "window_size": self.window_size,
            "ready": len(self.embedding_buffer) >= self.window_size,
            "frame_count": self.frame_count,
        }

