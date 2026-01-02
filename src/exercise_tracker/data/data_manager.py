"""Data manager for organizing reference video datasets."""

import os
from pathlib import Path
from typing import List, Dict, Optional
import json

from .video_processor import VideoProcessor
from ..pose_processing import PoseEmbedder


class DataManager:
    """Manage reference video datasets and processed pose data."""

    def __init__(
        self,
        data_root: str = "data",
        video_processor: Optional[VideoProcessor] = None,
    ):
        """
        Initialize data manager.

        Args:
            data_root: Root directory for data (raw and processed)
            video_processor: Video processor instance (default: VideoProcessor)
        """
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.video_processor = video_processor or VideoProcessor()

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_exercise_dir(self, exercise_name: str, raw: bool = True) -> Path:
        """
        Get directory path for an exercise.

        Args:
            exercise_name: Name of the exercise
            raw: If True, return raw video directory; else processed data directory

        Returns:
            Path to exercise directory
        """
        base_dir = self.raw_dir if raw else self.processed_dir
        return base_dir / exercise_name

    def list_exercises(self) -> List[str]:
        """
        List all exercises in the dataset.

        Returns:
            List of exercise names
        """
        if not self.raw_dir.exists():
            return []

        exercises = [
            d.name
            for d in self.raw_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(exercises)

    def list_videos(self, exercise_name: str) -> List[str]:
        """
        List video files for an exercise.

        Args:
            exercise_name: Name of the exercise

        Returns:
            List of video file paths
        """
        exercise_dir = self.get_exercise_dir(exercise_name, raw=True)

        if not exercise_dir.exists():
            return []

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = [
            str(f)
            for f in exercise_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]

        return sorted(videos)

    def process_exercise_videos(
        self,
        exercise_name: str,
        force_reprocess: bool = False,
        embedder: Optional[PoseEmbedder] = None,
    ) -> Dict[str, any]:
        """
        Process all videos for an exercise.

        Args:
            exercise_name: Name of the exercise
            force_reprocess: If True, reprocess even if cached
            embedder: Optional embedder for generating embeddings

        Returns:
            Dictionary mapping video paths to processed data
        """
        videos = self.list_videos(exercise_name)

        if len(videos) == 0:
            raise ValueError(f"No videos found for exercise: {exercise_name}")

        # Set embedder if provided
        if embedder:
            self.video_processor.embedder = embedder

        processed_data = {}

        for video_path in videos:
            print(f"Processing {video_path}...")
            cache_dir = str(self.get_exercise_dir(exercise_name, raw=False))
            data = self.video_processor.process_video(
                video_path, cache_dir=cache_dir, force_reprocess=force_reprocess
            )
            processed_data[video_path] = data

        return processed_data

    def save_exercise_metadata(
        self, exercise_name: str, metadata: Dict[str, any]
    ) -> None:
        """
        Save metadata for an exercise.

        Args:
            exercise_name: Name of the exercise
            metadata: Metadata dictionary to save
        """
        exercise_dir = self.get_exercise_dir(exercise_name, raw=False)
        exercise_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = exercise_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_exercise_metadata(self, exercise_name: str) -> Optional[Dict[str, any]]:
        """
        Load metadata for an exercise.

        Args:
            exercise_name: Name of the exercise

        Returns:
            Metadata dictionary or None if not found
        """
        exercise_dir = self.get_exercise_dir(exercise_name, raw=False)
        metadata_file = exercise_dir / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, "r") as f:
            return json.load(f)

