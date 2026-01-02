"""Video processing pipeline with pose extraction and caching."""

import os
import json
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from ..pose_extraction import MediaPipeExtractor, BasePoseExtractor
from ..pose_processing import PoseNormalizer, PoseEmbedder


class VideoProcessor:
    """Process videos to extract and cache pose sequences."""

    def __init__(
        self,
        pose_extractor: Optional[BasePoseExtractor] = None,
        normalizer: Optional[PoseNormalizer] = None,
        embedder: Optional[PoseEmbedder] = None,
    ):
        """
        Initialize video processor.

        Args:
            pose_extractor: Pose extractor instance (default: MediaPipeExtractor)
            normalizer: Pose normalizer instance (default: PoseNormalizer)
            embedder: Pose embedder instance (optional, can be fitted later)
        """
        self.pose_extractor = pose_extractor or MediaPipeExtractor()
        self.normalizer = normalizer or PoseNormalizer()
        self.embedder = embedder

    def process_video(
        self,
        video_path: str,
        cache_dir: Optional[str] = None,
        force_reprocess: bool = False,
    ) -> Dict[str, any]:
        """
        Process a video and extract pose data.

        Args:
            video_path: Path to video file
            cache_dir: Directory to cache processed data
            force_reprocess: If True, reprocess even if cached

        Returns:
            Dictionary containing:
            - 'keypoints': List of raw keypoint arrays
            - 'normalized_keypoints': List of normalized keypoint arrays
            - 'pose_vectors': List of flattened pose vectors
            - 'embeddings': List of embedding vectors (if embedder is set)
            - 'timestamps': List of timestamps
            - 'metadata': Video metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Check cache
        if cache_dir and not force_reprocess:
            cached_data = self._load_from_cache(video_path, cache_dir)
            if cached_data is not None:
                return cached_data

        # Extract poses from video
        pose_sequence, timestamps = self.pose_extractor.extract_poses_from_video(
            str(video_path)
        )

        # Normalize poses
        normalized_sequence = self.normalizer.normalize_sequence(pose_sequence)

        # Extract pose vectors
        pose_vectors = [
            self.normalizer.get_pose_vector(pose) for pose in normalized_sequence
        ]

        # Embed if embedder is available and fitted
        embeddings = None
        if self.embedder and self.embedder.is_fitted:
            embeddings = self.embedder.embed_sequence(pose_vectors)

        # Prepare result
        result = {
            "keypoints": pose_sequence,
            "normalized_keypoints": normalized_sequence,
            "pose_vectors": pose_vectors,
            "embeddings": embeddings,
            "timestamps": timestamps,
            "metadata": {
                "video_path": str(video_path),
                "num_frames": len(pose_sequence),
                "duration": timestamps[-1] if timestamps else 0.0,
                "fps": len(timestamps) / timestamps[-1] if timestamps and timestamps[-1] > 0 else 0.0,
            },
        }

        # Cache result
        if cache_dir:
            self._save_to_cache(result, video_path, cache_dir)

        return result

    def _get_cache_path(self, video_path: Path, cache_dir: str) -> Path:
        """Get cache file path for a video."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use video filename (without extension) as cache key
        video_name = video_path.stem
        cache_file = cache_dir / f"{video_name}.npz"
        return cache_file

    def _save_to_cache(self, data: Dict, video_path: Path, cache_dir: str) -> None:
        """Save processed data to cache."""
        cache_file = self._get_cache_path(video_path, cache_dir)

        # Convert lists to arrays for numpy save
        save_dict = {
            "timestamps": np.array(data["timestamps"]),
            "metadata": json.dumps(data["metadata"]),
        }

        # Save keypoints (handle None values)
        keypoints_list = data["keypoints"]
        normalized_list = data["normalized_keypoints"]
        pose_vectors_list = data["pose_vectors"]

        # Find max shape for padding
        if keypoints_list and keypoints_list[0] is not None:
            max_keypoints = max(
                len(kp) if kp is not None else 0 for kp in keypoints_list
            )
            kp_shape = keypoints_list[0].shape[1] if keypoints_list[0] is not None else 3

            # Pad and save keypoints
            keypoints_array = np.full(
                (len(keypoints_list), max_keypoints, kp_shape), np.nan
            )
            for i, kp in enumerate(keypoints_list):
                if kp is not None:
                    keypoints_array[i, : len(kp)] = kp

            save_dict["keypoints"] = keypoints_array

            # Save normalized keypoints
            normalized_array = np.full(
                (len(normalized_list), max_keypoints, kp_shape), np.nan
            )
            for i, nkp in enumerate(normalized_list):
                if nkp is not None:
                    normalized_array[i, : len(nkp)] = nkp

            save_dict["normalized_keypoints"] = normalized_array

        # Save pose vectors
        if pose_vectors_list and pose_vectors_list[0] is not None:
            max_dim = max(
                len(pv) if pv is not None else 0 for pv in pose_vectors_list
            )
            pose_vectors_array = np.full((len(pose_vectors_list), max_dim), np.nan)
            for i, pv in enumerate(pose_vectors_list):
                if pv is not None:
                    pose_vectors_array[i, : len(pv)] = pv

            save_dict["pose_vectors"] = pose_vectors_array

        # Save embeddings if available
        if data["embeddings"] is not None:
            embeddings_list = data["embeddings"]
            if embeddings_list and embeddings_list[0] is not None:
                embed_dim = len(embeddings_list[0]) if embeddings_list[0] is not None else 0
                embeddings_array = np.full((len(embeddings_list), embed_dim), np.nan)
                for i, emb in enumerate(embeddings_list):
                    if emb is not None:
                        embeddings_array[i] = emb

                save_dict["embeddings"] = embeddings_array

        np.savez_compressed(cache_file, **save_dict)

    def _load_from_cache(self, video_path: Path, cache_dir: str) -> Optional[Dict]:
        """Load processed data from cache."""
        cache_file = self._get_cache_path(video_path, cache_dir)

        if not cache_file.exists():
            return None

        try:
            loaded = np.load(cache_file, allow_pickle=True)

            # Reconstruct data structure
            timestamps = loaded["timestamps"].tolist()
            metadata = json.loads(loaded["metadata"].item())

            # Reconstruct keypoints (handle NaN padding)
            keypoints = []
            if "keypoints" in loaded:
                keypoints_array = loaded["keypoints"]
                for i in range(len(keypoints_array)):
                    kp = keypoints_array[i]
                    # Remove NaN rows
                    valid_mask = ~np.isnan(kp).all(axis=1)
                    if np.any(valid_mask):
                        keypoints.append(kp[valid_mask])
                    else:
                        keypoints.append(None)

            normalized_keypoints = []
            if "normalized_keypoints" in loaded:
                normalized_array = loaded["normalized_keypoints"]
                for i in range(len(normalized_array)):
                    nkp = normalized_array[i]
                    valid_mask = ~np.isnan(nkp).all(axis=1)
                    if np.any(valid_mask):
                        normalized_keypoints.append(nkp[valid_mask])
                    else:
                        normalized_keypoints.append(None)

            pose_vectors = []
            if "pose_vectors" in loaded:
                pose_vectors_array = loaded["pose_vectors"]
                for i in range(len(pose_vectors_array)):
                    pv = pose_vectors_array[i]
                    valid_mask = ~np.isnan(pv)
                    if np.any(valid_mask):
                        pose_vectors.append(pv[valid_mask])
                    else:
                        pose_vectors.append(None)

            embeddings = None
            if "embeddings" in loaded:
                embeddings_array = loaded["embeddings"]
                embeddings = []
                for i in range(len(embeddings_array)):
                    emb = embeddings_array[i]
                    if not np.isnan(emb).all():
                        embeddings.append(emb)
                    else:
                        embeddings.append(None)

            return {
                "keypoints": keypoints,
                "normalized_keypoints": normalized_keypoints,
                "pose_vectors": pose_vectors,
                "embeddings": embeddings,
                "timestamps": timestamps,
                "metadata": metadata,
            }
        except Exception as e:
            # If cache is corrupted, return None to force reprocessing
            print(f"Warning: Failed to load cache: {e}")
            return None

