"""Pose embedding to compact latent space using PCA."""

import numpy as np
from typing import List, Optional
from sklearn.decomposition import PCA
import joblib
import os
import time
import logging

logger = logging.getLogger(__name__)


class PoseEmbedder:
    """Embed poses to compact latent space using PCA."""

    def __init__(self, n_components: int = 32, pca_model: Optional[PCA] = None):
        """
        Initialize pose embedder.

        Args:
            n_components: Number of PCA components for embedding dimension
            pca_model: Pre-trained PCA model (optional)
        """
        self.n_components = n_components
        self.pca = pca_model if pca_model is not None else PCA(n_components=n_components)
        self.is_fitted = pca_model is not None

    def fit(self, pose_vectors: List[np.ndarray]) -> None:
        """
        Fit PCA model on pose vectors.

        Args:
            pose_vectors: List of pose vectors (flattened keypoint positions)
        """
        # Filter out None values
        valid_vectors = [v for v in pose_vectors if v is not None]

        if len(valid_vectors) == 0:
            raise ValueError("No valid pose vectors provided for fitting")

        # Stack into matrix
        X = np.array(valid_vectors)
        self.pca.fit(X)
        self.is_fitted = True

    def embed(self, pose_vector: Optional[np.ndarray], log_timing: bool = False) -> Optional[np.ndarray]:
        """
        Embed a single pose vector to latent space.

        Args:
            pose_vector: Flattened pose vector
            log_timing: If True, log timing information

        Returns:
            Embedding vector of shape (n_components,), or None if input is None
        """
        if pose_vector is None:
            return None

        if not self.is_fitted:
            raise ValueError("PCA model must be fitted before embedding")

        start_time = time.perf_counter()
        
        # Reshape to 2D for PCA
        pose_2d = pose_vector.reshape(1, -1)
        
        # Validate input dimension matches PCA model
        expected_features = self.pca.n_features_in_
        actual_features = pose_2d.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"Input dimension mismatch: PCA model expects {expected_features} features "
                f"(trained on {'3D' if expected_features == 99 else '2D'} data), "
                f"but got {actual_features} features ({'3D' if actual_features == 99 else '2D'} data). "
                f"Make sure to use the same use_3d setting during ingestion/training and analysis."
            )
        
        embedding = self.pca.transform(pose_2d)

        elapsed = time.perf_counter() - start_time
        if log_timing:
            logger.debug(f"Embedding time: {elapsed*1000:.3f} ms")

        return embedding[0]

    def embed_sequence(
        self, pose_vectors: List[Optional[np.ndarray]], log_timing: bool = False
    ) -> List[Optional[np.ndarray]]:
        """
        Embed a sequence of pose vectors.

        Args:
            pose_vectors: List of pose vectors
            log_timing: If True, log timing information

        Returns:
            List of embedding vectors
        """
        if log_timing:
            start_time = time.perf_counter()
            embeddings = [self.embed(v, log_timing=False) for v in pose_vectors]
            total_time = time.perf_counter() - start_time
            valid_count = sum(1 for e in embeddings if e is not None)
            if valid_count > 0:
                avg_time = total_time / valid_count * 1000  # ms per frame
                logger.info(
                    f"Embedding sequence: {len(pose_vectors)} frames, "
                    f"{valid_count} valid, {total_time*1000:.1f} ms total, "
                    f"{avg_time:.3f} ms/frame avg"
                )
            return embeddings
        else:
            return [self.embed(v, log_timing=False) for v in pose_vectors]

    def save(self, filepath: str) -> None:
        """
        Save the PCA model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        joblib.dump(self.pca, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PoseEmbedder":
        """
        Load a PCA model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            PoseEmbedder instance with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        pca = joblib.load(filepath)
        embedder = cls(pca_model=pca)
        return embedder

