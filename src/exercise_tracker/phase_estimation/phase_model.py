"""Phase estimation model to map pose embeddings to phase."""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import time
import logging

logger = logging.getLogger(__name__)


class PhaseModel:
    """Model to predict phase (sin/cos representation) from pose embeddings."""

    def __init__(
        self,
        model_type: str = "mlp",
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        random_state: Optional[int] = None,
        window_size: int = 5,
    ):
        """
        Initialize phase model.

        Args:
            model_type: Type of model ('mlp' or 'rf')
            hidden_layer_sizes: Hidden layer sizes for MLP (ignored for RF)
            random_state: Random state for reproducibility
            window_size: Number of embeddings to use in sliding window (1 = single frame)
        """
        self.model_type = model_type
        self.random_state = random_state
        self.window_size = window_size
        self.is_fitted = False
        self.embed_dim = None  # Will be set during training
        self.embedding_buffer = []  # For live feed inference

        if model_type == "mlp":
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=1000,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
        elif model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _phase_to_sincos(self, phase: float) -> np.ndarray:
        """
        Convert phase to sin/cos representation.

        Args:
            phase: Phase value in [0, 1)

        Returns:
            Array [sin(2πθ), cos(2πθ)]
        """
        theta = 2 * np.pi * phase
        return np.array([np.sin(theta), np.cos(theta)])

    def _sincos_to_phase(self, sincos: np.ndarray) -> float:
        """
        Convert sin/cos representation back to phase.

        Args:
            sincos: Array [sin(2πθ), cos(2πθ)]

        Returns:
            Phase value in [0, 1)
        """
        phase = np.arctan2(sincos[0], sincos[1]) / (2 * np.pi)
        return phase % 1.0

    def _create_window(
        self, embeddings: List[Optional[np.ndarray]], idx: int
    ) -> Optional[np.ndarray]:
        """
        Create a sliding window of embeddings for a given index.

        Args:
            embeddings: List of embedding vectors
            idx: Current index

        Returns:
            Concatenated window vector or None if insufficient data
        """
        if self.window_size == 1:
            # Single frame mode (backward compatible)
            if idx < len(embeddings) and embeddings[idx] is not None:
                return embeddings[idx]
            return None

        if idx >= len(embeddings):
            return None

        # Collect window (looking backward from idx)
        window = []
        start_idx = max(0, idx - self.window_size + 1)

        # First, find the first valid embedding in the window range
        first_valid_idx = None
        for i in range(start_idx, idx + 1):
            if i < len(embeddings) and embeddings[i] is not None:
                first_valid_idx = i
                break

        if first_valid_idx is None:
            return None

        # Collect valid embeddings in window
        for i in range(start_idx, idx + 1):
            if i < len(embeddings):
                if embeddings[i] is not None:
                    window.append(embeddings[i])
                else:
                    # If we hit a None, pad with the last valid embedding
                    if len(window) > 0:
                        window.append(window[-1])
                    elif i > first_valid_idx:
                        # We're past the first valid, use it
                        window.append(embeddings[first_valid_idx])
                    else:
                        # Can't create window
                        return None

        # Pad with first valid embedding if needed (for early frames)
        if len(window) < self.window_size:
            if len(window) == 0:
                return None
            # Pad with the first embedding in window
            first_emb = window[0]
            padding_needed = self.window_size - len(window)
            window = [first_emb] * padding_needed + window

        # Ensure we have exactly window_size embeddings
        if len(window) > self.window_size:
            window = window[-self.window_size:]

        # Concatenate window
        window_flat = np.concatenate(window)
        return window_flat

    def fit(
        self,
        embeddings: List[np.ndarray],
        phases: List[float],
    ) -> None:
        """
        Train the phase model with sliding window support.

        Args:
            embeddings: List of embedding vectors (sequence order matters for window_size > 1)
            phases: List of phase values in [0, 1)
        """
        if len(embeddings) == 0:
            raise ValueError("No training data provided")

        if len(embeddings) != len(phases):
            raise ValueError("embeddings and phases must have same length")

        # Create windows from sequence
        windows = []
        window_phases = []

        for i in range(len(embeddings)):
            window = self._create_window(embeddings, i)
            if window is not None:
                windows.append(window)
                window_phases.append(phases[i])

        if len(windows) == 0:
            raise ValueError("No valid windows created from training data")

        # Store embedding dimension for validation
        if self.window_size == 1:
            self.embed_dim = windows[0].shape[0]
        else:
            # Window size * embed_dim
            self.embed_dim = windows[0].shape[0] // self.window_size

        X = np.array(windows)
        phases_array = np.array(window_phases)

        # Convert phases to sin/cos representation
        y = np.array([self._phase_to_sincos(phase) for phase in phases_array])

        # Train model
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_phase(
        self,
        embedding: Optional[np.ndarray],
        update_buffer: bool = True,
        log_timing: bool = False,
    ) -> Optional[float]:
        """
        Predict phase from embedding (with sliding window support for live feed).

        Args:
            embedding: Embedding vector (current frame)
            update_buffer: If True, update internal buffer (for live feed). 
                          If False, use provided embedding only (for batch processing)
            log_timing: If True, log timing information

        Returns:
            Predicted phase in [0, 1), or None if input is None
        """
        if embedding is None:
            return None

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = time.perf_counter()

        if self.window_size == 1:
            # Single frame mode (backward compatible)
            window_vector = embedding
        else:
            # Sliding window mode
            if update_buffer:
                # For live feed: maintain buffer
                self.embedding_buffer.append(embedding)
                if len(self.embedding_buffer) > self.window_size:
                    self.embedding_buffer.pop(0)

                # Create window from buffer
                if len(self.embedding_buffer) < self.window_size:
                    # Pad with first embedding if buffer not full
                    padding_needed = self.window_size - len(self.embedding_buffer)
                    window = [self.embedding_buffer[0]] * padding_needed + self.embedding_buffer
                else:
                    window = self.embedding_buffer[-self.window_size:]
                
                # Concatenate window
                window_vector = np.concatenate(window)
            else:
                # For batch processing: single frame mode (predict_phases handles windows)
                window_vector = embedding

        # Predict sin/cos
        sincos = self.model.predict(window_vector.reshape(1, -1))[0]

        # Convert to phase
        phase = self._sincos_to_phase(sincos)

        elapsed = time.perf_counter() - start_time
        if log_timing:
            logger.debug(f"Phase prediction time: {elapsed*1000:.3f} ms")

        return phase

    def reset_buffer(self) -> None:
        """Reset the embedding buffer (useful for new video/sequence)."""
        self.embedding_buffer = []

    def predict_phases(
        self, embeddings: List[Optional[np.ndarray]], log_timing: bool = False
    ) -> List[Optional[float]]:
        """
        Predict phases for a sequence of embeddings (with sliding window support).

        Args:
            embeddings: List of embedding vectors (sequence order matters)
            log_timing: If True, log timing information

        Returns:
            List of predicted phases
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if log_timing:
            start_time = time.perf_counter()

        phases = []
        for i in range(len(embeddings)):
            if self.window_size == 1:
                # Single frame mode
                if embeddings[i] is not None:
                    sincos = self.model.predict(embeddings[i].reshape(1, -1))[0]
                    phase = self._sincos_to_phase(sincos)
                else:
                    phase = None
            else:
                # Sliding window mode: create window from sequence
                window_vector = self._create_window(embeddings, i)
                if window_vector is not None:
                    sincos = self.model.predict(window_vector.reshape(1, -1))[0]
                    phase = self._sincos_to_phase(sincos)
                else:
                    phase = None
            phases.append(phase)

        if log_timing:
            total_time = time.perf_counter() - start_time
            valid_count = sum(1 for p in phases if p is not None)
            if valid_count > 0:
                avg_time = total_time / valid_count * 1000  # ms per frame
                logger.info(
                    f"Phase prediction sequence: {len(embeddings)} frames, "
                    f"{valid_count} valid, {total_time*1000:.1f} ms total, "
                    f"{avg_time:.3f} ms/frame avg"
                )

        return phases

    def save(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "random_state": self.random_state,
            "window_size": self.window_size,
            "embed_dim": self.embed_dim,
        }

        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PhaseModel":
        """
        Load a model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            PhaseModel instance with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        phase_model = cls(
            model_type=model_data["model_type"],
            random_state=model_data["random_state"],
            window_size=model_data.get("window_size", 1),  # Backward compatibility
        )
        phase_model.model = model_data["model"]
        phase_model.is_fitted = True
        phase_model.embed_dim = model_data.get("embed_dim", None)

        return phase_model

