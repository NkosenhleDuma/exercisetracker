"""Base interface for phase estimation models."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BasePhaseModel(ABC):
    """Base class for phase estimation models."""

    @abstractmethod
    def fit(
        self,
        embeddings: List[np.ndarray],
        phases: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """
        Train the phase model.

        Args:
            embeddings: List of embedding vectors
            phases: List of phase values in [0, 1)
            timestamps: Optional list of timestamps (required for time-based models)
        """
        pass

    @abstractmethod
    def predict_phase(
        self,
        embedding: Optional[np.ndarray],
        timestamp: Optional[float] = None,
        update_buffer: bool = True,
        log_timing: bool = False,
    ) -> Optional[float]:
        """
        Predict phase from embedding.

        Args:
            embedding: Embedding vector (current frame)
            timestamp: Optional timestamp (required for time-based models)
            update_buffer: If True, update internal buffer
            log_timing: If True, log timing information

        Returns:
            Predicted phase in [0, 1), or None if input is None
        """
        pass

    @abstractmethod
    def predict_phases(
        self,
        embeddings: List[Optional[np.ndarray]],
        timestamps: Optional[List[float]] = None,
        log_timing: bool = False,
    ) -> List[Optional[float]]:
        """
        Predict phases for a sequence of embeddings.

        Args:
            embeddings: List of embedding vectors
            timestamps: Optional list of timestamps (required for time-based models)
            log_timing: If True, log timing information

        Returns:
            List of predicted phases
        """
        pass

    @abstractmethod
    def reset_buffer(self) -> None:
        """Reset internal buffers (useful for new video/sequence)."""
        pass

    # Note: is_fitted is implemented as an attribute in subclasses (self.is_fitted = False/True)
    # It's not abstract to allow attribute-based implementation
    # Subclasses should define: self.is_fitted = False in __init__

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BasePhaseModel":
        """Load a model from disk."""
        pass
