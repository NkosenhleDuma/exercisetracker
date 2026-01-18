"""Frame-based phase estimation model to map pose embeddings to phase."""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import time
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available. MLP model will not work.")

from .base_phase_model import BasePhaseModel

logger = logging.getLogger(__name__)


class PhaseMLP(nn.Module):
    """PyTorch MLP for phase prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_layer_sizes: Sizes of hidden layers
            dropout: Dropout rate (0.0 = no dropout)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size
        
        # Output layer (sin/cos representation)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class FrameBasedPhaseModel(BasePhaseModel):
    """Model to predict phase (sin/cos representation) from pose embeddings."""

    def __init__(
        self,
        model_type: str = "mlp",
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        random_state: Optional[int] = None,
        window_size: int = 5,
        phase_buffer_size: int = 10,
    ):
        """
        Initialize phase model.

        Args:
            model_type: Type of model ('mlp' or 'rf')
            hidden_layer_sizes: Hidden layer sizes for MLP (ignored for RF)
            random_state: Random state for reproducibility
            window_size: Number of embeddings to use in sliding window (1 = single frame)
            phase_buffer_size: Number of previous phases to include as context (default: 1)
        """
        self.model_type = model_type
        self.random_state = random_state
        self.window_size = window_size
        self.phase_buffer_size = phase_buffer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.is_fitted = False
        self.embed_dim = None  # Will be set during training
        self.embedding_buffer = []  # For live feed inference
        self.previous_phase_buffer = []  # For tracking previous phases (phase-aware model)

        if model_type == "mlp":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for MLP model")
            # Model will be created during fit() when we know input_dim
            self.model = None
        elif model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Set random seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            if TORCH_AVAILABLE:
                torch.manual_seed(random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(random_state)

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

    def _create_window_with_previous_phase(
        self,
        embeddings: List[Optional[np.ndarray]],
        idx: int,
        previous_phases: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Create a sliding window of embeddings with previous phase buffer included.

        Args:
            embeddings: List of embedding vectors
            idx: Current index
            previous_phases: List of previous phase values in [0, 1) (or None to pad with 0.0)

        Returns:
            Concatenated window vector with previous phase buffer sin/cos appended, or None if insufficient data
        """
        # Create base window
        window_vector = self._create_window(embeddings, idx)
        if window_vector is None:
            return None

        # Get previous phase buffer sin/cos representation
        if previous_phases is None or len(previous_phases) == 0:
            # For first window or empty buffer, pad with 0.0
            prev_phases = [0.0] * self.phase_buffer_size
        else:
            # Use provided phases, pad if needed
            prev_phases = list(previous_phases)
            if len(prev_phases) < self.phase_buffer_size:
                # Pad with the last phase (or 0.0 if empty)
                padding_value = prev_phases[-1] if len(prev_phases) > 0 else 0.0
                prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases
            elif len(prev_phases) > self.phase_buffer_size:
                # Take only the most recent phases
                prev_phases = prev_phases[-self.phase_buffer_size:]

        # Convert each phase to sin/cos and concatenate
        prev_phases_sincos = []
        for phase in prev_phases:
            prev_phases_sincos.extend(self._phase_to_sincos(phase))

        # Concatenate previous phase buffer sin/cos to window
        window_with_phase = np.concatenate([window_vector, np.array(prev_phases_sincos)])
        return window_with_phase

    def fit(
        self,
        embeddings: List[np.ndarray],
        phases: List[float],
        timestamps: Optional[List[float]] = None,
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

        # Create windows from sequence with previous phase buffer
        windows = []
        window_phases = []
        phase_buffer = []  # Track previous phases for buffer

        for i in range(len(embeddings)):
            # Get previous phase buffer (use current phase for first window if buffer is empty)
            if len(phase_buffer) == 0:
                # For first window, pad buffer with current phase
                prev_phases = [phases[i]] * self.phase_buffer_size
            else:
                # Use the phase buffer (most recent phases)
                prev_phases = list(phase_buffer[-self.phase_buffer_size:])
                # Pad if needed (shouldn't happen, but safety check)
                if len(prev_phases) < self.phase_buffer_size:
                    padding_value = prev_phases[-1] if len(prev_phases) > 0 else phases[i]
                    prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases

            window = self._create_window_with_previous_phase(embeddings, i, prev_phases)
            if window is not None:
                windows.append(window)
                window_phases.append(phases[i])
                # Update phase buffer with current phase
                phase_buffer.append(phases[i])
                # Keep buffer size manageable (only need last phase_buffer_size)
                if len(phase_buffer) > self.phase_buffer_size:
                    phase_buffer.pop(0)

        if len(windows) == 0:
            raise ValueError("No valid windows created from training data")

        # Store embedding dimension for validation
        # Account for previous phase buffer sin/cos (phase_buffer_size * 2 features)
        phase_features = self.phase_buffer_size * 2
        if self.window_size == 1:
            # Single frame: embed_dim + phase_features
            self.embed_dim = windows[0].shape[0] - phase_features
        else:
            # Window size * embed_dim + phase_features
            self.embed_dim = (windows[0].shape[0] - phase_features) // self.window_size

        X = np.array(windows)
        phases_array = np.array(window_phases)

        # Convert phases to sin/cos representation
        y = np.array([self._phase_to_sincos(phase) for phase in phases_array])

        # Train model
        if self.model_type == "mlp":
            # Create PyTorch MLP model
            input_dim = X.shape[1]
            self.model = PhaseMLP(
                input_dim=input_dim,
                hidden_layer_sizes=self.hidden_layer_sizes,
                dropout=0.0,  # No dropout for frame-based model
            )
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            val_split = 0.1
            val_size = int(len(X) * val_split)
            
            if val_size > 0:
                X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
                y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]
            else:
                X_train, X_val = X_tensor, X_tensor
                y_train, y_val = y_tensor, y_tensor
            
            self.model.train()
            for epoch in range(1000):  # Max epochs
                optimizer.zero_grad()
                y_pred = self.model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                
                # Validation
                if val_size > 0:
                    self.model.eval()
                    with torch.no_grad():
                        y_val_pred = self.model(X_val)
                        val_loss = criterion(y_val_pred, y_val).item()
                    self.model.train()
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            # Restore best model
                            self.model.load_state_dict(best_state)
                            break
                else:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
            
            self.model.eval()
        else:
            # RandomForest (sklearn)
            self.model.fit(X, y)
        
        self.is_fitted = True

    def predict_phase(
        self,
        embedding: Optional[np.ndarray],
        timestamp: Optional[float] = None,
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

        # Get previous phase buffer for phase-aware prediction
        if len(self.previous_phase_buffer) == 0:
            # First prediction: pad with 0.0
            prev_phases = [0.0] * self.phase_buffer_size
        else:
            # Use the phase buffer (most recent phases)
            prev_phases = list(self.previous_phase_buffer[-self.phase_buffer_size:])
            # Pad if needed
            if len(prev_phases) < self.phase_buffer_size:
                padding_value = prev_phases[-1] if len(prev_phases) > 0 else 0.0
                prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases

        if self.window_size == 1:
            # Single frame mode (backward compatible)
            window_vector = embedding
            # Add previous phase buffer sin/cos
            prev_phases_sincos = []
            for phase in prev_phases:
                prev_phases_sincos.extend(self._phase_to_sincos(phase))
            window_vector = np.concatenate([window_vector, np.array(prev_phases_sincos)])
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
                # Add previous phase buffer sin/cos
                prev_phases_sincos = []
                for phase in prev_phases:
                    prev_phases_sincos.extend(self._phase_to_sincos(phase))
                window_vector = np.concatenate([window_vector, np.array(prev_phases_sincos)])
            else:
                # For batch processing: single frame mode (predict_phases handles windows)
                window_vector = embedding
                # Add previous phase buffer sin/cos
                prev_phases_sincos = []
                for phase in prev_phases:
                    prev_phases_sincos.extend(self._phase_to_sincos(phase))
                window_vector = np.concatenate([window_vector, np.array(prev_phases_sincos)])

        # Predict sin/cos
        if self.model_type == "mlp":
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(window_vector.reshape(1, -1))
                output = self.model(input_tensor)
                sincos = output[0].cpu().numpy()
        else:
            sincos = self.model.predict(window_vector.reshape(1, -1))[0]

        # Convert to phase
        phase = self._sincos_to_phase(sincos)

        # Update previous phase buffer if updating buffer
        if update_buffer:
            self.previous_phase_buffer.append(phase)
            # Keep buffer size manageable (only need last phase_buffer_size)
            if len(self.previous_phase_buffer) > self.phase_buffer_size:
                self.previous_phase_buffer.pop(0)

        elapsed = time.perf_counter() - start_time
        if log_timing:
            logger.debug(f"Phase prediction time: {elapsed*1000:.3f} ms")

        return phase

    def reset_buffer(self) -> None:
        """Reset the embedding buffer and previous phase buffer (useful for new video/sequence)."""
        self.embedding_buffer = []
        self.previous_phase_buffer = []

    def predict_phases(
        self,
        embeddings: List[Optional[np.ndarray]],
        timestamps: Optional[List[float]] = None,
        log_timing: bool = False,
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
        phase_buffer = []  # Track previous phases for phase-aware prediction
        
        for i in range(len(embeddings)):
            if self.window_size == 1:
                # Single frame mode
                if embeddings[i] is not None:
                    # Get previous phase buffer (pad with 0.0 if empty)
                    if len(phase_buffer) == 0:
                        prev_phases = [0.0] * self.phase_buffer_size
                    else:
                        prev_phases = list(phase_buffer[-self.phase_buffer_size:])
                        # Pad if needed
                        if len(prev_phases) < self.phase_buffer_size:
                            padding_value = prev_phases[-1] if len(prev_phases) > 0 else 0.0
                            prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases
                    
                    # Convert phase buffer to sin/cos
                    prev_phases_sincos = []
                    for phase in prev_phases:
                        prev_phases_sincos.extend(self._phase_to_sincos(phase))
                    
                    # Create input with previous phase buffer
                    window_vector = np.concatenate([embeddings[i], np.array(prev_phases_sincos)])
                    if self.model_type == "mlp":
                        self.model.eval()
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(window_vector.reshape(1, -1))
                            output = self.model(input_tensor)
                            sincos = output[0].cpu().numpy()
                    else:
                        sincos = self.model.predict(window_vector.reshape(1, -1))[0]
                    phase = self._sincos_to_phase(sincos)
                    # Update phase buffer
                    phase_buffer.append(phase)
                    # Keep buffer size manageable
                    if len(phase_buffer) > self.phase_buffer_size:
                        phase_buffer.pop(0)
                else:
                    phase = None
            else:
                # Sliding window mode: create window from sequence with previous phase buffer
                if embeddings[i] is not None:
                    # Get previous phase buffer (pad with 0.0 if empty)
                    if len(phase_buffer) == 0:
                        prev_phases = [0.0] * self.phase_buffer_size
                    else:
                        prev_phases = list(phase_buffer[-self.phase_buffer_size:])
                        # Pad if needed
                        if len(prev_phases) < self.phase_buffer_size:
                            padding_value = prev_phases[-1] if len(prev_phases) > 0 else 0.0
                            prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases
                    
                    window_vector = self._create_window_with_previous_phase(embeddings, i, prev_phases)
                    if window_vector is not None:
                        if self.model_type == "mlp":
                            self.model.eval()
                            with torch.no_grad():
                                input_tensor = torch.FloatTensor(window_vector.reshape(1, -1))
                                output = self.model(input_tensor)
                                sincos = output[0].cpu().numpy()
                        else:
                            sincos = self.model.predict(window_vector.reshape(1, -1))[0]
                        phase = self._sincos_to_phase(sincos)
                        # Update phase buffer
                        phase_buffer.append(phase)
                        # Keep buffer size manageable
                        if len(phase_buffer) > self.phase_buffer_size:
                            phase_buffer.pop(0)
                    else:
                        phase = None
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

        if self.model_type == "mlp":
            # Save PyTorch model state dict separately
            base_name = os.path.splitext(filepath)[0]
            torch.save(self.model.state_dict(), base_name + "_state_dict.pt")
            model_data = {
                "model_type": self.model_type,
                "random_state": self.random_state,
                "window_size": self.window_size,
                "phase_buffer_size": self.phase_buffer_size,
                "embed_dim": self.embed_dim,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "input_dim": None,  # Will be calculated from embed_dim and window_size
            }
        else:
            # RandomForest - save model directly
            model_data = {
                "model": self.model,
                "model_type": self.model_type,
                "random_state": self.random_state,
                "window_size": self.window_size,
                "phase_buffer_size": self.phase_buffer_size,
                "embed_dim": self.embed_dim,
            }

        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FrameBasedPhaseModel":
        """
        Load a model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            FrameBasedPhaseModel instance with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        phase_model = cls(
            model_type=model_data["model_type"],
            random_state=model_data["random_state"],
            window_size=model_data.get("window_size", 1),  # Backward compatibility
            phase_buffer_size=model_data.get("phase_buffer_size", 1),  # Backward compatibility
            hidden_layer_sizes=model_data.get("hidden_layer_sizes", (64, 32)),  # Backward compatibility
        )
        
        if model_data["model_type"] == "mlp":
            # Load PyTorch model
            base_name = os.path.splitext(filepath)[0]
            state_dict_path = base_name + "_state_dict.pt"
            
            if not os.path.exists(state_dict_path):
                # Backward compatibility: try loading old format
                if "model" in model_data:
                    phase_model.model = model_data["model"]
                else:
                    raise FileNotFoundError(f"Model state dict not found: {state_dict_path}")
            else:
                # Calculate input dimension
                embed_dim = model_data.get("embed_dim")
                window_size = model_data.get("window_size", 1)
                phase_buffer_size = model_data.get("phase_buffer_size", 1)
                input_dim = (embed_dim * window_size) + (phase_buffer_size * 2)
                
                # Create model and load state dict
                phase_model.model = PhaseMLP(
                    input_dim=input_dim,
                    hidden_layer_sizes=model_data.get("hidden_layer_sizes", (64, 32)),
                    dropout=0.0,
                )
                phase_model.model.load_state_dict(torch.load(state_dict_path, map_location='cpu', weights_only=False))
                phase_model.model.eval()
        else:
            # RandomForest - load directly
            phase_model.model = model_data["model"]
        
        phase_model.is_fitted = True
        phase_model.embed_dim = model_data.get("embed_dim", None)

        return phase_model

