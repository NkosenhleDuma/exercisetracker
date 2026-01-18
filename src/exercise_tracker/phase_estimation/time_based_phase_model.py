"""Time-based phase estimation model using LSTM with PyTorch Lightning (handles variable frame rates)."""

import numpy as np
from typing import List, Tuple, Optional
import os
import time
import logging
import bisect

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch/PyTorch Lightning not available. LSTM model will not work.")

from .base_phase_model import BasePhaseModel

logger = logging.getLogger(__name__)


class PhaseLSTMModule(pl.LightningModule):
    """PyTorch Lightning module for LSTM-based phase prediction."""
    
    def __init__(
        self,
        input_dim: int,
        lstm_units: int = 64,
        lstm_layers: int = 1,
        dense_units: Tuple[int, ...] = (32,),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """
        Initialize LSTM module.
        
        Args:
            input_dim: Input feature dimension (embed_dim + phase_buffer_size * 2)
            lstm_units: Number of units in LSTM layer(s)
            lstm_layers: Number of LSTM layers (stacked)
            dense_units: Number of units in dense layers after LSTM
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        prev_dim = lstm_units
        for units in dense_units:
            self.dense_layers.append(nn.Linear(prev_dim, units))
            prev_dim = units
        
        # Output layer (sin/cos representation)
        self.output_layer = nn.Linear(prev_dim, 2)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, 2) - sin/cos representation
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_units)
        
        # Take the last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_units)
        
        # Dense layers
        x = last_hidden
        for dense in self.dense_layers:
            x = F.relu(dense(x))
            x = self.dropout_layer(x)
        
        # Output layer
        output = self.output_layer(x)  # (batch, 2)
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class PhaseDataset(Dataset):
    """Dataset for phase prediction sequences."""
    
    def __init__(self, sequences: List[np.ndarray], targets: List[np.ndarray]):
        """
        Initialize dataset.
        
        Args:
            sequences: List of sequences, each of shape (seq_len, features)
            targets: List of target sin/cos values, each of shape (2,)
        """
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        return seq, target


class TimeBasedPhaseModel(BasePhaseModel):
    """LSTM-based model to predict phase using time-based windows (handles variable frame rates)."""

    def __init__(
        self,
        lstm_units: int = 64,
        lstm_layers: int = 1,
        dense_units: Tuple[int, ...] = (32,),
        window_duration: float = 0.5,
        phase_buffer_size: int = 5,
        dropout: float = 0.2,
        random_state: Optional[int] = None,
    ):
        """
        Initialize time-based LSTM phase model.

        Args:
            lstm_units: Number of units in LSTM layer(s)
            lstm_layers: Number of LSTM layers (stacked)
            dense_units: Number of units in dense layers after LSTM
            window_duration: Duration of time window in seconds
            phase_buffer_size: Number of previous phases to include as context
            dropout: Dropout rate for regularization
            random_state: Random state for reproducibility
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and PyTorch Lightning are required for LSTM-based phase model")

        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dense_units = dense_units
        self.window_duration = window_duration
        self.phase_buffer_size = phase_buffer_size
        self.dropout = dropout
        self.random_state = random_state
        
        self.is_fitted = False
        self.embed_dim = None  # Will be set during training
        self.embedding_buffer = []  # For live feed inference: (embedding, timestamp)
        self.previous_phase_buffer = []  # For tracking previous phases
        
        # PyTorch Lightning model
        self.model = None
        
        # Set random seeds for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

    def _phase_to_sincos(self, phase: float) -> np.ndarray:
        """Convert phase to sin/cos representation."""
        theta = 2 * np.pi * phase
        return np.array([np.sin(theta), np.cos(theta)])

    def _sincos_to_phase(self, sincos: np.ndarray) -> float:
        """Convert sin/cos representation back to phase."""
        phase = np.arctan2(sincos[0], sincos[1]) / (2 * np.pi)
        return phase % 1.0

    def _create_time_window_sequence(
        self,
        embeddings: List[Optional[np.ndarray]],
        timestamps: List[float],
        idx: int,
    ) -> Optional[np.ndarray]:
        """
        Create a time-based window sequence of embeddings for a given index.
        Returns variable-length sequence (LSTM advantage - no normalization needed).

        Args:
            embeddings: List of embedding vectors
            timestamps: List of timestamps
            idx: Current index

        Returns:
            Array of shape (seq_length, embed_dim) or None if insufficient data
        """
        if idx >= len(embeddings) or idx >= len(timestamps):
            return None

        current_time = timestamps[idx]
        window_start = current_time - self.window_duration

        # Find all embeddings within the time window
        window_embeddings = []
        window_times = []

        # Find start index using binary search
        start_idx = bisect.bisect_left(timestamps, window_start)

        # Collect valid embeddings in window
        for i in range(start_idx, idx + 1):
            if i < len(embeddings) and embeddings[i] is not None:
                window_embeddings.append(embeddings[i])
                window_times.append(timestamps[i])

        if len(window_embeddings) == 0:
            return None

        # Return as sequence (no normalization needed - LSTM handles variable length)
        return np.array(window_embeddings)

    def fit(
        self,
        embeddings: List[np.ndarray],
        phases: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> None:
        """Train the LSTM phase model with time-based windows."""
        logger.info("=" * 60)
        logger.info("Starting LSTM Phase Model Training (PyTorch Lightning)")
        logger.info("=" * 60)
        
        if timestamps is None:
            raise ValueError("timestamps are required for time-based model")

        if len(embeddings) == 0:
            raise ValueError("No training data provided")

        if len(embeddings) != len(phases) or len(embeddings) != len(timestamps):
            raise ValueError("embeddings, phases, and timestamps must have same length")

        # Determine embedding dimension
        valid_emb = next((emb for emb in embeddings if emb is not None), None)
        if valid_emb is None:
            raise ValueError("No valid embeddings found")
        self.embed_dim = valid_emb.shape[0]
        
        logger.info(f"Training data statistics:")
        logger.info(f"  Total samples: {len(embeddings)}")
        logger.info(f"  Valid embeddings: {sum(1 for emb in embeddings if emb is not None)}")
        logger.info(f"  Embedding dimension: {self.embed_dim}")
        logger.info(f"  Time window duration: {self.window_duration:.3f} seconds")
        logger.info(f"  Phase buffer size: {self.phase_buffer_size}")
        logger.info(f"  LSTM units: {self.lstm_units}")
        logger.info(f"  LSTM layers: {self.lstm_layers}")
        logger.info(f"  Dense units: {self.dense_units}")
        logger.info(f"  Dropout rate: {self.dropout}")

        # Create sequences from time windows
        logger.info("\nCreating time window sequences...")
        sequences = []
        targets = []
        phase_buffer = []  # Track recent phases for buffer
        previous_window_phase = None  # Track the phase from the previous time window
        skipped_count = 0

        for i in range(len(embeddings)):
            if embeddings[i] is None:
                continue

            # Create time window sequence
            window_seq = self._create_time_window_sequence(embeddings, timestamps, i)
            if window_seq is None or len(window_seq) == 0:
                skipped_count += 1
                continue

            # Get previous phase buffer
            if len(phase_buffer) == 0:
                prev_phases = [phases[i]] * self.phase_buffer_size
            else:
                prev_phases = list(phase_buffer[-self.phase_buffer_size:])
                if len(prev_phases) < self.phase_buffer_size:
                    padding_value = prev_phases[-1] if len(prev_phases) > 0 else phases[i]
                    prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases

            # Convert phase buffer to sin/cos
            prev_phases_sincos = np.array([self._phase_to_sincos(p) for p in prev_phases])
            phase_features = prev_phases_sincos.flatten()  # (phase_buffer_size * 2,)
            
            # Add previous window's phase as additional feature (2 more features: sin/cos)
            if previous_window_phase is not None:
                prev_window_sincos = self._phase_to_sincos(previous_window_phase)
                # Concatenate: [phase_buffer_features..., previous_window_sin, previous_window_cos]
                phase_features = np.concatenate([phase_features, prev_window_sincos])
            else:
                # First window: use current phase or zeros
                prev_window_sincos = self._phase_to_sincos(phases[i])
                phase_features = np.concatenate([phase_features, prev_window_sincos])
            
            # Concatenate phase features to each embedding in sequence
            # window_seq shape: (seq_len, embed_dim)
            # phase_features shape: (phase_buffer_size * 2 + 2,) - buffer + previous window
            # Result: (seq_len, embed_dim + phase_features)
            phase_features_expanded = np.tile(phase_features, (len(window_seq), 1))
            window_with_phase = np.concatenate([window_seq, phase_features_expanded], axis=1)

            sequences.append(window_with_phase)
            targets.append(self._phase_to_sincos(phases[i]))
            
            # Update phase buffer
            phase_buffer.append(phases[i])
            if len(phase_buffer) > self.phase_buffer_size:
                phase_buffer.pop(0)
            
            # Update previous window phase to current window's phase for next iteration
            previous_window_phase = phases[i]

        if len(sequences) == 0:
            raise ValueError("No valid sequences created from training data")

        logger.info(f"  Created {len(sequences)} valid sequences")
        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} samples (insufficient time window data)")
        
        # Pad sequences to same length for batch training
        # Use the maximum sequence length
        max_len = max(len(seq) for seq in sequences)
        min_len = min(len(seq) for seq in sequences)
        avg_len = np.mean([len(seq) for seq in sequences])
        input_dim = sequences[0].shape[1]  # embed_dim + phase_buffer_size * 2 + 2 (previous window)
        
        logger.info(f"\nSequence statistics:")
        logger.info(f"  Sequence length - min: {min_len}, max: {max_len}, avg: {avg_len:.1f}")
        logger.info(f"  Input dimension (embed_dim + phase_features): {input_dim}")
        logger.info(f"    - Embedding dim: {self.embed_dim}")
        logger.info(f"    - Phase features: {self.phase_buffer_size * 2 + 2} (phase buffer: {self.phase_buffer_size * 2} + previous window: 2)")

        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                # Pad with last frame
                padding = np.tile(seq[-1:], (max_len - len(seq), 1))
                padded_seq = np.concatenate([seq, padding], axis=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        logger.info(f"\nTraining data shape: ({len(padded_sequences)}, {max_len}, {input_dim})")
        logger.info(f"Target shape: ({len(targets)}, 2)")
        
        # Create datasets and dataloaders
        val_split = 0.1
        val_size = int(len(padded_sequences) * val_split)
        if val_size > 0:
            train_sequences = padded_sequences[:-val_size]
            train_targets = targets[:-val_size]
            val_sequences = padded_sequences[-val_size:]
            val_targets = targets[-val_size:]
            logger.info(f"\nTrain/validation split: {len(train_sequences)} train, {len(val_sequences)} validation ({val_split*100:.1f}%)")
        else:
            train_sequences = padded_sequences
            train_targets = targets
            val_sequences = padded_sequences
            val_targets = targets
            logger.warning("  Warning: No validation split (insufficient data)")

        train_dataset = PhaseDataset(train_sequences, train_targets)
        val_dataset = PhaseDataset(val_sequences, val_targets)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        # Create model
        logger.info("\nBuilding LSTM model architecture...")
        self.model = PhaseLSTMModule(
            input_dim=input_dim,
            lstm_units=self.lstm_units,
            lstm_layers=self.lstm_layers,
            dense_units=self.dense_units,
            dropout=self.dropout,
            learning_rate=0.001,
        )
        
        logger.info(f"Model architecture:")
        logger.info(f"  LSTM: {self.lstm_layers} layers, {self.lstm_units} units")
        logger.info(f"  Dense: {self.dense_units}")
        logger.info(f"  Dropout: {self.dropout}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Setup trainer
        logger.info("\nStarting training...")
        logger.info(f"  Max epochs: 100")
        logger.info(f"  Batch size: 32")
        logger.info(f"  Early stopping: patience=10, monitor=val_loss")
        logger.info("-" * 60)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True,
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True,
        )
        
        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=[early_stopping, checkpoint_callback],
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,  # We'll use our own logging
        )
        
        # Train
        trainer.fit(self.model, train_loader, val_loader)
        
        # Load best model
        if checkpoint_callback.best_model_path:
            logger.info(f"\nLoading best model from: {checkpoint_callback.best_model_path}")
            self.model = PhaseLSTMModule.load_from_checkpoint(checkpoint_callback.best_model_path)
        
        # Log training results
        logger.info("-" * 60)
        logger.info(f"Training completed!")
        if val_size > 0:
            logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")
        
        # Set model to evaluation mode
        self.model.eval()
        self.is_fitted = True
        
        logger.info("=" * 60)
        logger.info(f"LSTM model training completed successfully!")
        logger.info(f"Total sequences trained: {len(sequences)}")
        logger.info("=" * 60)

    def predict_phase(
        self,
        embedding: Optional[np.ndarray],
        timestamp: Optional[float] = None,
        update_buffer: bool = True,
        log_timing: bool = False,
    ) -> Optional[float]:
        """Predict phase from embedding using LSTM."""
        if embedding is None:
            return None

        if timestamp is None:
            raise ValueError("timestamp is required for time-based model")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = time.perf_counter()

        # Get previous phase buffer
        if len(self.previous_phase_buffer) == 0:
            prev_phases = [0.0] * self.phase_buffer_size
        else:
            prev_phases = list(self.previous_phase_buffer[-self.phase_buffer_size:])
            if len(prev_phases) < self.phase_buffer_size:
                padding_value = prev_phases[-1] if len(prev_phases) > 0 else 0.0
                prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases

        # Update embedding buffer
        if update_buffer:
            self.embedding_buffer.append((embedding, timestamp))
            # Remove old entries outside time window
            cutoff_time = timestamp - self.window_duration
            self.embedding_buffer = [
                (emb, ts) for emb, ts in self.embedding_buffer if ts >= cutoff_time
            ]

        # Create sequence from buffer
        if len(self.embedding_buffer) == 0:
            return None

        # Extract embeddings and timestamps from buffer
        buffer_embeddings = [emb for emb, _ in self.embedding_buffer]
        buffer_timestamps = [ts for _, ts in self.embedding_buffer]

        # Create time window sequence
        window_seq = self._create_time_window_sequence(
            buffer_embeddings, buffer_timestamps, len(buffer_embeddings) - 1
        )
        if window_seq is None or len(window_seq) == 0:
            return None

        # Convert phase buffer to sin/cos
        prev_phases_sincos = np.array([self._phase_to_sincos(p) for p in prev_phases])
        phase_features = prev_phases_sincos.flatten()  # (phase_buffer_size * 2,)
        
        # Add previous window's phase as additional feature
        # Get the last predicted phase (previous window's phase)
        if len(self.previous_phase_buffer) > 0:
            prev_window_phase = self.previous_phase_buffer[-1]
        else:
            # First window: use 0.0
            prev_window_phase = 0.0
        
        prev_window_sincos = self._phase_to_sincos(prev_window_phase)
        # Concatenate: [phase_buffer_features..., previous_window_sin, previous_window_cos]
        phase_features = np.concatenate([phase_features, prev_window_sincos])
        
        phase_features_expanded = np.tile(phase_features, (len(window_seq), 1))
        window_with_phase = np.concatenate([window_seq, phase_features_expanded], axis=1)

        # Predict using PyTorch model
        self.model.eval()
        with torch.no_grad():
            # Pad to at least length 1
            if len(window_with_phase) == 0:
                return None
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(window_with_phase).unsqueeze(0)  # (1, seq_len, features)
            
            # Predict
            output = self.model(input_tensor)  # (1, 2)
            sincos = output[0].cpu().numpy()
        
        phase = self._sincos_to_phase(sincos)

        # Update previous phase buffer
        if update_buffer:
            self.previous_phase_buffer.append(phase)
            if len(self.previous_phase_buffer) > self.phase_buffer_size:
                self.previous_phase_buffer.pop(0)

        elapsed = time.perf_counter() - start_time
        if log_timing:
            logger.debug(f"Phase prediction time: {elapsed*1000:.3f} ms")

        return phase

    def predict_phases(
        self,
        embeddings: List[Optional[np.ndarray]],
        timestamps: Optional[List[float]] = None,
        log_timing: bool = False,
    ) -> List[Optional[float]]:
        """Predict phases for a sequence using LSTM."""
        if timestamps is None:
            raise ValueError("timestamps are required for time-based model")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if log_timing:
            start_time = time.perf_counter()

        phases = []
        phase_buffer = []
        previous_window_phase = None  # Track the phase from the previous time window

        # Collect all sequences with their indices
        sequences = []
        sequence_indices = []

        for i in range(len(embeddings)):
            if embeddings[i] is not None:
                # Get previous phase buffer
                if len(phase_buffer) == 0:
                    prev_phases = [0.0] * self.phase_buffer_size
                else:
                    prev_phases = list(phase_buffer[-self.phase_buffer_size:])
                    if len(prev_phases) < self.phase_buffer_size:
                        padding_value = prev_phases[-1] if len(prev_phases) > 0 else 0.0
                        prev_phases = [padding_value] * (self.phase_buffer_size - len(prev_phases)) + prev_phases

                window_seq = self._create_time_window_sequence(embeddings, timestamps, i)
                if window_seq is not None and len(window_seq) > 0:
                    # Convert phase buffer to sin/cos
                    prev_phases_sincos = np.array([self._phase_to_sincos(p) for p in prev_phases])
                    phase_features = prev_phases_sincos.flatten()  # (phase_buffer_size * 2,)
                    
                    # Add previous window's phase as additional feature
                    if previous_window_phase is not None:
                        prev_window_sincos = self._phase_to_sincos(previous_window_phase)
                    else:
                        # First window: use 0.0
                        prev_window_sincos = self._phase_to_sincos(0.0)
                    
                    # Concatenate: [phase_buffer_features..., previous_window_sin, previous_window_cos]
                    phase_features = np.concatenate([phase_features, prev_window_sincos])
                    phase_features_expanded = np.tile(phase_features, (len(window_seq), 1))
                    window_with_phase = np.concatenate([window_seq, phase_features_expanded], axis=1)
                    
                    sequences.append(window_with_phase)
                    sequence_indices.append(i)
                else:
                    phases.append(None)
            else:
                phases.append(None)

        if len(sequences) == 0:
            return phases

        # Pad sequences
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = np.tile(seq[-1:], (max_len - len(seq), 1))
                padded_seq = np.concatenate([seq, padding], axis=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Predict in batches
        self.model.eval()
        batch_size = 32
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(padded_sequences), batch_size):
                batch_seqs = padded_sequences[i:i+batch_size]
                batch_tensor = torch.FloatTensor(np.array(batch_seqs))  # (batch, seq_len, features)
                batch_output = self.model(batch_tensor)  # (batch, 2)
                predictions.extend(batch_output.cpu().numpy())

        # Map predictions back to original indices
        # Update phase_buffer and previous_window_phase as we process predictions
        for pred_idx, orig_idx in enumerate(sequence_indices):
            sincos = predictions[pred_idx]
            phase = self._sincos_to_phase(sincos)
            phases[orig_idx] = phase
            # Update phase buffer
            phase_buffer.append(phase)
            if len(phase_buffer) > self.phase_buffer_size:
                phase_buffer.pop(0)
            # Update previous window phase for next iteration
            previous_window_phase = phase

        if log_timing:
            total_time = time.perf_counter() - start_time
            valid_count = sum(1 for p in phases if p is not None)
            if valid_count > 0:
                avg_time = total_time / valid_count * 1000
                logger.info(
                    f"Phase prediction sequence: {len(embeddings)} frames, "
                    f"{valid_count} valid, {total_time*1000:.1f} ms total, "
                    f"{avg_time:.3f} ms/frame avg"
                )

        return phases

    def reset_buffer(self) -> None:
        """Reset buffers."""
        self.embedding_buffer = []
        self.previous_phase_buffer = []

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Remove extension from filepath
        base_name = os.path.splitext(filepath)[0]
        checkpoint_path = base_name + "_checkpoint.ckpt"
        
        # Save PyTorch Lightning checkpoint
        # Save state dict and hyperparameters in a format compatible with load_from_checkpoint
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'hyper_parameters': self.model.hparams,
        }
        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata = {
            "lstm_units": self.lstm_units,
            "lstm_layers": self.lstm_layers,
            "dense_units": self.dense_units,
            "window_duration": self.window_duration,
            "phase_buffer_size": self.phase_buffer_size,
            "dropout": self.dropout,
            "random_state": self.random_state,
            "embed_dim": self.embed_dim,
        }

        import joblib
        joblib.dump(metadata, filepath + "_metadata.joblib")

    @classmethod
    def load(cls, filepath: str) -> "TimeBasedPhaseModel":
        """Load a model from disk."""
        # Remove extension from filepath
        base_name = os.path.splitext(filepath)[0]
        checkpoint_path = base_name + "_checkpoint.ckpt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        import joblib
        metadata = joblib.load(filepath + "_metadata.joblib")

        phase_model = cls(
            lstm_units=metadata["lstm_units"],
            lstm_layers=metadata["lstm_layers"],
            dense_units=metadata["dense_units"],
            window_duration=metadata["window_duration"],
            phase_buffer_size=metadata["phase_buffer_size"],
            dropout=metadata["dropout"],
            random_state=metadata["random_state"],
        )

        # Load model from checkpoint
        # Input dim: embed_dim + phase_buffer_size * 2 + 2 (previous window phase)
        input_dim = metadata["embed_dim"] + metadata["phase_buffer_size"] * 2 + 2
        
        # Load checkpoint with weights_only=False for compatibility with Lightning checkpoints
        # Note: This is safe as we control the checkpoint source
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create model instance
        phase_model.model = PhaseLSTMModule(
            input_dim=input_dim,
            lstm_units=metadata["lstm_units"],
            lstm_layers=metadata["lstm_layers"],
            dense_units=metadata["dense_units"],
            dropout=metadata["dropout"],
        )
        
        # Load state dict (handle both formats: direct state_dict or nested in checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint format or our custom format
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from Lightning checkpoint)
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            phase_model.model.load_state_dict(state_dict)
        else:
            # Direct state dict
            phase_model.model.load_state_dict(checkpoint)
        
        phase_model.model.eval()

        phase_model.is_fitted = True
        phase_model.embed_dim = metadata["embed_dim"]

        return phase_model
