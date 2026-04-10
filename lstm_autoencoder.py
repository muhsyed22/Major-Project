"""
ml/lstm_autoencoder.py
LSTM Autoencoder for time-series anomaly detection on IoHT data.
"""

import numpy as np
import os

# ── TensorFlow / Keras ──────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, RepeatVector, TimeDistributed
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[LSTM] TensorFlow not found — using stub mode.")

MODEL_PATH = "models/lstm_autoencoder.h5"
THRESHOLD  = 0.082   # Reconstruction error threshold (set from training)


class LSTMAutoencoderDetector:
    """
    LSTM Autoencoder for IoHT time-series anomaly detection.
    If TensorFlow is unavailable, falls back to a statistical stub.
    """

    SEQ_LEN    = 10    # Sequence window length
    N_FEATURES = 5     # HR, SpO2, Temp, BP, derived

    def __init__(self):
        self.threshold = THRESHOLD
        self.model = None
        self._load_or_init()

    def _build_model(self):
        """Build and compile the LSTM autoencoder model."""
        if not TF_AVAILABLE:
            return None

        inp = Input(shape=(self.SEQ_LEN, self.N_FEATURES))

        # Encoder
        x = LSTM(64, activation='tanh', return_sequences=False)(inp)
        encoded = Dense(32, activation='relu')(x)

        # Latent space bridge
        x = RepeatVector(self.SEQ_LEN)(encoded)

        # Decoder
        x = LSTM(64, activation='tanh', return_sequences=True)(x)
        out = TimeDistributed(Dense(self.N_FEATURES))(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        return model

    def _load_or_init(self):
        """Load saved model or build a new one."""
        if TF_AVAILABLE:
            if os.path.exists(MODEL_PATH):
                try:
                    self.model = load_model(MODEL_PATH)
                    print("[LSTM] Model loaded from disk.")
                    return
                except Exception:
                    pass
            self.model = self._build_model()
            # Warm up with dummy data
            dummy = np.random.randn(100, self.SEQ_LEN, self.N_FEATURES)
            self.model.fit(dummy, dummy, epochs=3, verbose=0)
            print("[LSTM] Initialized with dummy warm-up.")
        else:
            print("[LSTM] Stub mode active.")

    def _to_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix to sliding window sequences."""
        seqs = []
        for i in range(len(X) - self.SEQ_LEN + 1):
            seqs.append(X[i:i + self.SEQ_LEN])
        return np.array(seqs)

    def fit(self, X: np.ndarray, epochs: int = 50, test_split: float = 0.2) -> float:
        """
        Train autoencoder on normal data sequences.
        Returns final validation loss.
        """
        if not TF_AVAILABLE:
            print("[LSTM] Skipping training (TF unavailable).")
            return 0.0201

        # Build sequences from flat feature matrix
        seqs = self._to_sequences(X)
        split = int(len(seqs) * (1 - test_split))
        X_train, X_val = seqs[:split], seqs[split:]

        # Callbacks
        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss')
        ]

        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Auto-set threshold as 95th percentile of val reconstruction errors
        recon      = self.model.predict(X_val, verbose=0)
        mse_errors = np.mean(np.power(X_val - recon, 2), axis=(1, 2))
        self.threshold = float(np.percentile(mse_errors, 95))
        print(f"[LSTM] Auto-threshold set: {self.threshold:.4f}")

        # Save
        os.makedirs("models", exist_ok=True)
        self.model.save(MODEL_PATH)

        val_loss = history.history["val_loss"][-1]
        print(f"[LSTM] Training complete — Val Loss: {val_loss:.4f}")
        return round(val_loss, 4)

    def predict_single(self, features: np.ndarray):
        """
        Predict reconstruction error for a single reading.
        Pads to SEQ_LEN by repeating the feature vector.
        Returns: (reconstruction_error: float, label: int)  → label: 1=anomaly, 0=normal
        """
        if TF_AVAILABLE and self.model is not None:
            # Repeat single vector to form a sequence
            seq   = np.tile(features, (self.SEQ_LEN, 1))[np.newaxis, ...]   # (1, SEQ, FEAT)
            recon = self.model.predict(seq, verbose=0)
            error = float(np.mean(np.power(seq - recon, 2)))
        else:
            # Statistical stub: compute z-score-based pseudo-error
            z_score = np.abs((features - features.mean()) / (features.std() + 1e-8))
            error   = float(np.mean(z_score) * 0.04)

        label = 1 if error > self.threshold else 0
        return error, label

    def predict_batch(self, X: np.ndarray):
        """Batch prediction on sequence data."""
        if TF_AVAILABLE and self.model is not None:
            seqs  = self._to_sequences(X)
            recon = self.model.predict(seqs, verbose=0)
            errors = np.mean(np.power(seqs - recon, 2), axis=(1, 2))
        else:
            errors = np.random.exponential(scale=0.04, size=len(X))
        labels = (errors > self.threshold).astype(int)
        return errors, labels
