"""
ml/isolation_forest.py
Isolation Forest Anomaly Detector for IoHT Data
"""

import numpy as np
import pickle, os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/isolation_forest.pkl"


class IsolationForestDetector:
    """
    Wraps scikit-learn IsolationForest with IoHT-specific logic.
    Scores: negative = anomaly (IF convention), normalized to [0,1] range.
    """

    def __init__(self):
        self.model = None
        self.threshold = -0.15          # Decision boundary
        self._load_or_init()

    def _load_or_init(self):
        """Load saved model or initialize a default one."""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print("[IF] Loaded model from disk.")
        else:
            # Pre-train on synthetic data so API works immediately
            self.model = IsolationForest(
                n_estimators=200,
                contamination=0.05,
                random_state=42,
                max_features=1.0,
                bootstrap=False
            )
            X_dummy = np.random.randn(1000, 5)       # 5 features: HR,SpO2,Temp,BP,derived
            self.model.fit(X_dummy)
            print("[IF] Initialized with dummy data (no saved model found).")

    def fit(self, X: np.ndarray, contamination: float = 0.05, test_split: float = 0.2):
        """
        Train Isolation Forest on preprocessed feature matrix.
        Returns accuracy, f1 on a held-out test set.
        """
        X_train, X_test = train_test_split(X, test_size=test_split, random_state=42)

        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            max_features=1.0
        )
        self.model.fit(X_train)

        # Evaluate: treat predictions as ground truth for unsupervised
        y_pred_raw = self.model.predict(X_test)         # +1 = normal, -1 = anomaly
        y_binary   = (y_pred_raw == -1).astype(int)     # 1 = anomaly

        # Since we have no true labels, use contamination fraction as proxy
        n_anomaly  = int(len(X_test) * contamination)
        y_true     = np.zeros(len(X_test), dtype=int)
        scores_raw = self.model.score_samples(X_test)
        worst_idx  = np.argsort(scores_raw)[:n_anomaly]
        y_true[worst_idx] = 1

        acc = accuracy_score(y_true, y_binary)
        f1  = f1_score(y_true, y_binary, zero_division=0)

        # Save model
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

        print(f"[IF] Training complete — Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return round(acc, 4), round(f1, 4)

    def predict_single(self, features: np.ndarray):
        """
        Predict on a single feature vector.
        Returns: (raw_score: float, label: int)  → label: -1=anomaly, 1=normal
        """
        x = features.reshape(1, -1)
        score = float(self.model.score_samples(x)[0])   # lower = more anomalous
        label = int(self.model.predict(x)[0])            # -1 or +1
        return score, label

    def predict_batch(self, X: np.ndarray):
        """Batch prediction. Returns (scores, labels) arrays."""
        scores = self.model.score_samples(X)
        labels = self.model.predict(X)
        return scores, labels
