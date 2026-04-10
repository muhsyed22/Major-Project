"""
ml/preprocessor.py
Data Preprocessing for IoHT Telemetry
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
import pickle, os

SCALER_PATH = "models/scaler.pkl"

FEATURE_COLS = ["heart_rate", "spo2", "temperature", "bp_systolic"]


class Preprocessor:
    """
    Normalise IoHT features and add derived indicators.
    Features after processing: [HR, SpO2, Temp, BP, HR_SpO2_risk]  (5-dim)
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False
        self._load_scaler()

    def _load_scaler(self):
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            self._fitted = True

    def _save_scaler(self):
        os.makedirs("models", exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

    def _extract_features(self, records: List[Dict]) -> np.ndarray:
        """Convert list of dicts to numpy feature matrix."""
        rows = []
        for r in records:
            hr   = float(r.get("heart_rate",  r.get("hr",  70)))
            spo2 = float(r.get("spo2",        r.get("SpO2", 98)))
            temp = float(r.get("temperature", r.get("temp", 36.6)))
            bp   = float(r.get("bp_systolic", r.get("bp",  120)))
            # Derived: combined cardiac risk indicator
            risk = (hr / 200) * (1 - spo2 / 100) * (bp / 200)
            rows.append([hr, spo2, temp, bp, risk])
        return np.array(rows, dtype=np.float32)

    def fit_transform(self, records: List[Dict]) -> np.ndarray:
        """Fit scaler and transform dataset."""
        X = self._extract_features(records)
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        self._save_scaler()
        return X_scaled

    def transform(self, records: List[Dict]) -> np.ndarray:
        """Transform without refitting (use fitted scaler)."""
        X = self._extract_features(records)
        if self._fitted:
            return self.scaler.transform(X)
        return X  # fallback if not fitted

    def transform_single(self, data: Dict) -> np.ndarray:
        """Transform a single reading dict to feature vector."""
        return self.transform([data])[0]

    @staticmethod
    def generate_synthetic_dataset(n_samples: int = 5000, anomaly_frac: float = 0.05) -> List[Dict]:
        """
        Generate a synthetic IoHT dataset for demo/testing.
        Normal: HR~N(75,12), SpO2~N(97,1), Temp~N(36.8,0.4), BP~N(120,15)
        Anomaly: HR~N(140,20), SpO2~N(88,4), Temp~N(39.0,0.8), BP~N(170,20)
        """
        np.random.seed(42)
        records = []
        devices = ["ECG-MON-01","SPO2-01","BP-MON-03","TEMP-02","GLUCM-01","ECG-MON-04"]

        for i in range(n_samples):
            is_anomaly = np.random.random() < anomaly_frac
            dev = devices[i % len(devices)]

            if is_anomaly:
                r = {
                    "device_id":   dev,
                    "heart_rate":  max(20, np.random.normal(140, 20)),
                    "spo2":        min(100, max(60, np.random.normal(88, 4))),
                    "temperature": np.random.normal(39.0, 0.8),
                    "bp_systolic": max(80, np.random.normal(170, 20)),
                    "label": "ANOMALY"
                }
            else:
                r = {
                    "device_id":   dev,
                    "heart_rate":  max(40, np.random.normal(75, 12)),
                    "spo2":        min(100, max(90, np.random.normal(97, 1))),
                    "temperature": np.random.normal(36.8, 0.4),
                    "bp_systolic": max(90, np.random.normal(120, 15)),
                    "label": "NORMAL"
                }
            records.append(r)

        return records
