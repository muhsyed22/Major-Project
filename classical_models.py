"""
Classical anomaly models for IoHT backend.
"""

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

SVM_PATH = "models/oneclass_svm.pkl"
RF_PATH = "models/random_forest.pkl"


class ClassicalDetectors:
    def __init__(self):
        self.svm = None
        self.rf = None
        self.rf_ready = False
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(SVM_PATH):
            with open(SVM_PATH, "rb") as f:
                self.svm = pickle.load(f)
        else:
            self.svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
            dummy = np.random.randn(1000, 5)
            self.svm.fit(dummy)

        if os.path.exists(RF_PATH):
            with open(RF_PATH, "rb") as f:
                self.rf = pickle.load(f)
            self.rf_ready = True
        else:
            self.rf = RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced_subsample",
            )
            self.rf_ready = False

    def fit_svm(self, X: np.ndarray, test_split: float = 0.2):
        X_train, X_test = train_test_split(X, test_size=test_split, random_state=42)
        self.svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        self.svm.fit(X_train)

        pred = self.svm.predict(X_test)  # -1 anomaly, +1 normal
        y_pred = (pred == -1).astype(int)
        n_anomaly = max(1, int(len(X_test) * 0.05))
        scores = self.svm.decision_function(X_test)
        worst_idx = np.argsort(scores)[:n_anomaly]
        y_true = np.zeros(len(X_test), dtype=int)
        y_true[worst_idx] = 1

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        os.makedirs("models", exist_ok=True)
        with open(SVM_PATH, "wb") as f:
            pickle.dump(self.svm, f)
        return round(acc, 4), round(f1, 4)

    def fit_rf(self, X: np.ndarray, y: Optional[np.ndarray], test_split: float = 0.2):
        if y is None or len(y) != len(X):
            return None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        self.rf = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced_subsample",
        )
        self.rf.fit(X_train, y_train)
        y_pred = self.rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        self.rf_ready = True

        os.makedirs("models", exist_ok=True)
        with open(RF_PATH, "wb") as f:
            pickle.dump(self.rf, f)
        return round(acc, 4), round(f1, 4)

    def predict_svm_single(self, features: np.ndarray):
        x = features.reshape(1, -1)
        score = float(self.svm.decision_function(x)[0])
        label = int(self.svm.predict(x)[0])  # -1 anomaly, +1 normal
        return score, label

    def predict_rf_single(self, features: np.ndarray):
        if not self.rf_ready:
            return 0.0, 0
        x = features.reshape(1, -1)
        proba = self.rf.predict_proba(x)[0]
        anomaly_proba = float(proba[1]) if len(proba) > 1 else 0.0
        label = 1 if anomaly_proba >= 0.5 else 0
        return anomaly_proba, label

