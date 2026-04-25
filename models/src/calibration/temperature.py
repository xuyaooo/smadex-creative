"""
Temperature scaling for the multiclass status classifier.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.

Fits a single scalar T > 0 on held-out logits so that softmax(logits / T) is
better calibrated. Tree-based models (XGBoost, LightGBM) are also miscalibrated;
the same trick works on `predict_proba` outputs by inverting them to logits.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def _probs_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert (N, K) probs to (N, K) logits via log + center."""
    p = np.clip(probs, eps, 1.0 - eps)
    return np.log(p) - np.log(p).mean(axis=1, keepdims=True)


def _softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0)).mean())


class TemperatureScaler:
    """Single-parameter calibration. Fit on OOF probs+labels, apply at inference."""

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits = _probs_to_logits(probs)
        res = minimize_scalar(
            lambda T: _nll(_softmax_T(logits, T), y),
            bounds=(0.05, 10.0), method="bounded",
        )
        self.T = float(res.x)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if abs(self.T - 1.0) < 1e-6:
            return probs
        return _softmax_T(_probs_to_logits(probs), self.T)

    def expected_calibration_error(self, probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """ECE: weighted gap between confidence and accuracy across bins."""
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == y).astype(np.float32)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() == 0:
                continue
            ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"T": self.T}, f)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        obj = cls()
        with open(path, "rb") as f:
            obj.T = float(pickle.load(f)["T"])
        return obj
