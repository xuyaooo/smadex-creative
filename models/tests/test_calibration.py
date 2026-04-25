"""Unit tests for src.calibration.temperature.TemperatureScaler."""
from pathlib import Path

import numpy as np
import pytest

from src.calibration.temperature import TemperatureScaler


def _miscalibrated_probs(n: int = 400, k: int = 4, seed: int = 0):
    """Build a synthetic, overconfident multiclass probs array + labels.

    Logits are scaled by 4x so softmax is sharper than warranted by the
    label distribution -> ECE is large and a T > 1 should fix it.
    """
    rng = np.random.default_rng(seed)
    # True latent logits
    logits = rng.normal(size=(n, k)).astype(np.float64)
    # Sample labels from a properly-calibrated softmax of `logits`
    p_true = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_true /= p_true.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=p_true[i]) for i in range(n)])

    # Now sharpen the logits 4x to mimic a miscalibrated classifier
    over_logits = logits * 4.0
    e = np.exp(over_logits - over_logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs.astype(np.float64), y


def test_fit_improves_ece_on_miscalibrated_probs():
    probs, y = _miscalibrated_probs()
    scaler = TemperatureScaler()
    ece_before = scaler.expected_calibration_error(probs, y)

    scaler.fit(probs, y)
    probs_cal = scaler.transform(probs)
    ece_after = scaler.expected_calibration_error(probs_cal, y)

    assert ece_after < ece_before, (
        f"calibration should not worsen ECE: before={ece_before:.4f} "
        f"after={ece_after:.4f}"
    )
    # Overconfident -> T should be > 1 to soften
    assert scaler.T > 1.0


def test_transform_is_identity_when_T_is_one():
    scaler = TemperatureScaler()
    assert scaler.T == 1.0
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=50).astype(np.float64)
    out = scaler.transform(probs)
    np.testing.assert_array_equal(out, probs)


def test_save_load_roundtrip_preserves_T(tmp_path: Path):
    scaler = TemperatureScaler()
    scaler.T = 2.345
    p = tmp_path / "temp.pkl"
    scaler.save(p)

    loaded = TemperatureScaler.load(p)
    assert loaded.T == pytest.approx(2.345)


def test_fit_on_well_calibrated_uniform_probs_keeps_T_near_1():
    """When inputs are already roughly uniform / matched to labels,
    the optimal T should be approximately 1."""
    rng = np.random.default_rng(2)
    n, k = 600, 4
    logits = rng.normal(size=(n, k)).astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    # Sample labels from the *same* distribution -> well-calibrated
    y = np.array([rng.choice(k, p=probs[i]) for i in range(n)])

    scaler = TemperatureScaler().fit(probs, y)
    # Should land in a sane window around 1.0
    assert 0.5 <= scaler.T <= 2.0
