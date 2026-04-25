"""
Bayesian Online Changepoint Detection on a daily CTR sequence.

Reference: Adams & MacKay, "Bayesian Online Changepoint Detection", 2007 (arXiv:0710.3742).
Used to flag the day a creative's CTR distribution materially shifts — a strong
fatigue signal, complementary to the LightGBM "will fatigue" classifier.

Implementation is the standard run-length forward filter with a Gaussian observation
likelihood and a constant hazard h(t) = 1 / lambda. Returns, for each timestep,
the posterior probability that a changepoint occurred at or before that step.
"""
from typing import Tuple

import numpy as np


def _gaussian_logpdf(x: float, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi * var) + ((x - mu) ** 2) / var)


def bocpd(
    series: np.ndarray,
    hazard_lambda: float = 50.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Run BOCPD with a Normal-Gamma conjugate prior.

    Returns:
      cp_prob: (T,) — posterior P(changepoint within last 5 steps) per t
      best_cp: int — argmax of cp_prob (the most likely changepoint day)
    """
    series = np.asarray(series, dtype=np.float64)
    series = np.where(np.isnan(series), 0.0, series)
    T = len(series)
    if T < 4:
        return np.zeros(max(T, 1)), -1

    H = 1.0 / max(hazard_lambda, 1.0)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient stats per run length
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    cp_prob = np.zeros(T)

    for t in range(T):
        x = series[t]
        # Predictive: Student-t with df=2*alpha; approximate via Gaussian for speed
        var = beta * (kappa + 1) / (alpha * kappa)
        var = np.maximum(var, 1e-8)
        log_pred = _gaussian_logpdf(x, mu, var)

        # Growth probabilities (no changepoint)
        growth = R[:t + 1, t] * np.exp(log_pred) * (1 - H)
        # Changepoint probability
        cp = float(np.sum(R[:t + 1, t] * np.exp(log_pred) * H))

        # Update R
        R[1:t + 2, t + 1] = growth
        R[0, t + 1] = cp
        R[:, t + 1] /= max(R[:, t + 1].sum(), 1e-12)

        # Update sufficient stats: append a new run length
        mu_new = np.concatenate([[mu0], (kappa * mu + x) / (kappa + 1)])
        kappa_new = np.concatenate([[kappa0], kappa + 1])
        alpha_new = np.concatenate([[alpha0], alpha + 0.5])
        beta_new = np.concatenate([
            [beta0],
            beta + (kappa * (x - mu) ** 2) / (2 * (kappa + 1)),
        ])
        mu, kappa, alpha, beta = mu_new, kappa_new, alpha_new, beta_new

        # Standard BOCPD output: P(run length = 0 at time t+1)
        # = posterior probability that *this* step was a changepoint.
        cp_prob[t] = float(R[0, t + 1])

    # Most likely changepoint day = argmax of cp_prob (after a small burn-in)
    burn = min(3, T - 1)
    best = int(np.argmax(cp_prob[burn:]) + burn) if T > burn else -1
    return cp_prob, best


def fatigue_changepoint(
    daily_ctr: np.ndarray,
    hazard_lambda: float = 30.0,
    threshold_multiple: float = 3.0,
) -> dict:
    """Convenience wrapper that returns a single dict consumable by Health Score.

    The previous static threshold (0.4) was unreachable: the run-length-zero
    posterior `R[0, t+1]` is bounded near the prior hazard rate `H = 1/lambda`
    when likelihoods are similar across run-lengths, so for `lambda=30` the
    typical baseline `cp_prob` sits around 0.033. Using a *relative* threshold
    (multiple of the prior baseline) makes the detector actually fire.
    """
    if len(daily_ctr) < 4:
        return {"has_changepoint": False, "changepoint_day": -1, "max_cp_prob": 0.0}
    cp_prob, best_day = bocpd(daily_ctr, hazard_lambda=hazard_lambda)
    if len(cp_prob) == 0:
        return {"has_changepoint": False, "changepoint_day": -1, "max_cp_prob": 0.0}
    baseline = 1.0 / max(hazard_lambda, 1.0)  # the prior P(run length = 0)
    threshold = threshold_multiple * baseline
    has_cp = bool(cp_prob.max() > threshold)
    return {
        "has_changepoint": has_cp,
        "changepoint_day": int(best_day) if has_cp else -1,
        "max_cp_prob": float(cp_prob.max()),
        "threshold": float(threshold),
    }
