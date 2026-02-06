"""Causal ML evaluation metrics.

Provides standard metrics for evaluating heterogeneous treatment effect
(uplift) models when ground-truth individual effects are unavailable
but randomized (or quasi-randomized) treatment assignment is available.

Functions
---------
- :func:`cumulative_gain_curve` – compute the cumulative gain (uplift) curve.
- :func:`auuc` – Area Under the Uplift Curve (AUUC).
- :func:`qini_curve` – compute the Qini curve.
- :func:`qini_coefficient` – Qini coefficient (area between Qini curve and
  random).
"""

from typing import Optional, Tuple

import numpy as np


def _sort_by_uplift(
    y: np.ndarray,
    w: np.ndarray,
    uplift_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sort arrays by descending uplift score."""
    order = np.argsort(-uplift_score)
    y_s = np.asarray(y)[order]
    w_s = np.asarray(w)[order]
    u_s = np.asarray(uplift_score)[order]
    sw_s = np.asarray(sample_weight)[order] if sample_weight is not None else None
    return y_s, w_s, u_s, sw_s


# ---------------------------------------------------------------------------
# Cumulative gain (uplift) curve
# ---------------------------------------------------------------------------


def cumulative_gain_curve(
    y_true: np.ndarray,
    w_true: np.ndarray,
    uplift_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the cumulative gain (uplift) curve.

    Samples are sorted by ``uplift_score`` in descending order.  At each
    fraction of the population the observed uplift (difference in conversion
    rates between treated and control) multiplied by the fraction of the
    population seen so far is computed.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed binary outcome (0 or 1).
    w_true : array-like of shape (n_samples,)
        Observed binary treatment (0 or 1).
    uplift_score : array-like of shape (n_samples,)
        Predicted CATE / uplift score (higher ⇒ more benefit from treatment).

    Returns
    -------
    fractions : ndarray of shape (n_samples,)
        Fraction of population from 0 to 1.
    gains : ndarray of shape (n_samples,)
        Cumulative gain at each fraction.
    """
    y_s, w_s, _, _ = _sort_by_uplift(y_true, w_true, uplift_score)
    n = len(y_s)

    n_treat = np.cumsum(w_s).astype(float)
    n_control = np.cumsum(1 - w_s).astype(float)
    y_treat = np.cumsum(y_s * w_s).astype(float)
    y_control = np.cumsum(y_s * (1 - w_s)).astype(float)

    # Avoid division by zero.
    rate_treat = np.divide(
        y_treat, n_treat, out=np.zeros_like(y_treat), where=n_treat > 0
    )
    rate_control = np.divide(
        y_control, n_control, out=np.zeros_like(y_control), where=n_control > 0
    )

    uplift_at_k = rate_treat - rate_control
    fractions = np.arange(1, n + 1) / n
    gains = uplift_at_k * fractions

    return fractions, gains


# ---------------------------------------------------------------------------
# AUUC
# ---------------------------------------------------------------------------


def auuc(
    y_true: np.ndarray,
    w_true: np.ndarray,
    uplift_score: np.ndarray,
    normalize: bool = True,
) -> float:
    """Area Under the Uplift Curve (AUUC).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed binary outcome.
    w_true : array-like of shape (n_samples,)
        Observed binary treatment indicator.
    uplift_score : array-like of shape (n_samples,)
        Predicted CATE / uplift score.
    normalize : bool, default=True
        If ``True``, subtract the area of a random model (diagonal) so that
        a random model scores 0.

    Returns
    -------
    float
        AUUC value.
    """
    fracs, gains = cumulative_gain_curve(y_true, w_true, uplift_score)
    area = float(np.trapezoid(gains, fracs))

    if normalize:
        # Random model area = 0.5 * overall_uplift
        y_arr, w_arr = np.asarray(y_true), np.asarray(w_true)
        m1 = w_arr == 1
        m0 = w_arr == 0
        overall_uplift = 0.0
        if m1.sum() > 0 and m0.sum() > 0:
            overall_uplift = y_arr[m1].mean() - y_arr[m0].mean()
        random_area = 0.5 * overall_uplift
        area -= random_area

    return float(area)


# ---------------------------------------------------------------------------
# Qini curve & coefficient
# ---------------------------------------------------------------------------


def qini_curve(
    y_true: np.ndarray,
    w_true: np.ndarray,
    uplift_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Qini curve.

    The Qini curve counts the *incremental* number of positive outcomes
    attributable to treatment as a function of the population fraction
    targeted.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed binary outcome.
    w_true : array-like of shape (n_samples,)
        Observed binary treatment indicator.
    uplift_score : array-like of shape (n_samples,)
        Predicted CATE / uplift score.

    Returns
    -------
    fractions : ndarray of shape (n_samples + 1,)
        Population fraction (starts at 0).
    qini : ndarray of shape (n_samples + 1,)
        Qini value at each fraction (starts at 0).
    """
    y_s, w_s, _, _ = _sort_by_uplift(y_true, w_true, uplift_score)
    n = len(y_s)

    n_treat = np.cumsum(w_s).astype(float)
    n_control = np.cumsum(1 - w_s).astype(float)
    y_treat = np.cumsum(y_s * w_s).astype(float)
    y_control = np.cumsum(y_s * (1 - w_s)).astype(float)

    # Qini: Y_treat - Y_control * (N_treat / N_control)  (adjusted for group sizes)
    adj_control = np.divide(
        y_control * n_treat,
        n_control,
        out=np.zeros_like(y_control),
        where=n_control > 0,
    )
    qini = y_treat - adj_control

    # Prepend origin
    fractions = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    qini = np.concatenate([[0.0], qini])

    return fractions, qini


def qini_coefficient(
    y_true: np.ndarray,
    w_true: np.ndarray,
    uplift_score: np.ndarray,
) -> float:
    """Qini coefficient: area between the Qini curve and the random diagonal.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed binary outcome.
    w_true : array-like of shape (n_samples,)
        Observed binary treatment indicator.
    uplift_score : array-like of shape (n_samples,)
        Predicted CATE / uplift score.

    Returns
    -------
    float
        Qini coefficient value.
    """
    fracs, q = qini_curve(y_true, w_true, uplift_score)
    model_area = float(np.trapezoid(q, fracs))

    # Random line: straight from (0, 0) to (1, qini_total)
    qini_total = q[-1]
    random_area = 0.5 * qini_total

    return float(model_area - random_area)
