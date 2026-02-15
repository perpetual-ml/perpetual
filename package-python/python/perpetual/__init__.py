"""Perpetual — a self-generalizing Gradient Boosting Machine.

Provides:
  - :class:`PerpetualBooster` — single-output gradient booster.
  - :class:`PerpetualClassifier`, :class:`PerpetualRegressor`, :class:`PerpetualRanker` — scikit-learn wrappers.
  - :class:`MultiOutputBooster` — multi-target booster (via internal import).
  - :class:`UpliftBooster` — R-Learner uplift estimator.
  - :class:`BraidedBooster` — Instrumental Variable (BoostIV) estimator.
  - :class:`SLearner`, :class:`TLearner`, :class:`XLearner`, :class:`DRLearner` — meta-learner CATE estimators.
  - :class:`DMLEstimator` — Double Machine Learning (DML) estimator with cross-fitting.
  - :class:`FairClassifier` — fairness-aware classifier (Demographic Parity / Equalized Odds).
  - :class:`PolicyLearner` — IPW / AIPW policy learning.
  - :class:`PerpetualRiskEngine` — adverse-action reason-code generator.
  - :mod:`causal_metrics` — AUUC, Qini coefficient, cumulative gain curves.
"""

from __future__ import annotations

from perpetual.booster import (
    PerpetualBooster,
    compute_calibration_curve,
    expected_calibration_error,
)
from perpetual.causal_metrics import (
    auuc,
    cumulative_gain_curve,
    qini_coefficient,
    qini_curve,
)
from perpetual.dml import DMLEstimator
from perpetual.fairness import FairClassifier
from perpetual.iv import BraidedBooster
from perpetual.meta_learners import DRLearner, SLearner, TLearner, XLearner
from perpetual.policy import PolicyLearner
from perpetual.risk import PerpetualRiskEngine
from perpetual.sklearn import PerpetualClassifier, PerpetualRanker, PerpetualRegressor
from perpetual.uplift import UpliftBooster

__all__ = [
    "PerpetualBooster",
    "compute_calibration_curve",
    "expected_calibration_error",
    "PerpetualClassifier",
    "PerpetualRegressor",
    "PerpetualRanker",
    "UpliftBooster",
    "BraidedBooster",
    "PerpetualRiskEngine",
    "SLearner",
    "TLearner",
    "XLearner",
    "DRLearner",
    "DMLEstimator",
    "FairClassifier",
    "PolicyLearner",
    "auuc",
    "cumulative_gain_curve",
    "qini_curve",
    "qini_coefficient",
]
