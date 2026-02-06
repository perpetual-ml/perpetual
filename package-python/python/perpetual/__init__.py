"""Perpetual — a self-generalizing Gradient Boosting Machine.

Provides:
  - :class:`PerpetualBooster` — single-output gradient booster.
  - :class:`PerpetualClassifier`, :class:`PerpetualRegressor`, :class:`PerpetualRanker` — scikit-learn wrappers.
  - :class:`MultiOutputBooster` — multi-target booster (via internal import).
  - :class:`UpliftBooster` — R-Learner uplift estimator.
  - :class:`BraidedBooster` — Instrumental Variable (BoostIV) estimator.
  - :class:`SLearner`, :class:`TLearner`, :class:`XLearner`, :class:`DRLearner` — meta-learner CATE estimators.
  - :class:`PerpetualRiskEngine` — adverse-action reason-code generator.
  - :mod:`causal_metrics` — AUUC, Qini coefficient, cumulative gain curves.
"""

from __future__ import annotations

from perpetual.booster import PerpetualBooster
from perpetual.causal_metrics import (
    auuc,
    cumulative_gain_curve,
    qini_coefficient,
    qini_curve,
)
from perpetual.iv import BraidedBooster
from perpetual.meta_learners import DRLearner, SLearner, TLearner, XLearner
from perpetual.risk import PerpetualRiskEngine
from perpetual.sklearn import PerpetualClassifier, PerpetualRanker, PerpetualRegressor
from perpetual.uplift import UpliftBooster

__all__ = [
    "PerpetualBooster",
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
    "auuc",
    "cumulative_gain_curve",
    "qini_curve",
    "qini_coefficient",
]
