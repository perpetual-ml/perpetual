"""Adverse-action reason-code generation for credit risk models.

Provides `PerpetualRiskEngine`, a wrapper around a fitted `PerpetualBooster`
that explains rejection decisions by identifying the features most responsible
for pushing a score below the approval threshold.
"""

from typing import List

import numpy as np

from perpetual.booster import PerpetualBooster


class PerpetualRiskEngine:
    """
    Risk Engine for generating Adverse Action (Reason) Codes.

    This engine wraps a fitted PerpetualBooster model and provides functionality
    to explain rejections (Adverse Actions) by attributing the negative decision
    to specific features.
    """

    def __init__(self, model: PerpetualBooster):
        """Wrap a fitted booster for reason-code generation.

        Parameters
        ----------
        model : PerpetualBooster
            A fitted `PerpetualBooster` instance.
        """
        self.booster = model

    def generate_reason_codes(
        self,
        X,
        threshold: float,
        n_codes: int = 3,
        method: str = "Average",
        rejection_direction: str = "lower",
    ) -> List[List[str]]:
        """
        Generate reason codes for samples that fall below/above the approval threshold.

        Logic:
        1. Predict score for X.
        2. Identify rejected samples based on ``rejection_direction``:
           - "lower": score < threshold (e.g. FICO score)
           - "higher": score > threshold (e.g. Default probability)
        3. Identify top N features dragging the score in the rejected direction.

        Parameters
        ----------
        X : array-like
            Applicant data.
        threshold : float
            Approval threshold.
        n_codes : int, default=3
            Number of reason codes to return per applicant.
        method : str, default="Average"
            Contribution method.
        rejection_direction : {"lower", "higher"}, default="lower"
            Direction of rejection. If "lower", scores below threshold are
            rejected. If "higher", scores above threshold are rejected.

        Returns
        -------
        reasons : list of list of str
            For each sample, a list of reason-code strings.
            Approved samples get an empty list.
        """
        if rejection_direction not in ("lower", "higher"):
            raise ValueError("rejection_direction must be 'lower' or 'higher'")

        # Get predictions
        if len(getattr(self.booster, "classes_", [])) == 2:
            # For binary classification, use probability of positive class
            preds = self.booster.predict_proba(X)[:, 1]
        else:
            preds = self.booster.predict(X)

        # Get contributions (SHAP values)
        # shape: (n_samples, n_features + 1) -> last column is bias
        contributions = self.booster.predict_contributions(X, method=method)

        # Remove bias column for feature attribution logic
        feature_contribs = contributions[:, :-1]

        reasons = []
        feature_names = getattr(self.booster, "feature_names_in_", None)

        for i, score in enumerate(preds):
            sample_reasons = []
            is_rejected = (
                (score < threshold)
                if rejection_direction == "lower"
                else (score > threshold)
            )

            if is_rejected:
                # Identify contributors
                row_contribs = feature_contribs[i]

                if rejection_direction == "lower":
                    # Lowest values (most negative) drag score DOWN
                    indices = np.argsort(row_contribs)[:n_codes]
                else:
                    # Highest values (most positive) push score UP
                    indices = np.argsort(row_contribs)[-n_codes:][::-1]

                for idx in indices:
                    val = row_contribs[idx]
                    name = feature_names[idx] if feature_names else str(idx)
                    sample_reasons.append(f"{name}: {val:.4f}")

            reasons.append(sample_reasons)

        return reasons
