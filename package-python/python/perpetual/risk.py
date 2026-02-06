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
        self, X, threshold: float, n_codes: int = 3, method: str = "Weight"
    ) -> List[List[str]]:
        """
        Generate reason codes for samples that fall below the approval threshold.

        Logic:
        1. Predict score for X.
        2. If score < threshold (Rejection), calculate feature contributions.
        3. Identify top N features dragging the score DOWN.

        Parameters
        ----------
        X : array-like
            Applicant data.
        threshold : float
            Approval threshold. Scores below this are considered rejections.
        n_codes : int
            Number of reason codes to return per applicant.
        method : str
            Contribution method. "Weight" is standard SHAP-like contribution.

        Returns
        -------
        reasons : list of list of str
            For each sample, a list of reason-code strings (e.g.
            ``"feature_name: -0.1234"``).  Approved samples get an empty list.
        """
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
            if score < threshold:
                # Identify negative contributors
                # We want the features with the lowest (most negative) impact
                # OR the features that differ most from the "ideal" applicant?
                # Standard approach: Features pushing score DOWN the most.

                row_contribs = feature_contribs[i]

                # Sort indices by contribution value (ascending)
                # Lowest values (most negative) first
                sorted_indices = np.argsort(row_contribs)

                # Take bottom N
                top_negative_indices = sorted_indices[:n_codes]

                for idx in top_negative_indices:
                    # Only include if contribution is actually negative?
                    # Risk models usually imply 0 is neutral.
                    # If all contribs are positive but score is low, it's tricky.
                    # Standard implementation takes lowest algebraic values.

                    val = row_contribs[idx]
                    name = feature_names[idx] if feature_names else str(idx)
                    sample_reasons.append(f"{name}: {val:.4f}")

            reasons.append(sample_reasons)

        return reasons
