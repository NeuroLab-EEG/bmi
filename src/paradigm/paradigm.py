"""
Customize paradigm scoring rule.

References
----------
.. [1] https://scikit-learn.org/stable/modules/model_evaluation.html
.. [2] https://moabb.neurotechx.com/docs/generated/moabb.paradigms.LeftRightImagery.html
"""

import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef
from moabb.paradigms import LeftRightImagery


class MultiScoreLeftRightImagery(LeftRightImagery):
    @property
    def scoring(self):
        return {
            "nll": "neg_log_loss",
            "brier": "neg_brier_score",
            "acc": "accuracy",
            "auroc": "roc_auc",
            "mcc": make_scorer(matthews_corrcoef, response_method="predict"),
            "ece": make_scorer(self._ece_score, response_method="predict_proba", greater_is_better=False),
        }

    def _ece_score(self, y_true, y_prob, n_bins=10):
        """
        The expected calibration error (ECE) for binary classification.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            weight = np.mean(in_bin)
            if weight > 0:
                positive = np.mean(y_true[in_bin])
                confidence = np.mean(y_prob[in_bin])
                ece += np.abs(positive - confidence) * weight

        return ece
