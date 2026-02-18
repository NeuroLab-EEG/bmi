"""
Customize paradigm scoring rule.

References
----------
.. [1] https://scikit-learn.org/stable/modules/model_evaluation.html
.. [2] https://moabb.neurotechx.com/docs/generated/moabb.paradigms.LeftRightImagery.html
"""

from sklearn.metrics import make_scorer, matthews_corrcoef
from netcal.metrics import ECE
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

    def _ece_score(self, y_true, y_prob):
        return ECE().measure(y_prob, y_true)
