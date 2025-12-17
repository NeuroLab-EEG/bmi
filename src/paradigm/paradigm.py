"""
Customize paradigm scoring rule
References:
    - https://scikit-learn.org/stable/modules/model_evaluation.html
    - https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss  # noqa: E501
    - https://moabb.neurotechx.com/docs/generated/moabb.paradigms.LeftRightImagery.html
"""

from sklearn.metrics import log_loss
from moabb.paradigms import LeftRightImagery


def nll_score(estimator, X, y_true):
    y_prob = estimator.predict_proba(X)
    return -log_loss(y_true, y_prob)


class LogLossLeftRightImagery(LeftRightImagery):
    @property
    def scoring(self):
        return nll_score
