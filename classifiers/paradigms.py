"""
References:
    - https://scikit-learn.org/stable/modules/model_evaluation.html
    - https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
    - https://moabb.neurotechx.com/docs/generated/moabb.paradigms.LeftRightImagery.html
"""

import numpy as np
from sklearn.metrics import log_loss, make_scorer
from moabb.paradigms import LeftRightImagery


def nll_score(y_true, y_proba):
    return log_loss(y_true, y_proba, labels=np.unique(y_true))


class LogLossLeftRightImagery(LeftRightImagery):
    @property
    def scoring(self):
        return make_scorer(nll_score, greater_is_better=False)
