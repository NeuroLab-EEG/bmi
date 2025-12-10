"""
References:
    - https://scikit-learn.org/stable/modules/model_evaluation.html
    - https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss  # noqa: E501
    - https://moabb.neurotechx.com/docs/generated/moabb.paradigms.LeftRightImagery.html
"""

import numpy as np
from sklearn.metrics import get_scorer
from moabb.paradigms import LeftRightImagery

class LogLossLeftRightImagery(LeftRightImagery):
    @property
    def scoring(self):
        return get_scorer("neg_log_loss")
