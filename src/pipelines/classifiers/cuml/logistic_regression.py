"""
Make CUDA accelerated logistic regression classifier.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
.. [2] https://docs.rapids.ai/api/cuml/stable/api/#logistic-regression
"""

from cuml import LogisticRegression as CuMLLogisticRegression
from src.pipelines.classifiers.cuml import CuMLBase


class LogisticRegression(CuMLBase):
    def __init__(self, **kwargs):
        super().__init__(CuMLLogisticRegression(**kwargs))
