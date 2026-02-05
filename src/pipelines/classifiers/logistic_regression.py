"""
Make CUDA accelerated logistic regression classifier.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
.. [2] https://docs.rapids.ai/api/cuml/stable/api/#logistic-regression
"""

import cupy as cp
from cuml import LogisticRegression as CuMLLogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, **params):
        self.params = params
        self.model_ = None

    def fit(self, X, y):
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        self.model_ = CuMLLogisticRegression(**self.params)
        self.model_.fit(X_gpu, y_gpu)
        classes = self.model_.classes_
        self.classes_ = classes.get() if hasattr(classes, "get") else classes
        return self

    def predict(self, X):
        X_gpu = cp.asarray(X)
        pred = self.model_.predict(X_gpu)
        result = pred.get() if hasattr(pred, "get") else pred
        return result

    def predict_proba(self, X):
        X_gpu = cp.asarray(X)
        proba = self.model_.predict_proba(X_gpu)
        result = proba.get() if hasattr(proba, "get") else proba
        return result

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self
