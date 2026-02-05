"""
Make CUDA accelerated support vector machine classifier.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
.. [2] https://docs.rapids.ai/api/cuml/stable/api/#cuml.svm.SVC
"""

import cupy as cp
from cuml.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


class SVC(ClassifierMixin, BaseEstimator):
    def __init__(self, **svc_params):
        self.svc_params = svc_params
        self.model_ = None

    def fit(self, X, y):
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        self.model_ = SVC(**self.svc_params)
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
        return self.svc_params

    def set_params(self, **params):
        self.svc_params.update(params)
        return self
