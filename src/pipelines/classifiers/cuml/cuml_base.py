"""
Make CUDA accelerated scikit-learn classifier.

References
----------
.. [1] https://docs.rapids.ai/api/cuml/stable/
.. [2] https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html#cupy.cuda.MemoryPool.free_all_blocks
.. [3] https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.PinnedMemoryPool.html#cupy.cuda.PinnedMemoryPool.free_all_blocks
"""

import gc
import cupy as cp
from sklearn.base import BaseEstimator, ClassifierMixin


class CuMLBase(ClassifierMixin, BaseEstimator):
    def __init__(self, classifier):
        self.classifier = classifier
        self.model_ = None

    def fit(self, X, y):
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        self.model_ = self.classifier
        self.model_.fit(X_gpu, y_gpu)
        self._cleanup_gpu()
        classes = self.model_.classes_
        self.classes_ = classes.get() if hasattr(classes, "get") else classes
        return self

    def predict(self, X):
        X_gpu = cp.asarray(X)
        pred = self.model_.predict(X_gpu)
        self._cleanup_gpu()
        result = pred.get() if hasattr(pred, "get") else pred
        return result

    def predict_proba(self, X):
        X_gpu = cp.asarray(X)
        proba = self.model_.predict_proba(X_gpu)
        self._cleanup_gpu()
        result = proba.get() if hasattr(proba, "get") else proba
        return result

    def _cleanup_gpu(self):
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
