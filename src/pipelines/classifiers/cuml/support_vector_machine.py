"""
Make CUDA accelerated support vector machine classifier.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
.. [2] https://docs.rapids.ai/api/cuml/stable/api/#cuml.svm.SVC
"""

import gc
import cupy as cp
from cuml.svm import SVC as CuMLSVC
from src.pipelines.classifiers.cuml import CuMLBase


class _CleaningSVC(CuMLSVC):
    def fit(self, X, y, **kwargs):
        results = super().fit(X, y, **kwargs)
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        return results


class SVC(CuMLBase):
    def __init__(self, **kwargs):
        super().__init__(_CleaningSVC(**kwargs))
