"""
Make CUDA accelerated support vector machine classifier.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
.. [2] https://docs.rapids.ai/api/cuml/stable/api/#cuml.svm.SVC
"""

from cuml.svm import SVC as CuMLSVC
from src.pipelines.classifiers.cuml import CuMLBase


class SVC(CuMLBase):
    def __init__(self, **kwargs):
        super().__init__(CuMLSVC(**kwargs))
