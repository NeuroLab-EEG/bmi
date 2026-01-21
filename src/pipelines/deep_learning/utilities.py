"""
Utilities shared across deep learning pipelines.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ToFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)
