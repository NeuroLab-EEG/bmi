"""
Utilities shared across deep learning pipelines.
"""

class ToFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)
