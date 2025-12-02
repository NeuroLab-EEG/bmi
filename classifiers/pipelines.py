"""
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def csp_lda():
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("csp", CSP(nfilter=6)),
        ("lda", LDA(solver="svd"))
    ])
