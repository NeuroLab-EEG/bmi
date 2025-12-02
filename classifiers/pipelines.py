"""
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
"""

from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC


csp_lda_pipeline = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("csp", CSP(nfilter=6)),
    ("lda", LDA(solver="svd"))
])

csp_svm_pipeline = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("csp", CSP(nfilter=6)),
    ("svc", SVC(kernel="linear"))
])

csp_svm_grid = {
    "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
    "svc__C": [0.5, 1, 1.5],
    "svc__kernel": ["rbf", "linear"]
}


def pipelines():
    return {
        "csp_lda": csp_lda_pipeline,
        "csp_svm": csp_svm_pipeline
    }


def grids():
    return {
        "csp_svm": csp_svm_grid
    }
