"""
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC


csp_lda = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("csp", CSP(nfilter=6)),
    ("lda", LDA(solver="svd"))
])

csp_svm = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("csp", CSP(nfilter=6)),
    ("svc", SVC(kernel="linear"))
])

csp_svm_params = {
    "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
    "svc__C": [0.5, 1, 1.5],
    "svc__kernel": ["rbf", "linear"]
}
