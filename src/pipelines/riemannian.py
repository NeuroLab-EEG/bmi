"""
Build pipelines with Riemannian spatial filters
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def ts_lr():
    return {
        "TS+LR": make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            LogisticRegression(C=1.0),
        )
    }, {}


def ts_svm():
    return {
        "TS+SVM": make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            SVC(kernel="linear", probability=True),
        )
    }, {"TS+SVM": {"svc__C": [0.5, 1, 1.5], "svc__kernel": ["rbf", "linear"]}}
