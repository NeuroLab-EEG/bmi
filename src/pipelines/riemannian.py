"""
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def ts_lr():
    return {
        "ts_lr": Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("lr", LogisticRegression(C=1.0)),
            ]
        )
    }, {}


def ts_svm():
    return {
        "ts_svm": Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("ts", TangentSpace(metric="riemann")),
                ("svc", SVC(kernel="linear")),
            ]
        )
    }, {"ts_svm": {"svc__C": [0.5, 1, 1.5], "svc__kernel": ["rbf", "linear"]}}
