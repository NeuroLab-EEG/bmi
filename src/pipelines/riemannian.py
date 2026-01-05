"""
Build pipelines with Riemannian spatial filters.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
.. [2] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
.. [3] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
.. [4] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC  # noqa: E501
"""

from os import getenv
from dotenv import load_dotenv
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Load environment variables
load_dotenv()
random_state = int(getenv("RANDOM_STATE"))


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
            SVC(kernel="linear", probability=True, random_state=random_state),
        )
    }, {"TS+SVM": {"svc__C": [0.5, 1, 1.5], "svc__kernel": ["rbf", "linear"]}}
