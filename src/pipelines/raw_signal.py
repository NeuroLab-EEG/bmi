"""
Build pipelines with Euclidean spatial filters
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
    - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC  # noqa: E501
"""

from os import getenv
from dotenv import load_dotenv
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC


# Load environment variables
load_dotenv()
random_state = int(getenv("RANDOM_STATE"))


def csp_lda():
    return {
        "CSP+LDA": make_pipeline(
            Covariances(estimator="oas"),
            CSP(nfilter=6),
            LDA(solver="svd"),
        )
    }, {}


def csp_svm():
    return {
        "CSP+SVM": make_pipeline(
            Covariances(estimator="oas"),
            CSP(nfilter=6),
            SVC(kernel="linear", probability=True, random_state=random_state),
        )
    }, {
        "CSP+SVM": {
            "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
            "svc__C": [0.5, 1, 1.5],
            "svc__kernel": ["rbf", "linear"],
        }
    }
