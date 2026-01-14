"""
Build pipelines with Euclidean spatial filters.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
.. [2] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
.. [3] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
.. [4] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC  # noqa: E501
"""

from os import getenv
from dotenv import load_dotenv
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from src.pipelines.pipeline import Pipeline


# Load environment variables
load_dotenv()
random_state = int(getenv("RANDOM_STATE"))


class CSPLDA(Pipeline):
    def pipeline(self):
        return {
            "CSP+LDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                LDA(solver="svd"),
            )
        }

    def params(self):
        return {}


class CSPSVM(Pipeline):
    def pipeline(self):
        return {
            "CSP+SVM": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                SVC(kernel="linear", probability=True, random_state=random_state),
            )
        }

    def params(self):
        return {
            "CSP+SVM": {
                "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
                "svc__C": [0.5, 1, 1.5],
                "svc__kernel": ["rbf", "linear"],
            }
        }
