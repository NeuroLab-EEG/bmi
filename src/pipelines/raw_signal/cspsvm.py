"""
Make pipeline for CSP+SVM.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""

from os import getenv
from dotenv import load_dotenv
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from src.pipelines.pipeline import Pipeline


class CSPSVM(Pipeline):
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))

    def pipeline(self):
        return {
            "cspsvm": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                SVC(kernel="linear", probability=True, random_state=self.random_state),
            )
        }

    def params(self):
        return {
            "cspsvm": {
                "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
                "svc__C": [0.5, 1, 1.5],
                "svc__kernel": ["rbf", "linear"],
            }
        }
