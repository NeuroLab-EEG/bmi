"""
Make pipeline for TS+SVM.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from src.pipelines import Pipeline


class TSSVM(Pipeline):
    def pipeline(self):
        return {
            "TSSVM": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                SVC(kernel="linear", probability=True, random_state=self.random_state),
            )
        }

    def params(self):
        return {"TSSVM": {"svc__C": [0.5, 1, 1.5], "svc__kernel": ["rbf", "linear"]}}
