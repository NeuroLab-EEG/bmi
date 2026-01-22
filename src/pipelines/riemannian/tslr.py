"""
Make pipeline for TS+LR.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from src.pipelines.pipeline import Pipeline


class TSLR(Pipeline):
    def pipeline(self):
        return {
            "TSLR": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                LogisticRegression(C=1.0, max_iter=1000),
            )
        }

    def params(self):
        return {}
