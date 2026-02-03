"""
Make pipeline for TS + Bayesian LR.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from src.pipelines import Pipeline
from src.pipelines.models import BayesianLogisticRegression


class TSBLR(Pipeline):
    def pipeline(self):
        return {
            "TSBLR": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                BayesianLogisticRegression(),
            )
        }

    def params(self):
        return {}
