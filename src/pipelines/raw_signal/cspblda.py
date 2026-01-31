"""
Make pipeline for CSP + Bayesian LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from src.pipelines import Pipeline
from src.pipelines.models import BayesianLinearDiscriminantAnalasis as BayesianLDA


class CSPBLDA(Pipeline):
    def pipeline(self):
        return {
            "CSPBLDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                BayesianLDA(),
            )
        }

    def params(self):
        return {}
