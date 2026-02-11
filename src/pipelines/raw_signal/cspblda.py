"""
Make pipeline for CSP + Bayesian LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from os import path, makedirs
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.pipelines import PipelineBase
from src.pipelines.classifiers import BayesianLinearDiscriminantAnalysis as BayesianLDA


class CSPBLDA(PipelineBase):
    def build(self):
        data_path = path.join(self.data_path, self.__class__.__name__)
        makedirs(data_path, exist_ok=True)
        return {
            "CSPBLDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                BayesianLDA(data_path=data_path, random_state=self.random_state),
            )
        }
