"""
Make pipeline for CSP + Bayesian LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pipelines import PipelineBase
from src.pipelines.classifiers import BayesianLinearDiscriminantAnalasis as BayesianLDA


class CSPBLDA(PipelineBase):
    def build(self):
        return {
            "CSPBLDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                PCA(n_components=0.95),
                BayesianLDA(progressbar=False, random_state=self.random_state),
            )
        }
