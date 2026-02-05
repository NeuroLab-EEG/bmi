"""
Make pipeline for TS + Bayesian LR.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pipelines import Pipeline
from src.pipelines.classifiers import BayesianLogisticRegression


class TSBLR(Pipeline):
    def build(self):
        return {
            "TSBLR": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                PCA(n_components=0.95),
                BayesianLogisticRegression(random_state=self.random_state),
            )
        }
