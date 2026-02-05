"""
Make pipeline for TS+GP.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pipelines.pipeline import Pipeline
from src.pipelines.models import GaussianProcess


class TSGP(Pipeline):
    def pipeline(self):
        return {
            "TSGP": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                PCA(n_components=0.95),
                GaussianProcess(random_state=self.random_state),
            )
        }

    def params(self):
        return {}
