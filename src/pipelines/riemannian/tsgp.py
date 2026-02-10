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
from src.pipelines import PipelineBase
from src.pipelines.classifiers import GaussianProcess


class TSGP(PipelineBase):
    def build(self):
        return {
            "TSGP": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                GaussianProcess(kernel="linear", random_state=self.random_state),
            )
        }
