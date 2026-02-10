"""
Make pipeline for TS+LR.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.pipelines import PipelineBase
from src.pipelines.classifiers import LogisticRegression


class TSLR(PipelineBase):
    def build(self):
        return {
            "TSLR": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000),
            )
        }
