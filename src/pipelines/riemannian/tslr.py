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
from sklearn.linear_model import LogisticRegression
from src.pipelines import PipelineBase


class TSLR(PipelineBase):
    def build(self):
        classname = self.__class__.__name__
        return {
            classname: make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000),
            )
        }
