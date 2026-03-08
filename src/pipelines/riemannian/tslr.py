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
from ..pipeline_base import PipelineBase
from ..classifiers import LogisticRegression, CuMLSubprocessor


class TSLR(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                CuMLSubprocessor(estimator=LogisticRegression(C=1.0, max_iter=1000), root_dir=self.data_path),
            )
        }
