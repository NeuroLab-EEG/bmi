"""
Make pipeline for CSP+GP.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
"""

from os import path, makedirs
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..pipeline_base import PipelineBase
from ..classifiers import GaussianProcess


class CSPGP(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                GaussianProcess(kernel="rbf", data_path=data_path, random_state=self.random_state),
            )
        }
