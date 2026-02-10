"""
Make pipeline for CSP+GP.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.pipelines import PipelineBase
from src.pipelines.classifiers import GaussianProcess


class CSPGP(PipelineBase):
    def build(self):
        return {
            "CSPGP": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                GaussianProcess(kernel="rbf", random_state=self.random_state),
            )
        }
