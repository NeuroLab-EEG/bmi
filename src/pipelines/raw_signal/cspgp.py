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
from ..pipeline_base import PipelineBase
from ..classifiers import RBFGP, PyMCSubprocessor


class CSPGP(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                PyMCSubprocessor(
                    estimator=RBFGP(
                        random_state=self.random_state
                    ),
                    save_dir=self.data_path
                )
            )
        }
