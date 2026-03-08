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
from ..pipeline_base import PipelineBase
from ..classifiers import BayesianLogisticRegression, PyMCSubprocessor


class TSBLR(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                PyMCSubprocessor(
                    estimator=BayesianLogisticRegression(random_state=self.random_state),
                    root_dir=self.data_path,
                ),
            )
        }
