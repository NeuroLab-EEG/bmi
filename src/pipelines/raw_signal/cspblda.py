"""
Make pipeline for CSP + Bayesian LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..pipeline_base import PipelineBase
from ..classifiers import (
    BayesianLinearDiscriminantAnalysis as BayesianLDA,
    PyMCSubprocessor,
)


class CSPBLDA(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                PyMCSubprocessor(
                    estimator=BayesianLDA(random_state=self.random_state),
                    root_dir=self.data_path,
                ),
            )
        }
