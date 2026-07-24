"""
Make pipeline for CSP+LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..classifiers import SklearnSubprocessor
from ..pipeline_base import PipelineBase


class CSPLDA(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                SklearnSubprocessor(estimator=LDA(solver="svd"), root_dir=self.data_path),
            )
        }
