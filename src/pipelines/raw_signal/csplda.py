"""
Make pipeline for CSP+LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from src.pipelines import PipelineBase


class CSPLDA(PipelineBase):
    def build(self):
        return {
            "CSPLDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                LDA(solver="svd"),
            )
        }
