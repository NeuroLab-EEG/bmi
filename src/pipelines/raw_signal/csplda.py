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
from src.pipelines.pipeline import Pipeline


class CSPLDA(Pipeline):
    def pipeline(self):
        return {
            "csplda": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                LDA(solver="svd"),
            )
        }

    def params(self):
        return {}
