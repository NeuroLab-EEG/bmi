"""
Make pipeline for CSP+LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from src.pipelines import Pipeline


class CSPLDA(Pipeline):
    def pipeline(self):
        return {
            "CSPLDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                PCA(n_components=0.95),
                LDA(solver="svd"),
            )
        }

    def params(self):
        return {}
