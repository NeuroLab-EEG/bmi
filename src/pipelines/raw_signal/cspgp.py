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
from sklearn.decomposition import PCA
from src.pipelines.pipeline import Pipeline
from src.pipelines.models import GaussianProcess


class CSPGP(Pipeline):
    def pipeline(self):
        return {
            "CSPGP": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                PCA(n_components=0.95),
                GaussianProcess(),
            )
        }

    def params(self):
        return {}
