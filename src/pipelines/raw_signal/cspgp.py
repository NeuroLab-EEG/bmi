"""
Make pipeline for CSP+GP.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
.. [2] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html#example-2-classification
.. [3] https://doi.org/10.7551/mitpress/3206.001.0001
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.pipelines.pipeline import Pipeline
from src.pipelines.models import GaussianProcess


class CSPGP(Pipeline):
    def pipeline(self):
        return {
            "CSPGP": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                GaussianProcess(),
            )
        }

    def params(self):
        return {}
