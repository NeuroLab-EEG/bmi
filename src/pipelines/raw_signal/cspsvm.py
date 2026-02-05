"""
Make pipeline for CSP+SVM.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pipelines import Pipeline
from src.pipelines.classifiers import SVC


class CSPSVM(Pipeline):
    def build(self):
        return {
            "CSPSVM": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                StandardScaler(),
                PCA(n_components=0.95),
                SVC(C=1.0, kernel="rbf", probability=True, random_state=self.random_state),
            )
        }
