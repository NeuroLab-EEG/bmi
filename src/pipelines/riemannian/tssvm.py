"""
Make pipeline for TS+SVM.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pipelines import PipelineBase
from src.pipelines.classifiers import SVC


class TSSVM(PipelineBase):
    def build(self):
        return {
            "TSSVM": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                PCA(n_components=0.95),
                SVC(C=1.0, kernel="rbf", probability=True, random_state=self.random_state),
            )
        }
