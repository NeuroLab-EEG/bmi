"""
Make pipeline for TS+SVM.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.pipelines import PipelineBase


class TSSVM(PipelineBase):
    def build(self):
        classname = self.__class__.__name__
        return {
            classname: make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                SVC(C=1.0, kernel="linear", probability=True, random_state=self.random_state),
            )
        }
