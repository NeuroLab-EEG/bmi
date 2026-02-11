"""
Make pipeline for TS + Bayesian LR.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
"""

from os import path, makedirs
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.pipelines import PipelineBase
from src.pipelines.classifiers import BayesianLogisticRegression


class TSBLR(PipelineBase):
    def build(self):
        classname = self.__class__.__name__
        data_path = path.join(self.data_path, classname)
        makedirs(data_path, exist_ok=True)
        return {
            classname: make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                StandardScaler(),
                BayesianLogisticRegression(data_path=data_path, random_state=self.random_state),
            )
        }
