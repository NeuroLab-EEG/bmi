"""
Make pipeline for Bayesian DCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
"""

from os import path, makedirs
from sklearn.pipeline import make_pipeline
from src.pipelines import PipelineBase
from src.pipelines.classifiers import DeepCNN, BayesianNeuralNetwork


class BDCNN(PipelineBase):
    def build(self):
        classname = self.__class__.__name__
        data_path = path.join(self.data_path, classname)
        makedirs(data_path, exist_ok=True)
        return {
            classname: make_pipeline(
                BayesianNeuralNetwork(
                    data_path=data_path,
                    random_state=self.random_state,
                    network=DeepCNN(
                        n_features=self.n_features,
                        n_classes=self.n_classes,
                        n_timepoints=self.n_timepoints,
                        random_state=self.random_state,
                    ),
                )
            )
        }
