"""
Make pipeline for Bayesian DCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
"""

from sklearn.pipeline import make_pipeline
from src.pipelines import PipelineBase
from src.pipelines.classifiers import DeepCNN, BayesianNeuralNetwork


class BDCNN(PipelineBase):
    def build(self):
        return {
            "BDCNN": make_pipeline(
                BayesianNeuralNetwork(
                    random_state=self.random_state,
                    network=DeepCNN(
                        n_features=self.n_features,
                        n_classes=self.n_classes,
                        n_times=self.n_times,
                        random_state=self.random_state,
                    ),
                )
            )
        }
