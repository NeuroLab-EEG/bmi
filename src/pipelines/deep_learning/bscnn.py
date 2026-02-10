"""
Make pipeline for Bayesian SCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
"""

from sklearn.pipeline import make_pipeline
from src.pipelines import PipelineBase
from src.pipelines.classifiers import ShallowCNN, BayesianNeuralNetwork


class BSCNN(PipelineBase):
    def build(self):
        return {
            "BSCNN": make_pipeline(
                BayesianNeuralNetwork(
                    random_state=self.random_state,
                    network=ShallowCNN(
                        n_features=self.n_features,
                        n_classes=self.n_classes,
                        n_times=self.n_times,
                        random_state=self.random_state,
                    ),
                )
            )
        }
