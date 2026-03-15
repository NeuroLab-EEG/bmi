"""
Make pipeline for Bayesian SCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
"""

from sklearn.pipeline import make_pipeline
from ..pipeline_base import PipelineBase
from ..classifiers import (
    ShallowConvNet,
    BayesianNeuralNetwork,
    BNNPyMCSubprocessor,
    BNNPyTorchSubprocessor,
)


class BSCNN(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                BNNPyMCSubprocessor(
                    estimator=BayesianNeuralNetwork(
                        random_state=self.random_state,
                        network=BNNPyTorchSubprocessor(
                            estimator=ShallowConvNet(
                                n_features=self.n_features,
                                n_classes=self.n_classes,
                                n_timepoints=self.n_timepoints,
                                random_state=self.random_state,
                            ),
                            root_dir=self.data_path,
                        ),
                    ),
                    root_dir=self.data_path,
                )
            )
        }
