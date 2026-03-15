"""
Make pipeline for DCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
"""

from sklearn.pipeline import make_pipeline
from ..pipeline_base import PipelineBase
from ..classifiers import DeepConvNet, PyTorchSubprocessor


class DCNN(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                PyTorchSubprocessor(
                    estimator=DeepConvNet(
                        n_features=self.n_features,
                        n_classes=self.n_classes,
                        n_timepoints=self.n_timepoints,
                        random_state=self.random_state,
                    ),
                    root_dir=self.data_path,
                )
            )
        }
