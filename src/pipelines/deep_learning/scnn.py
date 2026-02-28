"""
Make pipeline for SCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
"""

from sklearn.pipeline import make_pipeline
from ..pipeline_base import PipelineBase
from ..classifiers import ShallowCNN


class SCNN(PipelineBase):
    def build(self):
        return {
            self.__class__.__name__: make_pipeline(
                ShallowCNN(
                    n_features=self.n_features,
                    n_classes=self.n_classes,
                    n_timepoints=self.n_timepoints,
                    random_state=self.random_state,
                ),
            ),
        }
