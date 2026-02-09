"""
Make shallow CNN classifier.

References
----------
.. [1] https://braindecode.org/stable/generated/braindecode.models.ShallowFBCSPNet.html
"""

from braindecode.models import ShallowFBCSPNet
from src.pipeline.classifiers.neural_network import NeuralNetworkBase


class ShallowCNN(NeuralNetworkBase):
    def __init__(self, n_features=None, n_classes=None, n_times=None, random_state=None):
        super().__init__(
            ShallowFBCSPNet(
                n_chans=n_features,
                n_outputs=n_classes,
                n_times=n_times,
            ),
            random_state=random_state,
        )
