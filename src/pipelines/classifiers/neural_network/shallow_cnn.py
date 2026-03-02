"""
Make shallow CNN classifier.

References
----------
.. [1] https://braindecode.org/stable/generated/braindecode.models.ShallowFBCSPNet.html
"""

from braindecode.models import ShallowFBCSPNet
from .neural_network_base import NeuralNetworkBase


class ShallowCNN(NeuralNetworkBase):
    def __init__(self, n_features=None, n_classes=None, n_timepoints=None, random_state=None):
        super().__init__(
            ShallowFBCSPNet(
                n_chans=n_features,
                n_outputs=n_classes,
                n_times=n_timepoints,
            ),
            random_state=random_state,
        )
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_timepoints = n_timepoints
        self.random_state = random_state
