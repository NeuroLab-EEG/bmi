"""
Base class for all machine learning pipelines.
"""

from abc import ABC, abstractmethod


class PipelineBase(ABC):
    def __init__(self, data_path=None, random_state=None, n_features=None, n_classes=None, n_timepoints=None):
        self.data_path = data_path
        self.random_state = random_state
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_timepoints = n_timepoints

    @abstractmethod
    def build(self):
        pass
