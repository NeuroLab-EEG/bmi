"""
Base class for all ML pipelines.
"""

from os import getenv
from dotenv import load_dotenv
from abc import ABC, abstractmethod


class Pipeline(ABC):
    def __init__(self, n_features=None, n_classes=None, n_times=None):
        # Load environment variables
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))

        # Define dataset dimensions
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_times = n_times

    @abstractmethod
    def pipeline(self):
        pass

    @abstractmethod
    def params(self):
        pass
