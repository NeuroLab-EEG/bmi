"""
Base class for all ML pipelines.
"""

from abc import ABC, abstractmethod


class Pipeline(ABC):
    @abstractmethod
    def pipeline(self):
        pass

    @abstractmethod
    def params(self):
        pass
