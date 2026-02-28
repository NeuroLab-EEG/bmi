from .cuml import LogisticRegression, SVC
from .neural_network import ShallowCNN, DeepCNN
from .model_builder import (
    BayesianLogisticRegression,
    BayesianLinearDiscriminantAnalysis,
    GaussianProcess,
    BayesianNeuralNetwork,
)

__all__ = [
    LogisticRegression.__name__,
    SVC.__name__,
    ShallowCNN.__name__,
    DeepCNN.__name__,
    BayesianLogisticRegression.__name__,
    BayesianLinearDiscriminantAnalysis.__name__,
    GaussianProcess.__name__,
    BayesianNeuralNetwork.__name__,
]
