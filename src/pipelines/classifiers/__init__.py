from .cuml import LogisticRegression, SVC
from .bayesian_logistic_regression import BayesianLogisticRegression
from .bayesian_linear_discriminant_analysis import BayesianLinearDiscriminantAnalasis
from .gaussian_process import GaussianProcess

__all__ = [
    "LogisticRegression",
    "SVC",
    "BayesianLogisticRegression",
    "BayesianLinearDiscriminantAnalasis",
    "GaussianProcess",
]
