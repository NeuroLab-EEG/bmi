"""
Produce probability graph model.

References
----------
.. [1] https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model_graph.model_to_graphviz.html
"""

from src.pipelines.classifiers import (
    BayesianLogisticRegression,
    BayesianLinearDiscriminantAnalysis,
    GaussianProcess,
    BayesianNeuralNetwork,
)


class PGM:
    def run(self):
        BayesianLogisticRegression().graph_model()
        BayesianLinearDiscriminantAnalysis().graph_model()
        BayesianNeuralNetwork().graph_model()
        GaussianProcess.from_kernel(kernel="linear").graph_model()
        GaussianProcess.from_kernel(kernel="rbf").graph_model()
