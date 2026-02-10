"""
Build Bayesian neural network classifier.

References
----------
.. [1] http://probml.github.io/book2
.. [2] https://www.pymc.io/projects/examples/en/latest/variational_inference/bayesian_neural_network_advi.html
.. [3] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [4] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import torch
import pandas as pd
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler
from src.pipelines.classifiers.model_builder import ModelBuilderBase


class BayesianNeuralNetwork(ModelBuilderBase):
    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.scaler = StandardScaler()

    def build_model(self, X, y):
        with pm.Model() as self.model:
            x_data = pm.Data("x_data", X)
            y_data = pm.Data("y_data", y)

            # Define priors
            w = pm.Normal(
                "w",
                mu=self.model_config["w_mu"],
                sigma=self.model_config["w_sigma"],
                shape=X.shape[1],
            )
            b = pm.Normal(
                "b",
                mu=self.model_config["b_mu"],
                sigma=self.model_config["b_sigma"],
            )

            # Define likelihood
            logit = pm.math.dot(x_data, w) + b
            pm.Bernoulli(self.output_var, logit_p=logit, observed=y_data)

    def fit(self, X, y):
        # Train neural network
        self.network.fit(X, y)
        modules = list(self.network.model_.module_.children())

        # Extract features from backbone
        self.backbone = torch.nn.Sequential(*modules[:-1])
        self.backbone.eval()
        X_features = self._extract_features(X)
        X_features_scaled = self.scaler.fit_transform(X_features)

        # Sample posterior of classification parameters
        return super().fit(X_features_scaled, y)

    def _extract_features(self, X):
        X_tensor = torch.from_numpy(X).float()
        device = next(self.backbone.parameters()).device
        X_tensor = X_tensor.to(device)
        with torch.no_grad():
            X_features = self.backbone(X_tensor).flatten(start_dim=1).cpu().numpy()
        return X_features

    def predict_proba(self, X):
        X_features = self._extract_features(X)
        X_features_scaled = self.scaler.transform(X_features)
        return super().predict_proba(X_features_scaled)

    @staticmethod
    def get_default_model_config():
        return {
            "w_mu": 0,
            "w_sigma": 1.0,
            "b_mu": 0,
            "b_sigma": 3.0,
        }
