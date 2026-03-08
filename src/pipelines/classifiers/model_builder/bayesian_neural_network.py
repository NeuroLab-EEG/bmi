"""
Build Bayesian neural network classifier.

References
----------
.. [1] http://probml.github.io/book2
.. [2] https://bayesiancomputationbook.com/markdown/chp_04.html#posterior-geometry-matters
.. [3] https://www.pymc.io/projects/examples/en/latest/variational_inference/bayesian_neural_network_advi.html
.. [4] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [5] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import torch
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
from .model_builder_base import ModelBuilderBase


class BayesianNeuralNetwork(ModelBuilderBase):
    def __init__(self, network=None, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.scaler = StandardScaler()

    def build_model(self, X, y):
        with pm.Model() as self.model:
            X_obs = pm.Data("X_obs", X)
            y_obs = pm.Data("y_obs", y)

            # Non-centered parameterization
            X_centered = np.array(X) - np.array(X).mean(axis=0)
            Kxx = (X_centered.T @ X_centered) / X_centered.shape[0]
            L = pt.linalg.cholesky(pm.gp.util.stabilize(pt.as_tensor_variable(Kxx)))

            # Define priors
            w_raw = pm.Normal("w_raw", mu=0.0, sigma=1.0, shape=X.shape[1])
            w = pm.Deterministic("w", pt.dot(L, w_raw) / np.sqrt(X.shape[1]))
            b = pm.Normal("b", mu=self.model_config["b_mu"], sigma=self.model_config["b_sigma"])

            # Define likelihood
            logit = pm.math.dot(X_obs, w) + b
            pm.Bernoulli(self.output_var, logit_p=logit, observed=y_obs)

    def fit(self, X, y):
        self.network.fit(X, y)
        X_features = self._extract_features(X)
        X_features_scaled = self.scaler.fit_transform(X_features)
        return super().fit(X_features_scaled, y)

    def _extract_features(self, X):
        module = self.network.load_backbone()
        backbone = torch.nn.Sequential(*list(module.children())[:-1])
        backbone.eval()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            return backbone(X_tensor).flatten(start_dim=1).numpy().copy()

    def predict_proba(self, X):
        X_features = self._extract_features(X)
        X_features_scaled = self.scaler.transform(X_features)
        return super().predict_proba(X_features_scaled)

    @staticmethod
    def get_default_model_config():
        return {
            "b_mu": 0,
            "b_sigma": 1.0,
        }
