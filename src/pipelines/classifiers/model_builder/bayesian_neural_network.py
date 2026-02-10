"""
Build Bayesian neural network classifier.
"""

import torch
import pandas as pd
import numpy as np
import pymc as pm
from src.pipelines.classifiers.model_builder import ModelBuilderBase


class BayesianNeuralNetwork(ModelBuilderBase):
    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def build_model(self, X, y):
        with pm.Model() as self.model:
            x_data = pm.Data("x_data", X)
            y_data = pm.Data("y_data", y)

            # Define priors
            w = pm.Normal(
                "w",
                mu=self.weight.flatten(),
                sigma=self.model_config["w_sigma"],
                shape=X.shape[1],
            )
            b = pm.Normal(
                "b",
                mu=self.bias.item(),
                sigma=self.model_config["b_sigma"],
            )

            # Define likelihood
            logit = pm.math.dot(x_data, w) + b
            pm.Bernoulli(self.output_var, logit_p=logit, observed=y_data)

    def fit(self, X, y):
        # Train neural network
        self.network.fit(X, y)

        # Extract features from backbone
        self.backbone = self.network.module_[:-1]
        self.backbone.eval()
        X_features = self._extract_features(X)

        # Get parameters from linear classification
        self.weight = self.network.module_[-1].conv_classifier.weight.detach().cpu().numpy()
        self.bias = self.network.module_[-1].conv_classifier.bias.detach().cpu().numpy()

        # Sample posterior of classification parameters
        self.classes_ = np.unique(y)
        X_features_df = pd.DataFrame(X_features, columns=[f"x{i}" for i in range(X_features.shape[1])])
        y_series = pd.Series(y, name=self.output_var)
        return super().fit(
            X_features_df, y=y_series, progressbar=self.progressbar, random_seed=self.random_state
        )

    def _extract_features(self, X):
        X_tensor = torch.from_numpy(X).float().cuda()
        with torch.no_grad():
            features = self.backbone(X_tensor).flatten(start_dim=1).cpu().numpy()
        return features

    def predict_proba(self, X):
        X_features = self._extract_features(X)
        return super().predict_proba(X_features)

    @staticmethod
    def get_default_model_config():
        return {
            "w_sigma": 0.5,
            "b_sigma": 0.5,
        }
