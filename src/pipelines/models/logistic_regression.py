"""
Build Bayesian logistic regression model.

References
----------
.. [1] https://doi.org/10.1007/978-0-387-84858-7_4
.. [2] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [3] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import pandas as pd
import numpy as np
import pymc as pm
from os import getenv
from dotenv import load_dotenv
from pymc_extras.model_builder import ModelBuilder
from sklearn.base import BaseEstimator, ClassifierMixin


class BayesianLogisticRegression(ModelBuilder, ClassifierMixin, BaseEstimator):
    _model_type = "BayesianLogisticRegression"
    version = "0.1"

    def __init__(self, model_config=None, sampler_config=None):
        super().__init__(model_config, sampler_config)

        # Load environment variables
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))

    def build_model(self, X, y):
        self._generate_and_preprocess_model_data(X, y)
        n_features = X.shape[1]

        with pm.Model() as self.model:
            x_data = pm.Data("x_data", X)
            y_data = pm.Data("y_data", y)

            # Define priors
            b = pm.Normal(
                "b",
                mu=self.model_config["b_mu_prior"],
                sigma=self.model_config["b_sigma_prior"],
            )
            w = pm.Normal(
                "w",
                mu=self.model_config["w_mu_prior"],
                sigma=self.model_config["w_sigma_prior"],
                shape=n_features,
            )

            # Define likelihood
            logit = pm.math.dot(x_data, w) + b
            pm.Bernoulli("y", logit_p=logit, observed=y_data)

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name=self.output_var)
        self.classes_ = np.unique(y)
        super().fit(X, y=y, progressbar=False, random_seed=self.random_state)

    def predict_proba(self, X):
        posterior_samples = super().predict_proba(X)
        proba = posterior_samples.mean(dim=["chain", "draw"]).values
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @staticmethod
    def get_default_model_config():
        return {
            "b_mu_prior": 0,
            "b_sigma_prior": 10.0,
            "w_mu_prior": 0,
            "w_sigma_prior": 10.0,
        }

    @staticmethod
    def get_default_sampler_config():
        return {
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
        }

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self):
        return self.model_config

    def _data_setter(self, X, y=None):
        with self.model:
            pm.set_data({"x_data": X})
            if y is not None:
                pm.set_data({"y_data": y})
            else:
                pm.set_data({"y_data": np.zeros(X.shape[0], dtype=np.int32)})

    def _generate_and_preprocess_model_data(self, X, y):
        self.X = X
        self.y = y
