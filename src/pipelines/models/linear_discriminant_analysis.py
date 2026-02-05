"""
Build Bayesian linear discriminant analysis model.

References
----------
.. [1] https://doi.org/10.1007/978-0-387-84858-7_4
.. [2] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [3] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
.. [4] https://www.pymc.io/projects/examples/en/latest/howto/LKJ.html
.. [5] https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html
"""

import pandas as pd
import numpy as np
import pymc as pm
from os import getenv
from dotenv import load_dotenv
from pymc_extras.model_builder import ModelBuilder
from sklearn.base import BaseEstimator, ClassifierMixin


class BayesianLinearDiscriminantAnalasis(ModelBuilder, ClassifierMixin, BaseEstimator):
    _model_type = "BayesianLinearDiscriminantAnalysis"
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

            # Define class prior
            pi = pm.Beta("pi", alpha=self.model_config["pi_alpha"], beta=self.model_config["pi_beta"])

            # Define mean priors
            mu_0 = pm.Normal(
                "mu_0",
                mu=self.model_config["mu_0_mu"],
                sigma=self.model_config["mu_0_sigma"],
                shape=n_features,
            )
            mu_1 = pm.Normal(
                "mu_1",
                mu=self.model_config["mu_1_mu"],
                sigma=self.model_config["mu_1_sigma"],
                shape=n_features,
            )

            # Define covariance prior
            chol, _, _ = pm.LKJCholeskyCov(
                "chol",
                n=n_features,
                eta=self.model_config["chol_eta"],
                sd_dist=pm.Exponential.dist(self.model_config["chol_sd_dist_lambda"]),
                compute_corr=True,
            )

            # Define feature likelihood
            mu = pm.math.switch(y_data[:, np.newaxis], mu_1, mu_0)
            pm.MvNormal("X", mu=mu, chol=chol, observed=x_data, shape=x_data.shape)

            # Define class label likelihood
            pm.Bernoulli("y", p=pi, observed=y_data)

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name=self.output_var)
        self.classes_ = np.unique(y)
        super().fit(X, y=y, random_seed=self.random_state)

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
            "pi_alpha": 1.0,
            "pi_beta": 1.0,
            "mu_0_mu": 0,
            "mu_0_sigma": 10.0,
            "mu_1_mu": 0,
            "mu_1_sigma": 10.0,
            "chol_eta": 2.0,
            "chol_sd_dist_lambda": 1.0,
        }

    @staticmethod
    def get_default_sampler_config():
        return {
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
            "random_seed": None,
            "progressbar": False,
            "nuts_sampler": "numpyro",
            "nuts_sampler_kwargs": {"chain_method": "parallel"},
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
