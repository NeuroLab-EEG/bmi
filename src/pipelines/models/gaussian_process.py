"""
Make Gaussian process model.

References
----------
.. [1] https://doi.org/10.7551/mitpress/3206.001.0001
.. [2] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html#example-2-classification
.. [3] https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html
.. [4] https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.SVGD.html
"""

import pandas as pd
import numpy as np
import pymc as pm
from os import getenv
from dotenv import load_dotenv
from pymc_extras.model_builder import ModelBuilder
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianProcess(ModelBuilder, ClassifierMixin, BaseEstimator):
    _model_type = "GaussianProcess"
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

            # Define covariance priors
            ell = pm.InverseGamma("ell", mu=self.model_config["ell_mu"], sigma=self.model_config["ell_sigma"], shape=n_features)
            eta = pm.HalfNormal("eta", sigma=self.model_config["eta_sigma"])
            cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=n_features, ls=ell)

            # Define latent function priors
            gp = pm.gp.Latent(cov_func=cov)
            f = gp.prior("f", X=x_data)

            # Define likelihood
            p = pm.Deterministic("p", pm.math.invlogit(f))
            pm.Bernoulli("y", p=p, observed=y_data)

    def fit(self, X, y):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name=self.output_var)
        self.classes_ = np.unique(y)
        super().fit(X, y=y, random_seed=self.random_state)

    def sample_model(self, **kwargs):
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}

            approx = pm.SVGD(
                n_particles=sampler_args["n_particles"],
                random_seed=sampler_args["random_seed"],
            )
            approx.fit(
                n=sampler_args["n"],
                progressbar=sampler_args["progressbar"],
            )

            idata = approx.sample(draws=sampler_args["draws"])
            idata.extend(pm.sample_prior_predictive(), join="right")
            idata.extend(pm.sample_posterior_predictive(idata, var_names=[self.output_var]), join="right")

        idata = self.set_idata_attrs(idata)
        return idata

    def predict_proba(self, X):
        posterior_samples = super().predict_proba(
            X,
            extend_idata=False,
            combined=True,
            var_names=[self.output_var],
        )
        proba = posterior_samples.mean(dim="sample").values
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    @staticmethod
    def get_default_model_config():
        return {
            "ell_mu": 1.0,
            "ell_sigma": 0.5,
            "eta_sigma": 1.0,
        }

    @staticmethod
    def get_default_sampler_config():
        return {
            "n_particles": 300,
            "n": 50000,
            "progressbar": False,
            "draws": 2000,
        }
    
    @property
    def output_var(self):
        return "p"

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
