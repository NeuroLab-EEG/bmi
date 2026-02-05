"""
Make Gaussian process model.

References
----------
.. [1] https://doi.org/10.7551/mitpress/3206.001.0001
.. [2] https://doi.org/10.48550/arXiv.1312.0906
.. [3] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html
.. [4] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html#sparse-heteroskedastic-gp
.. [5] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [6] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from os import getenv
from dotenv import load_dotenv
from pymc_extras.model_builder import ModelBuilder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans


class SparseLatent:
    def __init__(self, cov_func):
        self.cov = cov_func

    def prior(self, name, X, Xu):
        Kuu = self.cov(Xu)
        L = pt.linalg.cholesky(pm.gp.util.stabilize(Kuu))

        v = pm.Normal(f"v_{name}", mu=0.0, sigma=1.0, shape=Xu.shape[0])
        u = pt.dot(L, v)

        Kfu = self.cov(X, Xu)
        L_inv_u = pt.linalg.solve_triangular(L, u, lower=True)
        Kuiu = pt.linalg.solve_triangular(L.T, L_inv_u, lower=False)

        return pt.dot(Kfu, Kuiu)


class GaussianProcess(ModelBuilder, ClassifierMixin, BaseEstimator):
    _model_type = "GaussianProcess"
    version = "0.1"

    def __init__(self, model_config=None, sampler_config=None):
        super().__init__(model_config, sampler_config)

        # Load environment variables
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))

    def build_model(self, X, y):
        n_features = X.shape[1]
        n_inducing = self.model_config.get("n_inducing")

        # Get inducing points
        kmeans = KMeans(n_clusters=min(n_inducing, X.shape[0]))
        Xu = kmeans.fit(X).cluster_centers_

        with pm.Model() as self.model:
            x_data = pm.Data("x_data", X)
            y_data = pm.Data("y_data", y)
            xu_data = pm.Data("xu_data", Xu)

            # Define covariance priors
            ell = pm.InverseGamma(
                "ell", mu=self.model_config["ell_mu"], sigma=self.model_config["ell_sigma"], shape=n_features
            )
            eta = pm.HalfNormal("eta", sigma=self.model_config["eta_sigma"])
            cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=n_features, ls=ell)

            # Define latent function priors
            gp = SparseLatent(cov_func=cov)
            f = gp.prior("f", X=x_data, Xu=xu_data)

            # Define likelihood
            p = pm.math.invlogit(f)
            pm.Bernoulli("y", p=p, observed=y_data, shape=x_data.shape[0])

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
            "ell_mu": 1.0,
            "ell_sigma": 0.5,
            "eta_sigma": 1.0,
            "n_inducing": 50,
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
