"""
Make Gaussian process classifier.

References
----------
.. [1] https://doi.org/10.7551/mitpress/3206.001.0001
.. [2] https://bayesiancomputationbook.com/markdown/chp_04.html#posterior-geometry-matters
.. [3] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html#example-2-classification
.. [4] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html#sparse-heteroskedastic-gp
.. [5] https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-SparseApprox.html#initializing-the-inducing-points-with-k-means
.. [6] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [7] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from abc import abstractmethod
from .model_builder_base import ModelBuilderBase


class SparseLatent:
    def __init__(self, cov_func):
        self.cov = cov_func

    def prior(self, name, X, Xu):
        Kuu = self.cov(Xu)
        L = pt.linalg.cholesky(pm.gp.util.stabilize(Kuu))

        v = pm.Normal(name, mu=0.0, sigma=1.0, shape=Xu.shape[0])
        u = pt.dot(L, v)

        Kfu = self.cov(X, Xu)
        L_inv_u = pt.linalg.solve_triangular(L, u, lower=True)
        Kuiu = pt.linalg.solve_triangular(L.T, L_inv_u, lower=False)

        return pt.dot(Kfu, Kuiu)


class GaussianProcess(ModelBuilderBase):
    KERNELS = {}

    def __init_subclass__(cls, kernel, **kwargs):
        super().__init_subclass__(**kwargs)
        GaussianProcess.KERNELS[kernel] = cls

    @classmethod
    def from_kernel(cls, kernel, **kwargs):
        return cls.KERNELS[kernel](**kwargs)

    def build_model(self, X, y):
        with pm.Model() as self.model:
            X_obs = pm.Data("X_obs", X)
            y_obs = pm.Data("y_obs", y)

            # Get inducing points
            n_inducing = self.model_config["n_inducing"]
            Xu = pm.gp.util.kmeans_inducing_points(n_inducing, np.array(X))

            # Define covariance priors
            cov = self._covariance(X.shape[1])

            # Define latent function priors
            gp = SparseLatent(cov_func=cov)
            f = gp.prior("f", X=X_obs, Xu=Xu)

            # Define likelihood
            pm.Bernoulli(self.output_var, logit_p=f, observed=y_obs)

    @abstractmethod
    def _covariance(self, n_features):
        pass


class LinearGP(GaussianProcess, kernel="linear"):
    def _covariance(self, n_features):
        eta = pm.HalfNormal("eta", sigma=self.model_config["eta_sigma"])
        return eta**2 * pm.gp.cov.Linear(input_dim=n_features, c=0)

    @staticmethod
    def get_default_model_config():
        return {
            "eta_sigma": 1.0,
            "n_inducing": 100,
        }


class RBFGP(GaussianProcess, kernel="rbf"):
    def _covariance(self, n_features):
        ell = pm.LogNormal("ell", mu=self.model_config["ell_mu"], sigma=self.model_config["ell_sigma"])
        eta = pm.HalfNormal("eta", sigma=self.model_config["eta_sigma"])
        return eta**2 * pm.gp.cov.ExpQuad(input_dim=n_features, ls=ell)

    @staticmethod
    def get_default_model_config():
        return {
            "ell_mu": 0,
            "ell_sigma": 0.5,
            "eta_sigma": 1.0,
            "n_inducing": 100,
        }
