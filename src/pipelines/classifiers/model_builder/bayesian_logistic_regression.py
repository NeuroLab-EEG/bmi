"""
Build Bayesian logistic regression classifier.

References
----------
.. [1] https://doi.org/10.1007/978-0-387-84858-7_4
.. [2] https://bayesiancomputationbook.com/markdown/chp_04.html#posterior-geometry-matters
.. [3] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [4] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.decomposition import TruncatedSVD
from .model_builder_base import ModelBuilderBase


class LowRankPrior:
    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random_state = random_state

    def prior(self, name, X):
        X_centered = X - X.mean(axis=0)
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        X_reduced = svd.fit_transform(X_centered)
        Vt = svd.components_

        Kxx_reduced = (X_reduced.T @ X_reduced) / X_centered.shape[0]
        L = pt.linalg.cholesky(pm.gp.util.stabilize(pt.as_tensor_variable(Kxx_reduced)))

        w_raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=1.0, shape=self.n_components)
        w_reduced = pt.dot(L, w_raw)
        return pm.Deterministic(
            name, pt.dot(pt.as_tensor_variable(Vt.T), w_reduced) / np.sqrt(self.n_components)
        )


class BayesianLogisticRegression(ModelBuilderBase):
    def build_model(self, X, y):
        with pm.Model() as self.model:
            X_obs = pm.Data("X_obs", X)
            y_obs = pm.Data("y_obs", y)

            # Define priors
            prior = LowRankPrior(
                n_components=self.model_config["n_components"], random_state=self.random_state
            )
            w = prior.prior("w", X=np.array(X))
            b = pm.Normal("b", mu=self.model_config["b_mu"], sigma=self.model_config["b_sigma"])

            # Define likelihood
            logit = pm.math.dot(X_obs, w) + b
            pm.Bernoulli(self.output_var, logit_p=logit, observed=y_obs)

    @staticmethod
    def get_default_model_config():
        return {
            "b_mu": 0,
            "b_sigma": 1.0,
            "n_components": 100,
        }
