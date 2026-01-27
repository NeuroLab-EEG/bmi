"""
Make pipeline for TS + Bayesian LR.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
.. [2] https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM_linear.html#glm-linear
.. [3] https://python.arviz.org/en/stable/getting_started/XarrayforArviZ.html#xarray-for-arviz
.. [4] https://doi.org/10.1007/978-0-387-84858-7_4
"""

import numpy as np
import pymc as pm
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import expit
from src.pipelines.pipeline import Pipeline


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_samples=2000, tune=1000, chains=4, prior_sigma=5.0, random_state=None):
        self.n_samples = n_samples
        self.tune = tune
        self.chains = chains
        self.prior_sigma = prior_sigma
        self.random_state = random_state

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.classes_ = np.unique(y)
        y_binary = (y == self.classes_[1]).astype(int)

        with pm.Model() as _:
            # Define prior parameters
            b = pm.Normal("b", mu=0, sigma=self.prior_sigma)
            w = pm.Normal("w", mu=0, sigma=self.prior_sigma, shape=X.shape[1])

            # Define likelihood
            logit = pm.math.dot(X, w) + b
            _ = pm.Bernoulli("y", logit_p=logit, observed=y_binary)

            # Sample posterior using HMC
            self.idata_ = pm.sample(
                draws=self.n_samples,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_state,
                return_inferencedata=True,
                progressbar=False,
            )

        return self

    def predict_proba(self, X):
        X = self._validate_data(X, reset=False)

        # Extract posterior parameters
        posterior = self.idata_.posterior
        b = posterior["b"].values.flatten()
        w = posterior["w"].values.reshape(-1, X.shape[1])

        # Compute posterior predictive distribution
        logit = b[:, np.newaxis] + w @ X.T
        proba = expit(logit).mean(axis=0)
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class TSBLR(Pipeline):
    def pipeline(self):
        return {
            "TSBLR": make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                BayesianLogisticRegression(),
            )
        }

    def params(self):
        return {}
