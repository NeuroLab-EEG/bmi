"""
Make pipeline for CSP + Bayesian LDA.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
.. [2] https://www.pymc.io/projects/examples/en/latest/howto/LKJ.html
.. [3] https://python.arviz.org/en/stable/getting_started/XarrayforArviZ.html#xarray-for-arviz
.. [4] https://doi.org/10.1007/978-0-387-84858-7_4
.. [5] https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html
"""

import numpy as np
import pymc as pm
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import multivariate_normal as mvn
from src.pipelines.pipeline import Pipeline


class BayesianLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, draws=2000, tune=1000, chains=4, random_state=None):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.random_state = random_state

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        y_binary = (y == self.classes_[1]).astype(int)

        with pm.Model() as _:
            # Model class prior and likelihood
            pi = pm.Beta("pi", alpha=1.0, beta=1.0)
            pm.Bernoulli("y", p=pi, observed=y_binary)

            # Define mean priors
            mu_0 = pm.Normal("mu_0", mu=0, sigma=10.0, shape=self.n_features_)
            mu_1 = pm.Normal("mu_1", mu=0, sigma=10.0, shape=self.n_features_)

            # Define covariance prior
            chol, _, _ = pm.LKJCholeskyCov(
                "chol",
                n=self.n_features_,
                eta=2.0,
                sd_dist=pm.Exponential.dist(1.0),
                compute_corr=True,
            )
            pm.Deterministic("cov", chol.dot(chol.T))

            # Define multivariate normal likelihood
            mu = pm.math.switch(y_binary[:, np.newaxis], mu_1, mu_0)
            pm.MvNormal("X", mu=mu, chol=chol, observed=X, shape=X.shape)

            # Sample posterior using MCMC
            self.idata_ = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_state,
                quiet=True,
            )

        return self

    def predict_proba(self, X):
        X = self._validate_data(X, reset=False)

        # Extract posterior parameters
        posterior = self.idata_.posterior
        pi = posterior["pi"].values.flatten()
        mu_0 = posterior["mu_0"].values.reshape(-1, self.n_features_)
        mu_1 = posterior["mu_1"].values.reshape(-1, self.n_features_)
        cov = posterior["cov"].values.reshape(-1, self.n_features_, self.n_features_)

        # Compute posterior predictive distribution
        p_1_given_x = np.array(
            [self._compute_class_logproba(X, pi[i], mu_0[i], mu_1[i], cov[i]) for i in range(len(pi))]
        )
        proba_class_1 = p_1_given_x.mean(axis=0)
        return np.column_stack([1 - proba_class_1, proba_class_1])

    def _compute_class_logproba(self, X, pi, mu_0, mu_1, cov):
        """
        Evaluate generative model posterior.
        """
        # Evaluate priors
        log_p_0 = np.log(1 - pi)
        log_p_1 = np.log(pi)

        # Evaluate likelihoods
        log_p_x_given_0 = mvn.logpdf(X, mu_0, cov)
        log_p_x_given_1 = mvn.logpdf(X, mu_1, cov)

        # Evaluate posteriors
        log_p_0_given_x = log_p_x_given_0 + log_p_0
        log_p_1_given_x = log_p_x_given_1 + log_p_1

        # Subtract evidence
        log_sum = np.logaddexp(log_p_0_given_x, log_p_1_given_x)
        return np.exp(log_p_1_given_x - log_sum)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class CSPBLDA(Pipeline):
    def pipeline(self):
        return {
            "CSPBLDA": make_pipeline(
                Covariances(estimator="oas"),
                CSP(nfilter=6),
                BayesianLDA(),
            )
        }

    def params(self):
        return {}
