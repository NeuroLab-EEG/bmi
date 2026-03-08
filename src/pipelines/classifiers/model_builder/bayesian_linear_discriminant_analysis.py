"""
Build Bayesian linear discriminant analysis classifier.

References
----------
.. [1] https://doi.org/10.1007/978-0-387-84858-7_4
.. [2] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [3] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import pymc as pm
from .model_builder_base import ModelBuilderBase


class BayesianLinearDiscriminantAnalysis(ModelBuilderBase):
    def build_model(self, X, y):
        with pm.Model() as self.model:
            X_obs = pm.Data("X_obs", X)
            y_obs = pm.Data("y_obs", y)

            # Define class prior
            pi = pm.Beta("pi", alpha=self.model_config["pi_alpha"], beta=self.model_config["pi_beta"])

            # Define location priors
            mu_0 = pm.Normal(
                "mu_0",
                mu=self.model_config["mu_0_mu"],
                sigma=self.model_config["mu_0_sigma"],
                shape=X.shape[1],
            )
            mu_1 = pm.Normal(
                "mu_1",
                mu=self.model_config["mu_1_mu"],
                sigma=self.model_config["mu_1_sigma"],
                shape=X.shape[1],
            )

            # Define shared scale prior
            sigma = pm.HalfNormal("sigma", sigma=self.model_config["sigma_sigma"], shape=X.shape[1])

            # Define likelihood of observations
            pm.Mixture(
                "X",
                w=pm.math.stack([1 - pi, pi]),
                comp_dists=[
                    pm.Normal.dist(mu=mu_0, sigma=sigma),
                    pm.Normal.dist(mu=mu_1, sigma=sigma),
                ],
                observed=X_obs,
            )

            # Define likelihood of labels
            logp_0 = pm.logp(pm.Normal.dist(mu=mu_0, sigma=sigma), X_obs).sum(axis=1)
            logp_1 = pm.logp(pm.Normal.dist(mu=mu_1, sigma=sigma), X_obs).sum(axis=1)
            logit = logp_1 + pm.math.log(pi) - logp_0 - pm.math.log(1 - pi)
            pm.Bernoulli(self.output_var, logit_p=logit, observed=y_obs)

    @staticmethod
    def get_default_model_config():
        return {
            "pi_alpha": 2.0,
            "pi_beta": 2.0,
            "mu_0_mu": 0,
            "mu_0_sigma": 1.0,
            "mu_1_mu": 0,
            "mu_1_sigma": 1.0,
            "sigma_sigma": 1.0,
        }
