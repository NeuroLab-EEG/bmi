"""
Build Bayesian logistic regression classifier.

References
----------
.. [1] https://doi.org/10.1007/978-0-387-84858-7_4
.. [2] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [3] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import numpy as np
import pymc as pm
from .model_builder_base import ModelBuilderBase


class BayesianLogisticRegression(ModelBuilderBase):
    def build_model(self, X, y):
        with pm.Model() as self.model:
            X_obs = pm.Data("X_obs", X)
            y_obs = pm.Data("y_obs", y)

            # Define priors
            w = pm.Normal(
                "w",
                mu=self.model_config["w_mu"],
                sigma=1 / np.sqrt(X.shape[1]),
                shape=X.shape[1],
            )
            b = pm.Normal(
                "b",
                mu=self.model_config["b_mu"],
                sigma=self.model_config["b_sigma"],
            )

            # Define likelihood
            logit = pm.math.dot(X_obs, w) + b
            pm.Bernoulli(self.output_var, logit_p=logit, observed=y_obs)

    @staticmethod
    def get_default_model_config():
        return {
            "w_mu": 0,
            "b_mu": 0,
            "b_sigma": 1.0,
        }
