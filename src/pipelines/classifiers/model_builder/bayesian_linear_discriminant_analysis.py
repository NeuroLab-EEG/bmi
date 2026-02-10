"""
Build Bayesian linear discriminant analysis classifier.

References
----------
.. [1] https://doi.org/10.1007/978-0-387-84858-7_4
.. [2] https://www.pymc.io/projects/examples/en/latest/howto/LKJ.html
.. [3] https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html
"""

import pymc as pm
from src.pipelines.classifiers.model_builder import ModelBuilderBase


class BayesianLinearDiscriminantAnalysis(ModelBuilderBase):
    def build_model(self, X, y):
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
                shape=X.shape[1],
            )
            mu_1 = pm.Normal(
                "mu_1",
                mu=self.model_config["mu_1_mu"],
                sigma=self.model_config["mu_1_sigma"],
                shape=X.shape[1],
            )

            # Define covariance prior
            chol, _, _ = pm.LKJCholeskyCov(
                "chol",
                n=X.shape[1],
                eta=self.model_config["chol_eta"],
                sd_dist=pm.Exponential.dist(self.model_config["chol_sd_dist_lambda"]),
                compute_corr=True,
            )

            # Define feature likelihood
            mu = pm.math.switch(y_data[:, None], mu_1, mu_0)
            pm.MvNormal("X", mu=mu, chol=chol, observed=x_data, shape=x_data.shape)

            # Define class label likelihood
            pm.Bernoulli(self.output_var, p=pi, observed=y_data)

    @staticmethod
    def get_default_model_config():
        return {
            "pi_alpha": 1.0,
            "pi_beta": 1.0,
            "mu_0_mu": 0,
            "mu_0_sigma": 1.0,
            "mu_1_mu": 0,
            "mu_1_sigma": 1.0,
            "chol_eta": 2.0,
            "chol_sd_dist_lambda": 1.0,
        }
