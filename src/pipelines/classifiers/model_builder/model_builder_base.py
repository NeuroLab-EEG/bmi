"""
Make Bayesian classifier.

References
----------
.. [1] https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
.. [2] https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.model_builder.ModelBuilder.html
"""

import numpy as np
import pandas as pd
import pymc as pm
from pymc_extras.model_builder import ModelBuilder
from sklearn.base import BaseEstimator, ClassifierMixin


class ModelBuilderBase(ModelBuilder, ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model_config=None,
        sampler_config=None,
        progressbar=False,
        random_state=None,
    ):
        super().__init__(model_config, sampler_config)
        self.progressbar = progressbar
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "model_config": self.model_config,
            "sampler_config": self.sampler_config,
            "progressbar": self.progressbar,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        self.model_config = params.get("model_config", self.model_config)
        self.sampler_config = params.get("sampler_config", self.sampler_config)
        self.progressbar = params.get("progressbar", self.progressbar)
        self.random_state = params.get("random_state", self.random_state)
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y, name=self.output_var)
        return super().fit(
            X_df,
            y=y_series,
            progressbar=self.progressbar,
            random_seed=self.random_state,
        )

    def predict_proba(self, X):
        posterior_samples = super().predict_proba(X, var_names=[self.output_var])
        proba = posterior_samples.mean(dim=["chain", "draw"]).values
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def graph_model(self):
        if self.model is None:
            # Hardcode (n_trials, n_channels) from the gold standard BNCI2014_001
            n_trials, n_channels = (576, 22)
            X_dummy = np.random.randn(n_trials, n_channels)
            y_dummy = np.random.randint(0, 2, size=n_trials)
            self.build_model(X_dummy, y_dummy)

        with self.model:
            graph = pm.model_to_graphviz(self.model)
            graph.render(self.__class__.__name__, format="png", cleanup=True)

    @staticmethod
    def get_default_sampler_config():
        return {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
            "random_seed": None,
            "progressbar": None,
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
            pm.set_data({"X_obs": X})
            if y is not None:
                pm.set_data({"y_obs": y})
            else:
                pm.set_data({"y_obs": np.zeros(X.shape[0], dtype=np.int32)})

    def _generate_and_preprocess_model_data(self, X, y):
        self.X = X
        self.y = y
