from os import path

import numpy as np

from .pymc_subprocessor import PyMCSubprocessor


class GPPyMCSubprocessor(PyMCSubprocessor):
    def __init__(self, estimator, root_dir, save_weights=False):
        super().__init__(estimator, root_dir, save_weights)

    def _build_model(self, X, y):
        Xu = np.load(path.join(self.save_dir, "Xu.npy"))
        self.estimator.build_model(X, y, Xu)

    def save_fitted_state(self):
        np.save(path.join(self.save_dir, "Xu.npy"), self.estimator.Xu)
        super().save_fitted_state()
