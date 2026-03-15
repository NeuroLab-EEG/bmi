import numpy as np
from os import path
from .pymc_subprocessor import PyMCSubprocessor


class GPPyMCSubprocessor(PyMCSubprocessor):
    def _build_model(self, X, y):
        Xu = np.load(path.join(self.save_dir, "Xu.npy"))
        self.estimator.build_model(X, y, Xu)

    def save_fitted_state(self):
        np.save(path.join(self.save_dir, "Xu.npy"), self.estimator.Xu)
        super().save_fitted_state()
