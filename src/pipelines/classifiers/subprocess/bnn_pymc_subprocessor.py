from os import path

import joblib
import numpy as np

from .pymc_subprocessor import PyMCSubprocessor


class BNNPyMCSubprocessor(PyMCSubprocessor):
    def __init__(self, estimator, root_dir, save_weights=False):
        super().__init__(estimator, root_dir, save_weights)

    def save_fitted_state(self):
        joblib.dump(self.estimator.scaler, path.join(self.save_dir, "scaler.pkl"))
        np.save(path.join(self.save_dir, "network_save_dir.npy"), np.array([self.estimator.network.save_dir]))
        super().save_fitted_state()

    def load_fitted_state(self):
        self.estimator.scaler = joblib.load(path.join(self.save_dir, "scaler.pkl"))
        self.estimator.network.save_dir = np.load(path.join(self.save_dir, "network_save_dir.npy"))[0]
        super().load_fitted_state()
