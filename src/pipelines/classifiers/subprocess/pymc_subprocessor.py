from os import path, remove

import numpy as np
from arviz import InferenceData

from .subprocessor_base import SubprocessorBase


class PyMCSubprocessor(SubprocessorBase):
    def __init__(self, estimator, root_dir, save_weights=False):
        super().__init__(estimator, root_dir, save_weights)

    def __del__(self):
        super().__del__()
        self._cleanup_save_dir()

    def _cleanup_save_dir(self):
        """Remove temporary data files within current fold."""
        if self.save_dir is None:
            return
        for filename in ["X.npy", "y.npy"]:
            filepath = path.join(self.save_dir, filename)
            if path.exists(filepath):
                remove(filepath)

    def _build_model(self, X, y):
        self.estimator.build_model(X, y)

    def save_fitted_state(self):
        np.save(path.join(self.save_dir, "X.npy"), self.estimator.X)
        np.save(path.join(self.save_dir, "y.npy"), self.estimator.y)
        self.estimator.idata.to_netcdf(path.join(self.save_dir, "idata.nc"))

    def load_fitted_state(self):
        X = np.load(path.join(self.save_dir, "X.npy"))
        y = np.load(path.join(self.save_dir, "y.npy"))
        self._build_model(X, y)
        self.estimator.idata = InferenceData.from_netcdf(path.join(self.save_dir, "idata.nc"))
        self.estimator.classes_ = self.classes_
