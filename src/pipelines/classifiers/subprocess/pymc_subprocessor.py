import numpy as np
from os import path, remove
from arviz import InferenceData
from .subprocessor_base import SubprocessorBase


class PyMCSubprocessor(SubprocessorBase):
    def __del__(self):
        self._cleanup_disk()

    def _cleanup_disk(self):
        """Remove data files accessed across subprocesses."""
        remove(path.join(self.save_dir, "X.npy"))
        remove(path.join(self.save_dir, "y.npy"))

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
