import numpy as np
from os import path
from arviz import InferenceData
from .subprocess_estimator import SubprocessEstimator


class PyMCSubprocessEstimator(SubprocessEstimator):
    def save_fitted_state(self):
        self.estimator.idata.to_netcdf(path.join(self._save_path, "idata.nc"))
        np.save(path.join(self._save_path, "X.npy"), self.estimator.X)
        np.save(path.join(self._save_path, "y.npy"), self.estimator.y)

    def load_fitted_state(self):
        X = np.load(path.join(self._save_path, "X.npy"))
        y = np.load(path.join(self._save_path, "y.npy"))
        self.estimator.build_model(X, y)
        self.estimator.idata = InferenceData.from_netcdf(path.join(self._save_path, "idata.nc"))
        self.estimator.classes_ = self.classes_
