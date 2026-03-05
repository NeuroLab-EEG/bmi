import numpy as np
from os import path
from arviz import InferenceData
from .subprocessor_base import SubprocessorBase


class PyMCSubprocessor(SubprocessorBase):
    def save_fitted_state(self):
        np.save(path.join(self.save_dir, "X.npy"), self.estimator.X)
        np.save(path.join(self.save_dir, "y.npy"), self.estimator.y)
        self.estimator.idata.to_netcdf(path.join(self.save_dir, "idata.nc"))

    def load_fitted_state(self):
        X = np.load(path.join(self.save_dir, "X.npy"))
        y = np.load(path.join(self.save_dir, "y.npy"))
        self.estimator.build_model(X, y)
        self.estimator.idata = InferenceData.from_netcdf(path.join(self.save_dir, "idata.nc"))
        self.estimator.classes_ = self.classes_
