import pickle
from os import path
from .subprocessor_base import SubprocessorBase


class CuMLSubprocessor(SubprocessorBase):
    def save_fitted_state(self):
        with open(path.join(self.save_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.estimator.model_, f)

    def load_fitted_state(self):
        with open(path.join(self.save_dir, "model.pkl"), "rb") as f:
            self.estimator.model_ = pickle.load(f)
