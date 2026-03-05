from os import path
from .subprocessor_base import SubprocessorBase


class PyTorchSubprocessor(SubprocessorBase):
    def save_fitted_state(self):
        self.estimator.model_.save_params(
            f_params=path.join(self.save_dir, "params.pt"),
            f_optimizer=path.join(self.save_dir, "optimizer.pt"),
            f_history=path.join(self.save_dir, "history.json"),
        )

    def load_fitted_state(self):
        self.estimator.model_ = self.estimator.build_classifier()
        self.estimator.model_.initialize()
        self.estimator.model_.load_params(
            f_params=path.join(self.save_dir, "params.pt"),
            f_optimizer=path.join(self.save_dir, "optimizer.pt"),
            f_history=path.join(self.save_dir, "history.json"),
        )
        self.estimator.classes_ = self.classes_
