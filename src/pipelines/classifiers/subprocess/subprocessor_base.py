"""
Spawn subprocess for estimator fitting with OS reclaiming all memory on finish.

References
----------
.. [1] https://docs.python.org/3/library/multiprocessing.html
"""

import multiprocessing as mp
import shutil
from abc import abstractmethod
from datetime import datetime, timezone
from os import makedirs, path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _fit_worker(subprocessor, X, y, queue):
    try:
        subprocessor.estimator.fit(X, y)
        subprocessor.save_fitted_state()
        queue.put(("ok", None))
    except Exception as e:  # noqa: BLE001
        queue.put(("err", e))


def _predict_worker(subprocessor, X, queue):
    try:
        subprocessor.load_fitted_state()
        result = subprocessor.estimator.predict(X)
        queue.put(("ok", result))
    except Exception as e:  # noqa: BLE001
        queue.put(("err", e))


def _predict_proba_worker(subprocessor, X, queue):
    try:
        subprocessor.load_fitted_state()
        result = subprocessor.estimator.predict_proba(X)
        queue.put(("ok", result))
    except Exception as e:  # noqa: BLE001
        queue.put(("err", e))


def _spawn_worker(target, args):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    proc.join()
    status, payload = queue.get()
    if status == "err":
        raise payload
    return payload


class SubprocessorBase(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, root_dir, save_weights=False):
        self.estimator = estimator
        self.root_dir = root_dir
        self.save_dir = None
        self.save_weights = save_weights

    def __del__(self):
        self._cleanup_root_dir()

    def __sklearn_is_fitted__(self):
        return True

    def _cleanup_root_dir(self):
        """Remove fitted model weights across all folds."""
        if self.save_weights:
            return
        filepath = path.join(self.root_dir, self.__class__.__name__)
        if path.exists(filepath):
            shutil.rmtree(filepath)

    def _make_save_dir(self):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        save_dir = path.join(self.root_dir, self.__class__.__name__, timestamp)
        makedirs(save_dir, exist_ok=True)
        return save_dir

    @abstractmethod
    def save_fitted_state(self):
        pass

    @abstractmethod
    def load_fitted_state(self):
        pass

    def fit(self, X, y):
        self.save_dir = self._make_save_dir()
        self.classes_ = np.unique(y)
        _spawn_worker(_fit_worker, (self, X, y))
        return self

    def predict(self, X):
        return _spawn_worker(_predict_worker, (self, X))

    def predict_proba(self, X):
        return _spawn_worker(_predict_proba_worker, (self, X))
