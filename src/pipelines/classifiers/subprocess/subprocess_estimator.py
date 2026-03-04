import uuid
import numpy as np
import multiprocessing as mp
from os import path, makedirs
from datetime import datetime
from abc import abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin


def _fit_worker(subprocess_estimator, X, y, queue):
    try:
        subprocess_estimator.estimator.fit(X, y)
        subprocess_estimator.save_fitted_state()
        queue.put(("ok", None))
    except Exception as e:
        queue.put(("err", e))


def _predict_worker(subprocess_estimator, X, queue):
    try:
        subprocess_estimator.load_fitted_state()
        result = subprocess_estimator.estimator.predict(X)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", e))


def _predict_proba_worker(subprocess_estimator, X, queue):
    try:
        subprocess_estimator.load_fitted_state()
        result = subprocess_estimator.estimator.predict_proba(X)
        queue.put(("ok", result))
    except Exception as e:
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


class SubprocessEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, save_dir):
        self.estimator = estimator
        self.save_dir = save_dir
        self._save_path = None

    def __sklearn_is_fitted__(self):
        return True

    def _make_save_path(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = path.join(self.save_dir, self.__class__.__name__, timestamp)
        makedirs(save_path, exist_ok=True)
        return save_path

    @abstractmethod
    def save_fitted_state(self):
        pass

    @abstractmethod
    def load_fitted_state(self):
        pass

    def fit(self, X, y):
        self._save_path = self._make_save_path()
        self.classes_ = np.unique(y)
        _spawn_worker(_fit_worker, (self, X, y))
        return self

    def predict(self, X):
        return _spawn_worker(_predict_worker, (self, X))

    def predict_proba(self, X):
        return _spawn_worker(_predict_proba_worker, (self, X))
