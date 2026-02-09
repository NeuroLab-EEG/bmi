"""
Make neural network scikit-learn classifier.

References
----------
.. [1] https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html
.. [2] https://braindecode.org/stable/generated/braindecode.classifier.EEGClassifier.html#braindecode.classifier.EEGClassifier
"""

import torch
from braindecode.util import set_random_seeds
from braindecode.classifier import EEGClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from sklearn.base import BaseEstimator, ClassifierMixin


class NeuralNetworkBase(ClassifierMixin, BaseEstimator):
    def __init__(self, classifier, random_state=None):
        self.classifier = classifier
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        set_random_seeds(seed=self.random_state, cuda=torch.cuda.is_available())
        self.model_ = EEGClassifier(
            module=self.classifier,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.001,
            optimizer__weight_decay=0.01,
            max_epochs=300,
            batch_size=64,
            train_split=ValidSplit(cv=0.2, random_state=self.random_state),
            iterator_train__shuffle=True,
            iterator_train__num_workers=0,
            iterator_valid__shuffle=False,
            iterator_valid__num_workers=0,
            verbose=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            callbacks=[
                EarlyStopping(
                    monitor="valid_loss",
                    patience=150,
                    threshold=0.0001,
                    lower_is_better=True,
                    load_best=True,
                ),
                LRScheduler(
                    policy="ReduceLROnPlateau",
                    monitor="valid_loss",
                    patience=50,
                    factor=0.5,
                    mode="min",
                    min_lr=1e-6,
                ),
            ],
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
