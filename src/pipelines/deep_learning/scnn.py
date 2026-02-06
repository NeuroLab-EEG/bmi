"""
Make pipeline for SCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
.. [2] https://github.com/NeuroTechX/moabb/blob/v1.1.2/moabb/pipelines/deep_learning.py
.. [3] https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html
.. [4] https://braindecode.org/stable/generated/braindecode.models.ShallowFBCSPNet.html#braindecode.models.ShallowFBCSPNet
.. [5] https://braindecode.org/stable/generated/braindecode.classifier.EEGClassifier.html#braindecode.classifier.EEGClassifier
"""

import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.classifier import EEGClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from sklearn.pipeline import make_pipeline
from src.pipelines import Pipeline


class SCNN(Pipeline):
    def build(self):
        set_random_seeds(seed=self.random_state, cuda=torch.cuda.is_available())
        return {
            "SCNN": make_pipeline(
                EEGClassifier(
                    ShallowFBCSPNet(
                        n_chans=self.n_features,
                        n_outputs=self.n_classes,
                        n_times=self.n_times,
                    ),
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
                ),
            )
        }
