"""
Make pipeline for SCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
.. [2] https://github.com/NeuroTechX/moabb/blob/v1.1.2/moabb/pipelines/deep_learning.py
"""

import torch
import torch.nn as nn
import numpy as np
from os import getenv
from dotenv import load_dotenv
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler, Callback
from moabb.pipelines.features import Convert_Epoch_Array, StandardScaler_Epoch
from sklearn.pipeline import make_pipeline
from src.pipelines.pipeline import Pipeline
from src.pipelines.deep_learning.utilities import ToFloat32


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, drop_rate=0.5):
        super(ShallowConvNet, self).__init__()
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=40,
            kernel_size=(1, 25),
            stride=1,
            bias=True,
        )
        self.conv_spatial = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=(n_channels, 1),
            stride=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(40, momentum=0.1, eps=1e-5)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=drop_rate)
        conv_out_time = n_times - 24
        pool_out_time = (conv_out_time - 75) // 15 + 1
        self.flatten_size = 40 * pool_out_time
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.clamp(x, min=1e-7, max=10000)
        x = torch.log(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MaxNormCallback(Callback):
    def on_batch_end(self, net, **kwargs):
        module = net.module_.module

        # Constrain temporal convolution
        w = module.conv_temporal.weight.data
        norm = w.norm(2, dim=(0, 1, 2), keepdim=True)
        desired = torch.clamp(norm, 0, 2.0)
        w *= desired / (norm + 1e-8)

        # Constrain spatial convolution
        w = module.conv_spatial.weight.data
        norm = w.norm(2, dim=(0, 1, 2), keepdim=True)
        desired = torch.clamp(norm, 0, 2.0)
        w *= desired / (norm + 1e-8)

        # Constrain fully connected layer
        w = module.fc.weight.data
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, 0.5)
        w *= desired / (norm + 1e-8)


class SkorchShallowConvNet(NeuralNetClassifier):
    def __init__(
        self,
        lr=0.001,
        drop_rate=0.5,
        max_epochs=300,
        batch_size=64,
        verbose=0,
        random_state=None,
        train_split=0.2,
        device="cuda",
        callbacks=None,
        **kwargs,
    ):
        callbacks = list(callbacks) if callbacks is not None else list()
        callbacks.append(("max_norm", MaxNormCallback()))
        train_split = ValidSplit(train_split)
        super().__init__(
            module=ShallowConvNet,
            module__drop_rate=drop_rate,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=train_split,
            verbose=verbose,
            device=device,
            callbacks=callbacks,
            **kwargs,
        )

    def initialize_module(self):
        super().initialize_module()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            gpu_ids = list(range(torch.cuda.device_count()))
            self.module_ = nn.DataParallel(self.module_, device_ids=gpu_ids)
        return self

    def fit(self, X, y=None, **fit_params):
        n_channels = X.shape[1]
        n_times = X.shape[2]
        n_classes = len(np.unique(y))
        self.set_params(
            module__n_channels=n_channels,
            module__n_times=n_times,
            module__n_classes=n_classes,
        )
        return super().fit(X, y, **fit_params)


class SCNN(Pipeline):
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))

        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def pipeline(self):
        return {
            "SCNN": make_pipeline(
                Convert_Epoch_Array(),
                StandardScaler_Epoch(),
                ToFloat32(),
                SkorchShallowConvNet(
                    lr=0.001,
                    drop_rate=0.5,
                    max_epochs=300,
                    batch_size=64,
                    verbose=0,
                    random_state=self.random_state,
                    train_split=0.2,
                    callbacks=[
                        (
                            "early_stopping",
                            EarlyStopping(
                                monitor="valid_loss", patience=75, lower_is_better=True, load_best=True
                            ),
                        ),
                        (
                            "lr_scheduler",
                            LRScheduler(
                                policy="ReduceLROnPlateau",
                                monitor="valid_loss",
                                patience=75,
                                factor=0.5,
                                mode="min",
                            ),
                        ),
                    ],
                ),
            )
        }

    def params(self):
        return {}
