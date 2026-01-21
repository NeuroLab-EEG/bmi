"""
Make pipeline for DCNN.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from src.pipelines.pipeline import Pipeline
from src.pipelines.deep_learning.utilities import ToFloat32


class DeepConvNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, drop_rate=0.5):
        super(DeepConvNet, self).__init__()

        # Block 1
        self.conv_temporal1 = nn.Conv2d(
            in_channels=1,
            out_channels=25,
            kernel_size=(1, 10),
            stride=1,
            bias=True,
        )
        self.conv_spatial1 = nn.Conv2d(
            in_channels=25,
            out_channels=25,
            kernel_size=(n_channels, 1),
            stride=1,
            bias=True,
        )
        self.batch_norm1 = nn.BatchNorm2d(25, momentum=0.1, eps=1e-5)
        self.elu1 = nn.ELU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(p=drop_rate)

        # Block 2
        self.conv2 = nn.Conv2d(
            in_channels=25,
            out_channels=50,
            kernel_size=(1, 10),
            stride=1,
            bias=True,
        )
        self.batch_norm2 = nn.BatchNorm2d(50, momentum=0.1, eps=1e-5)
        self.elu2 = nn.ELU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(p=drop_rate)

        # Block 3
        self.conv3 = nn.Conv2d(
            in_channels=50,
            out_channels=100,
            kernel_size=(1, 10),
            stride=1,
            bias=True,
        )
        self.batch_norm3 = nn.BatchNorm2d(100, momentum=0.1, eps=1e-5)
        self.elu3 = nn.ELU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(p=drop_rate)

        # Block 4
        self.conv4 = nn.Conv2d(
            in_channels=100,
            out_channels=200,
            kernel_size=(1, 10),
            stride=1,
            bias=True,
        )
        self.batch_norm4 = nn.BatchNorm2d(200, momentum=0.1, eps=1e-5)
        self.elu4 = nn.ELU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout4 = nn.Dropout(p=drop_rate)

        # Fully connected layer
        time_after_conv1 = n_times - 9
        time_after_pool1 = (time_after_conv1 - 3) // 3 + 1
        time_after_conv2 = time_after_pool1 - 9
        time_after_pool2 (time_after_conv2 - 3) // 3 + 1
        time_after_conv3 = time_after_pool2 - 9
        time_after_pool3 = (time_after_conv3 - 3) // 3 + 1
        time_after_conv4 = time_after_pool3 - 9
        time_after_pool4 = (time_after_conv4 - 3) // 3 + 1
        self.flatten_size = 200 * time_after_pool4
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal1(x)
        x = self.conv_spatial1(x)
        x = self.batch_norm1(x)
        x = self.elu1(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.elu2(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.elu3(x)
        x = self.max_pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.elu4(x)
        x = self.max_pool4(x)
        x = self.dropout4(x)

        # Fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MaxNormCallback(Callback):
    def on_batch_end(self, net, **kwargs):
        module = net.module_.module

        # Constrain block 1 temporal convolution
        w = module.conv_temporal1.weight.data
        norm = w.norm(2, dim=(0, 1, 2), keepdim=True)
        desired = torch.clamp(norm, 0, 2.0)
        w *= (desired / norm + 1e-8)

        # Constrain block 1 spatial convolution
        w = module.conv_spatial1.weight.data
        norm = w.norm(2, dim=(0, 1, 2), keepdim=True)
        desired = torch.clamp(norm, 0, 2.0)
        w *= (desired / (norm + 1e-8))

        # Constrain blocks 2-4 convolution layers
        for conv_layer in [module.conv2, module.conv3, module.conv4]:
            w = conv_layer.weight.data
            norm = w.norm(2, dim=(0, 1, 2), keepdim=True)
            desired = torch.clamp(norm, 0, 2.0)
            w *= (desired / (norm + 1e-8))

        # Constrain fully connected layer
        w = module.fc.weight.data
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, 0.5)
        w *= (desired / (norm + 1e-8))


class SkorchDeepConvNet(NeuralNetClassifier):
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
            module=DeepConvNet,
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


class DCNN(Pipeline):
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))

        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def pipeline(self):
        return {
            "dcnn": make_pipeline(
                Convert_Epoch_Array(),
                StandardScaler_Epoch(),
                ToFloat32(),
                SkorchDeepConvNet(
                    lr=0.001,
                    drop_rate=0.5,
                    max_epochs=300,
                    batch_size=64,
                    verbose=0,
                    random_state=self.random_state,
                    train_split=0.2,
                    callbacks=[
                        ("early_stopping", EarlyStopping(
                            monitor="valid_loss",
                            patience=75,
                            lower_is_better=True,
                            load_best=True,
                        )),
                        ("lr_scheduler", LRScheduler(
                            policy="ReduceLROnPlateau",
                            monitor="valid_loss",
                            patience=75,
                            factor=0.5,
                            mode="min",
                        ))
                    ]
                )
            )
        }

    def params(self):
        return {}
