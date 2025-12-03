"""
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
    - https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
    - https://github.com/NeuroTechX/moabb/blob/v1.1.2/moabb/pipelines/deep_learning.py
"""

from moabb.pipelines.features import Resampler_Epoch, Convert_Epoch_Array, StandardScaler_Epoch
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense
)
from keras.models import Model
from keras.constraints import max_norm
from tensorflow.keras import backend as K


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


class ShallowConvNet(KerasClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss = "sparse_categorical_crossentropy"
        self.optimizer = Adam(learning_rate=0.001)
        self.epochs = 300
        self.batch_size = 64
        self.verbose = 0
        self.random_state = 42
        self.validation_split = 0.2

    def _keras_build_fn(self, compile_kwargs):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        block = Conv2D(
            40,
            (1, 25),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(input_main)
        block = Conv2D(
            40,
            (self.X_shape_[1], 1),
            use_bias=False,
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(block)
        block = BatchNormalization(epsilon=1e-05, momentum=0.9)(block)
        block = Activation(square)(block)
        block = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block)
        block = Activation(log)(block)
        block = Dropout(0.5)(block)
        flatten = Flatten()(block)
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model
    

scnn = Pipeline([
    ("re", Resampler_Epoch(sfreq=250)),
    ("cea", Convert_Epoch_Array()),
    ("sse", StandardScaler_Epoch()),
    ("scnn", ShallowConvNet(
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=75),
            ReduceLROnPlateau(monitor="val_loss", patience=75, factor=0.5)
        ]
    ))
])
