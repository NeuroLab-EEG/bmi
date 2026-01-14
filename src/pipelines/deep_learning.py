"""
Build deep learning pipelines.

References
----------
.. [1] https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
.. [2] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml  # noqa: E501
.. [3] https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
.. [4] https://github.com/NeuroTechX/moabb/blob/v1.1.2/moabb/pipelines/deep_learning.py
.. [5] https://adriangb.com/scikeras/stable/generated/scikeras.wrappers.KerasClassifier.html  # noqa: E501
.. [6] https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism  # noqa: E501
"""

import tensorflow as tf
from os import getenv
from dotenv import load_dotenv
from moabb.pipelines.features import Convert_Epoch_Array, StandardScaler_Epoch
from sklearn.pipeline import make_pipeline
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
    Dense,
    MaxPooling2D,
)
from keras.models import Model
from keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from src.pipelines.pipeline import Pipeline


# Load environment variables
load_dotenv()
random_state = int(getenv("RANDOM_STATE"))

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(random_state)
tf.config.experimental.enable_op_determinism()


@register_keras_serializable()
def square(x):
    return K.square(x)


@register_keras_serializable()
def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


class ShallowConvNet(KerasClassifier):
    def __init__(
        self,
        loss,
        optimizer,
        epochs,
        batch_size,
        verbose,
        random_state,
        validation_split,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split

    def _keras_build_fn(self, compile_kwargs):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        block1 = Conv2D(
            40,
            (1, 25),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(input_main)
        block1 = Conv2D(
            40,
            (self.X_shape_[1], 1),
            use_bias=False,
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(0.5)(block1)
        flatten = Flatten()(block1)
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)
        model.compile(
            loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"]
        )
        return model


class DeepConvNet(KerasClassifier):
    def __init__(
        self,
        loss,
        optimizer,
        epochs,
        batch_size,
        verbose,
        random_state,
        validation_split,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split

    def _keras_build_fn(self, compile_kwargs):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        block1 = Conv2D(
            25,
            (1, 10),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(input_main)
        block1 = Conv2D(
            25,
            (self.X_shape_[1], 1),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation("elu")(block1)
        block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
        block1 = Dropout(0.5)(block1)

        block2 = Conv2D(50, (1, 10), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
            block1
        )
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2 = Activation("elu")(block2)
        block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
        block2 = Dropout(0.5)(block2)

        block3 = Conv2D(100, (1, 10), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
            block2
        )
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3 = Activation("elu")(block3)
        block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
        block3 = Dropout(0.5)(block3)

        block4 = Conv2D(200, (1, 10), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
            block3
        )
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4 = Activation("elu")(block4)
        block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
        block4 = Dropout(0.5)(block4)
        flatten = Flatten()(block4)
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)
        model.compile(
            loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"]
        )
        return model


class SCNN(Pipeline):
    def pipeline(self):
        return {
            "SCNN": make_pipeline(
                Convert_Epoch_Array(),
                StandardScaler_Epoch(),
                ShallowConvNet(
                    loss="sparse_categorical_crossentropy",
                    optimizer=Adam(learning_rate=0.001),
                    epochs=300,
                    batch_size=64,
                    verbose=0,
                    random_state=random_state,
                    validation_split=0.2,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=75),
                        ReduceLROnPlateau(monitor="val_loss", patience=75, factor=0.5),
                    ],
                ),
            )
        }

    def params(self):
        return {}


class DCNN(Pipeline):
    def pipeline(self):
        return {
            "DCNN": make_pipeline(
                Convert_Epoch_Array(),
                StandardScaler_Epoch(),
                DeepConvNet(
                    loss="sparse_categorical_crossentropy",
                    optimizer=Adam(learning_rate=0.001),
                    epochs=300,
                    batch_size=64,
                    verbose=0,
                    random_state=random_state,
                    validation_split=0.2,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=75),
                        ReduceLROnPlateau(monitor="val_loss", patience=75, factor=0.5),
                    ],
                ),
            )
        }

    def params(self):
        return {}
