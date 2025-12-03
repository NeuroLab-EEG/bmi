"""
Definition of classifier piplines
References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
    - https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
    - https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_ShallowConvNet.yml
    - https://github.com/NeuroTechX/moabb/blob/v1.1.2/pipelines/Keras_DeepConvNet.yml
"""

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from moabb.pipelines.features import Resampler_Epoch, Convert_Epoch_Array, StandardScaler_Epoch
from moabb.pipelines.deep_learning import KerasShallowConvNet, KerasDeepConvNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


csp_lda_pipeline = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("csp", CSP(nfilter=6)),
    ("lda", LDA(solver="svd"))
])

csp_svm_pipeline = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("csp", CSP(nfilter=6)),
    ("svc", SVC(kernel="linear"))
])

csp_svm_grid = {
    "csp__nfilter": [2, 3, 4, 5, 6, 7, 8],
    "svc__C": [0.5, 1, 1.5],
    "svc__kernel": ["rbf", "linear"]
}

ts_lr_pipeline = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("ts", TangentSpace(metric="riemann")),
    ("lr", LogisticRegression(C=1.0))
])

ts_svm_pipeline = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("ts", TangentSpace(metric="riemann")),
    ("svm", SVC(kernel="linear"))
])

ts_svm_grid = {
    "svc__C": [0.5, 1, 1.5],
    "svc__kernel": ["rbf", "linear"]
}

scnn_pipeline = Pipeline([
    ("re", Resampler_Epoch(sfreq=250)),
    ("cea", Convert_Epoch_Array()),
    ("sse", StandardScaler_Epoch()),
    ("scnn", KerasShallowConvNet(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        epochs=300,
        batch_size=64,
        verbose=0,
        random_state=42,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=75),
            ReduceLROnPlateau(monitor="val_loss", patience=75, factor=0.5)
        ]
    ))
])

dcnn_pipeline = Pipeline([
    ("re", Resampler_Epoch(sfreq=250)),
    ("cea", Convert_Epoch_Array()),
    ("sse", StandardScaler_Epoch()),
    ("dcnn", KerasDeepConvNet(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        epochs=300,
        batch_size=64,
        verbose=0,
        random_state=42,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=75),
            ReduceLROnPlateau(monitor="val_loss", patience=75, factor=0.5)
        ]
    ))
])


def pipelines():
    return {
        "csp_lda": csp_lda_pipeline,
        "csp_svm": csp_svm_pipeline,
        "ts_lr": ts_lr_pipeline,
        "ts_svm": ts_svm_pipeline,
        "scnn": scnn_pipeline,
        "dcnn": dcnn_pipeline
    }


def grids():
    return {
        "csp_svm": csp_svm_grid,
        "ts_svm": ts_svm_grid
    }
