"""
Collect additional metrics from trained models
References:
    - https://github.com/NeuroTechX/moabb/blob/develop/moabb/analysis/results.py
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/noplot_load_model.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_within_session_splitter.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectSplitter.html#moabb.evaluations.CrossSubjectSplitter  # noqa: E501
    - https://github.com/NeuroTechX/moabb/blob/develop/moabb/evaluations/evaluations.py#L581-L772  # noqa: E501
"""

import numpy as np
from os import path, getenv
from dotenv import load_dotenv
from pickle import load
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)
from moabb.evaluations import CrossSubjectEvaluation, CrossSubjectSplitter
from moabb.analysis.results import Results
from src.evaluation.paradigms import LogLossLeftRightImagery


def ece_score(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bin_edges) - 1
    counts = np.array([np.sum(bin_idx == i) for i in range(len(prob_true))])
    return np.sum(np.abs(prob_true - prob_pred) * counts / len(y_prob))


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read raw results
df = Results(
    CrossSubjectEvaluation, LogLossLeftRightImagery, hdf5_path=data_path
).to_dataframe()

# Initialize combined results
df[["nll", "brier", "ece", "mcc", "acc", "auroc"]] = np.nan

# Define metrics collection parameters
params = [
    # PhysionetMI
    (
        "PhysionetMotorImagery",
        "csp_lda",
        LogLossLeftRightImagery(resample=160),
        PhysionetMI(),
        False,
    ),
    (
        "PhysionetMotorImagery",
        "csp_svm",
        LogLossLeftRightImagery(resample=160),
        PhysionetMI(),
        False,
    ),
    (
        "PhysionetMotorImagery",
        "ts_lr",
        LogLossLeftRightImagery(resample=160),
        PhysionetMI(),
        False,
    ),
    (
        "PhysionetMotorImagery",
        "ts_svm",
        LogLossLeftRightImagery(resample=160),
        PhysionetMI(),
        False,
    ),
    (
        "PhysionetMotorImagery",
        "scnn",
        LogLossLeftRightImagery(resample=160),
        PhysionetMI(),
        True,
    ),
    (
        "PhysionetMotorImagery",
        "dcnn",
        LogLossLeftRightImagery(resample=160),
        PhysionetMI(),
        True,
    ),
    # Lee2019_MI
    ("", "csp_lda", LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    ("", "csp_svm", LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    ("", "ts_lr", LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    ("", "ts_svm", LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    ("", "scnn", LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True),
    ("", "dcnn", LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True),
    # Cho2017
    ("", "csp_lda", LogLossLeftRightImagery(resample=512), Cho2017(), False),
    ("", "csp_svm", LogLossLeftRightImagery(resample=512), Cho2017(), False),
    ("", "ts_lr", LogLossLeftRightImagery(resample=512), Cho2017(), False),
    ("", "ts_svm", LogLossLeftRightImagery(resample=512), Cho2017(), False),
    ("", "scnn", LogLossLeftRightImagery(resample=512), Cho2017(), True),
    ("", "dcnn", LogLossLeftRightImagery(resample=512), Cho2017(), True),
    # Schirrmeister2017
    ("", "csp_lda", LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    ("", "csp_svm", LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    ("", "ts_lr", LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    ("", "ts_svm", LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    ("", "scnn", LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True),
    ("", "dcnn", LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True),
    # Shin2017A
    ("", "csp_lda", LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    ("", "csp_svm", LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    ("", "ts_lr", LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    ("", "ts_svm", LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    ("", "scnn", LogLossLeftRightImagery(resample=200), Shin2017A(), True),
    ("", "dcnn", LogLossLeftRightImagery(resample=200), Shin2017A(), True),
    # BNCI2014_001
    (
        "BNCI2014-001",
        "csp_lda",
        LogLossLeftRightImagery(resample=250),
        BNCI2014_001(),
        False,
    ),
    (
        "BNCI2014-001",
        "csp_svm",
        LogLossLeftRightImagery(resample=250),
        BNCI2014_001(),
        False,
    ),
    (
        "BNCI2014-001",
        "ts_lr",
        LogLossLeftRightImagery(resample=250),
        BNCI2014_001(),
        False,
    ),
    (
        "BNCI2014-001",
        "ts_svm",
        LogLossLeftRightImagery(resample=250),
        BNCI2014_001(),
        False,
    ),
    (
        "BNCI2014-001",
        "scnn",
        LogLossLeftRightImagery(resample=250),
        BNCI2014_001(),
        True,
    ),
    (
        "BNCI2014-001",
        "dcnn",
        LogLossLeftRightImagery(resample=250),
        BNCI2014_001(),
        True,
    ),
]

# Generate combined results
for name, pipeline, paradigm, dataset, epochs in params:
    # Prepare dataset
    X, y, metadata = paradigm.get_data(dataset=dataset, return_epochs=epochs)
    le = LabelEncoder()
    y = le.fit_transform(y)
    groups = metadata.subject.values
    sessions = metadata.session.values

    # Split dataset into same folds as evaluation
    cv = CrossSubjectSplitter(cv_class=GroupKFold, **dict(n_splits=5))

    # Read models and compute scores
    for cv_ind, (train, test) in enumerate(cv.split(y, metadata)):
        # Load classifier from one fold of evaluation
        subject = groups[test[0]]
        with open(
            path.join(
                data_path,
                "Search_CrossSubject",
                name,
                str(subject),
                pipeline,
                f"fitted_model_{cv_ind}.pkl",
            ),
            "rb",
        ) as pickle_file:
            model = load(pickle_file)

        # Measure scores per session same as evaluation
        for session in np.unique(sessions[test]):
            # Compute actual and expected predictions
            ix = sessions[test] == session
            X_session = X[test[ix]]
            y_true = y[test[ix]]
            y_pred = model.predict(X_session)
            y_prob = model.predict_proba(X_session)[:, 1]

            # Score predictions
            nll = log_loss(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            ece = ece_score(y_true, y_prob)
            mcc = matthews_corrcoef(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            auroc = roc_auc_score(y_true, y_prob)

            # Combine raw results and additional scores
            df.loc[
                (df["subject"] == str(subject))
                & (df["session"] == session)
                & (df["dataset"] == name)
                & (df["pipeline"] == pipeline),
                ["nll", "brier", "ece", "mcc", "acc", "auroc"],
            ] = [nll, brier, ece, mcc, acc, auroc]

# Save combined results to disk
df_rounded = df.round(6)
df_rounded.to_csv(path.join(data_path, "results.csv"), index=False)
