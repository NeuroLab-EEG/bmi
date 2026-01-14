"""
Extract additional scores from trained ML models.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/moabb/analysis/results.py
.. [2] https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/noplot_load_model.html
.. [3] https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_within_session_splitter.html
.. [4] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectSplitter.html#moabb.evaluations.CrossSubjectSplitter
.. [5] https://github.com/NeuroTechX/moabb/blob/develop/moabb/evaluations/evaluations.py#L581-L772
"""

import pandas as pd
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
from sklearn.preprocessing import LabelEncoder
from moabb.evaluations import CrossSubjectEvaluation, CrossSubjectSplitter
from moabb.analysis.results import Results
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)
from src.paradigm.paradigm import LogLossLeftRightImagery


def ece_score(y_true, y_prob, n_bins=10):
    """
    The expected calibration error (ECE) for binary classification.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        weight = np.mean(in_bin)
        if weight > 0:
            positive = np.mean(y_true[in_bin])
            confidence = np.mean(y_prob[in_bin])
            ece += np.abs(positive - confidence) * weight

    return ece


class Scores:
    """
    Helper class to extract multiple scores from ML classifiers.
    """

    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.data_path = getenv("DATA_PATH")

        # Read raw results
        self.raw = Results(
            CrossSubjectEvaluation, LogLossLeftRightImagery, hdf5_path=self.data_path
        ).to_dataframe()

        # Initialize final results
        self.scores = pd.DataFrame(
            columns=[
                "score",
                "time",
                "samples",
                "carbon_emission",
                "subject",
                "session",
                "channels",
                "n_sessions",
                "dataset",
                "pipeline",
                "codecarbon_task_name",
                "nll",
                "brier",
                "ece",
                "mcc",
                "acc",
                "auroc",
            ]
        )

    def extract(self):
        # Iterate over parameters from dataset-pipeline combinations
        for dir_name, pipeline_name, paradigm, resample, dataset, epochs, splits in self.params():
            # Preprocess dataset by its paradigm
            X, y, metadata = paradigm(resample=resample).get_data(
                dataset=dataset(),
                return_epochs=epochs,
            )
            y = LabelEncoder().fit_transform(y)
            groups = metadata.subject.values
            sessions = metadata.session.values

            # Create same cross-validation splits from training evaluation
            cv = CrossSubjectSplitter(cv_class=GroupKFold, **dict(n_splits=splits))

            # Iterate over folds from cross-validation
            for cv_ind, (train, test) in enumerate(cv.split(y, metadata)):
                # Read saved ML model from disk
                subject = groups[test[0]]
                with open(
                    path.join(
                        self.data_path,
                        "Search_CrossSubject",
                        dir_name,
                        str(subject),
                        pipeline_name,
                        f"fitted_model_{cv_ind}.pkl",
                    ),
                    "rb",
                ) as pickle_file:
                    model = load(pickle_file)

                # Iterate over sessions from dataset same as training evaluation
                for session in np.unique(sessions[test]):
                    # Compute true and predicted labels
                    ix = sessions[test] == session
                    X_session = X[test[ix]]
                    y_true = y[test[ix]]
                    y_pred = model.predict(X_session)
                    y_prob = model.predict_proba(X_session)[:, 1]

                    # Score predictive performance
                    nll = log_loss(y_true, y_prob)
                    brier = brier_score_loss(y_true, y_prob)
                    ece = ece_score(y_true, y_prob)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    acc = accuracy_score(y_true, y_pred)
                    auroc = roc_auc_score(y_true, y_prob)

                    # Lookup current row from raw results
                    cur = self.results[
                        (self.results["subject"] == str(subject))
                        & (self.results["session"] == session)
                        & (self.results["dataset"] == dir_name)
                        & (self.results["pipeline"] == pipeline_name)
                    ].iloc[0]

                    # Save new row with all scores
                    self.scores.loc[len(self.scores)] = {
                        "score": cur["score"],
                        "time": cur["time"],
                        "samples": cur["samples"],
                        "carbon_emission": cur["carbon_emission"],
                        "subject": cur["subject"],
                        "session": cur["session"],
                        "channels": cur["channels"],
                        "n_sessions": cur["n_sessions"],
                        "dataset": cur["dataset"],
                        "pipeline": cur["pipeline"],
                        "codecarbon_task_name": cur["codecarbon_task_name"],
                        "nll": nll,
                        "brier": brier,
                        "ece": ece,
                        "mcc": mcc,
                        "acc": acc,
                        "auroc": auroc,
                    }

        # Save all scores to disk
        self.scores.to_csv(path.join(self.data_path, "scores.csv"), index=False)

    def params(self):
        yield from self.physionetmi()
        yield from self.lee2019_mi()
        yield from self.cho2017()
        yield from self.schirrmeister2017()
        yield from self.shin2017a()
        yield from self.bnci2014_001()

    def physionetmi(self):
        yield ("PhysionetMotorImagery", "CSP+LDA", LogLossLeftRightImagery, 160, PhysionetMI, False, 10)
        yield ("PhysionetMotorImagery", "CSP+SVM", LogLossLeftRightImagery, 160, PhysionetMI, False, 10)
        yield ("PhysionetMotorImagery", "TS+LR", LogLossLeftRightImagery, 160, PhysionetMI, False, 10)
        yield ("PhysionetMotorImagery", "TS+SVM", LogLossLeftRightImagery, 160, PhysionetMI, False, 10)
        yield ("PhysionetMotorImagery", "SCNN", LogLossLeftRightImagery, 160, PhysionetMI, True, 10)
        yield ("PhysionetMotorImagery", "DCNN", LogLossLeftRightImagery, 160, PhysionetMI, True, 10)

    def lee2019_mi(self):
        yield ("Lee2019-MI", "CSP+LDA", LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10)
        yield ("Lee2019-MI", "CSP+SVM", LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10)
        yield ("Lee2019-MI", "TS+LR", LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10)
        yield ("Lee2019-MI", "TS+SVM", LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10)
        yield ("Lee2019-MI", "SCNN", LogLossLeftRightImagery, 1000, Lee2019_MI, True, 10)
        yield ("Lee2019-MI", "DCNN", LogLossLeftRightImagery, 1000, Lee2019_MI, True, 10)

    def cho2017(self):
        yield ("Cho2017", "CSP+LDA", LogLossLeftRightImagery, 512, Cho2017, False, 10)
        yield ("Cho2017", "CSP+SVM", LogLossLeftRightImagery, 512, Cho2017, False, 10)
        yield ("Cho2017", "TS+LR", LogLossLeftRightImagery, 512, Cho2017, False, 10)
        yield ("Cho2017", "TS+SVM", LogLossLeftRightImagery, 512, Cho2017, False, 10)
        yield ("Cho2017", "SCNN", LogLossLeftRightImagery, 512, Cho2017, True, 10)
        yield ("Cho2017", "DCNN", LogLossLeftRightImagery, 512, Cho2017, True, 10)

    def schirrmeister2017(self):
        yield ("Schirrmeister2017", "CSP+LDA", LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5)
        yield ("Schirrmeister2017", "CSP+SVM", LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5)
        yield ("Schirrmeister2017", "TS+LR", LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5)
        yield ("Schirrmeister2017", "TS+SVM", LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5)
        yield ("Schirrmeister2017", "SCNN", LogLossLeftRightImagery, 500, Schirrmeister2017, True, 5)
        yield ("Schirrmeister2017", "DCNN", LogLossLeftRightImagery, 500, Schirrmeister2017, True, 5)

    def shin2017a(self):
        yield ("Shin2017A", "CSP+LDA", LogLossLeftRightImagery, 200, Shin2017A, False, 5)
        yield ("Shin2017A", "CSP+SVM", LogLossLeftRightImagery, 200, Shin2017A, False, 5)
        yield ("Shin2017A", "TS+LR", LogLossLeftRightImagery, 200, Shin2017A, False, 5)
        yield ("Shin2017A", "TS+SVM", LogLossLeftRightImagery, 200, Shin2017A, False, 5)
        yield ("Shin2017A", "SCNN", LogLossLeftRightImagery, 200, Shin2017A, True, 5)
        yield ("Shin2017A", "DCNN", LogLossLeftRightImagery, 200, Shin2017A, True, 5)

    def bnci2014_001(self):
        yield ("BNCI2014-001", "CSP+LDA", LogLossLeftRightImagery, 250, BNCI2014_001, False, 9)
        yield ("BNCI2014-001", "CSP+SVM", LogLossLeftRightImagery, 250, BNCI2014_001, False, 9)
        yield ("BNCI2014-001", "TS+LR", LogLossLeftRightImagery, 250, BNCI2014_001, False, 9)
        yield ("BNCI2014-001", "TS+SVM", LogLossLeftRightImagery, 250, BNCI2014_001, False, 9)
        yield ("BNCI2014-001", "SCNN", LogLossLeftRightImagery, 250, BNCI2014_001, True, 9)
        yield ("BNCI2014-001", "DCNN", LogLossLeftRightImagery, 250, BNCI2014_001, True, 9)


Scores().extract()
