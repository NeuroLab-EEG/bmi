"""
Merge training compute cost and testing predictive performance.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/moabb/analysis/results.py
.. [2] https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/noplot_load_model.html  # noqa: E501
.. [3] https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_within_session_splitter.html  # noqa: E501
.. [4] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectSplitter.html#moabb.evaluations.CrossSubjectSplitter  # noqa: E501
.. [5] https://github.com/NeuroTechX/moabb/blob/develop/moabb/evaluations/evaluations.py#L581-L772  # noqa: E501
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
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from moabb.evaluations import CrossSubjectEvaluation, CrossSubjectSplitter
from moabb.analysis.results import Results
from src.paradigm.paradigm import LogLossLeftRightImagery
from src.evaluation.results_parameters import parameters, columns


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


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read raw results
results = Results(
    CrossSubjectEvaluation, LogLossLeftRightImagery, hdf5_path=data_path
).to_dataframe()

# Read emissions
emissions = pd.read_csv(path.join(data_path, "emissions.csv"))

# Initialize merged scores and emissions
data = pd.DataFrame(columns=columns)

# Generate merged scores and emissions
for name, new_name, pipeline, paradigm, dataset, epochs, n_splits in parameters:
    # Prepare dataset
    X, y, metadata = paradigm.get_data(dataset=dataset, return_epochs=epochs)
    le = LabelEncoder()
    y = le.fit_transform(y)
    groups = metadata.subject.values
    sessions = metadata.session.values

    # Split dataset into same folds as evaluation
    cv = CrossSubjectSplitter(cv_class=GroupKFold, **dict(n_splits=n_splits))

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

            # Additional score predictions
            nll = log_loss(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            ece = ece_score(y_true, y_prob)
            mcc = matthews_corrcoef(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            auroc = roc_auc_score(y_true, y_prob)

            # Lookup single rows
            results_row = results[
                (results["subject"] == str(subject))
                & (results["session"] == session)
                & (results["dataset"] == name)
                & (results["pipeline"] == pipeline)
            ].iloc[0]
            emissions_row = emissions[emissions["experiment_id"] == f"{pipeline}+{new_name}"].iloc[0]

            # Merge scores and emissions
            data.loc[len(data)] = {
                "samples": results_row["samples"],
                "subject": results_row["subject"],
                "session": results_row["session"],
                "channels": results_row["channels"],
                "n_sessions": results_row["n_sessions"],
                "dataset": new_name,
                "pipeline": pipeline,
                "nll": nll,
                "brier": brier,
                "ece": ece,
                "mcc": mcc,
                "acc": acc,
                "auroc": auroc,
                "timestamp": emissions_row["timestamp"],
                "duration": emissions_row["duration"],
                "emissions": emissions_row["emissions"],
                "emissions_rate": emissions_row["emissions_rate"],
                "cpu_power": emissions_row["cpu_power"],
                "gpu_power": emissions_row["gpu_power"],
                "ram_power": emissions_row["ram_power"],
                "cpu_energy": emissions_row["cpu_energy"],
                "gpu_energy": emissions_row["gpu_energy"],
                "ram_energy": emissions_row["ram_energy"],
                "energy_consumed": emissions_row["energy_consumed"],
                "water_consumed": emissions_row["water_consumed"],
                "country_name": emissions_row["country_name"],
                "country_iso_code": emissions_row["country_iso_code"],
                "region": emissions_row["region"],
                "os": emissions_row["os"],
                "cpu_count": emissions_row["cpu_count"],
                "cpu_model": emissions_row["cpu_model"],
                "gpu_count": emissions_row["gpu_count"],
                "gpu_model": emissions_row["gpu_model"],
                "longitude": emissions_row["longitude"],
                "latitude": emissions_row["latitude"],
                "ram_total_size": emissions_row["ram_total_size"],
                "tracking_mode": emissions_row["tracking_mode"]
            }

# Save combined results to disk
data.to_csv(path.join(data_path, "results_final.csv"), index=False)
