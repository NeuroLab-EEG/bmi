"""
Generate grouped and stacked bar charts showing class balances per fold
References:
    - https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    - https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
"""

import matplotlib.pyplot as plt
import numpy as np
from os import getenv
from collections import Counter
from dotenv import load_dotenv
from src.paradigm.paradigm import LogLossLeftRightImagery
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)
from moabb.utils import set_download_dir
from moabb.evaluations import CrossSubjectSplitter


def generate_plot(name, train_left, train_right, test_left, test_right):
    # Initialize data structures
    folds = ("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")
    results = {
        "train": {
            "left": np.array(train_left),
            "right": np.array(train_right),
        },
        "test": {
            "left": np.array(test_left),
            "right": np.array(test_right),
        },
    }

    # Initialize bar plot
    n_folds = len(folds)
    x = np.arange(n_folds)
    width = 0.35
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked train bars
    train_left = ax.bar(
        x - width / 2,
        results["train"]["left"],
        width,
        label="Train Left",
        color="tab:blue",
    )
    train_right = ax.bar(
        x - width / 2,
        results["train"]["right"],
        width,
        bottom=results["train"]["left"],
        label="Train Right",
        color="tab:orange",
    )

    # Create stacked test bars
    test_left = ax.bar(
        x + width / 2,
        results["test"]["left"],
        width,
        label="Test Left",
        color="tab:green",
    )
    test_right = ax.bar(
        x + width / 2,
        results["test"]["right"],
        width,
        bottom=results["test"]["left"],
        label="Test Right",
        color="tab:red",
    )

    # Generate grouped and stacked bar plot
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Trials")
    ax.legend(ncols=2)
    ax.grid(False)
    plt.savefig(name)


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

# Define plot generation parameters
params = [
    ("PhysionetMI", LogLossLeftRightImagery(resample=160), PhysionetMI()),
    ("Lee2019_MI", LogLossLeftRightImagery(resample=1000), Lee2019_MI()),
    ("Cho2017", LogLossLeftRightImagery(resample=512), Cho2017()),
    ("Schirrmeister2017", LogLossLeftRightImagery(resample=500), Schirrmeister2017()),
    ("Shin2017A", LogLossLeftRightImagery(resample=200), Shin2017A()),
    ("BNCI2014_001", LogLossLeftRightImagery(resample=250), BNCI2014_001()),
]

# Generate grouped and stacked plots
for name, paradigm, dataset in params:
    # Initialize data structures
    train_left, train_right, test_left, test_right = [], [], [], []

    # Prepare dataset
    X, y, metadata = paradigm.get_data(dataset=dataset)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split dataset into same folds as evaluation
    cv = CrossSubjectSplitter(cv_class=GroupKFold, **dict(n_splits=5))

    # Count class balance per fold
    for cv_ind, (train, test) in enumerate(cv.split(y, metadata)):
        # Fill train counts
        train = Counter(y[train])
        train_left.append(train[0])
        train_right.append(train[1])

        # Fill test counts
        test = Counter(y[test])
        test_left.append(test[0])
        test_right.append(test[1])

    generate_plot(name, train_left, train_right, test_left, test_right)
