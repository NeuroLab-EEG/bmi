"""
Generate summary plot for all algorithms.

References
----------
.. [1] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_statistical_analysis.html
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)
from moabb.analysis.plotting import summary_plot

# Define metric for analysis
METRIC = "acc"

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "final_results.csv"))
results["score"] = results[METRIC]

# Compute statistics
stats = compute_dataset_statistics(results)
P, T = find_significant_differences(stats)

# Generate summary plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = summary_plot(P, T)
plt.savefig("summary")
