"""
Explore accuracy/AUROC for two models over all datasets.

References
----------
.. [1] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_statistical_analysis.html
.. [2] https://moabb.neurotechx.com/docs/generated/moabb.analysis.plotting.paired_plot.html#moabb.analysis.plotting.paired_plot
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.plotting import paired_plot

# Define metric and algorithms to compare
METRIC = "acc"
ALGO1 = "CSP+SVM"
ALGO2 = "TS+SVM"

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "final_results.csv"))
results["score"] = results[METRIC]

# Generate paired plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = paired_plot(results, ALGO1, ALGO2)
plt.savefig("paired")
