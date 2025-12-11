"""
Generate tree plot for two algorithms
References:
    - https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_statistical_analysis.html
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.meta_analysis import compute_dataset_statistics
from moabb.analysis.plotting import meta_analysis_plot


# Define metric and algorithms to compare
METRIC = "acc"
ALGO1 = "dcnn"
ALGO2 = "csp_lda"

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "results.csv"))

# Compute statistics
results["score"] = results[METRIC]
stats = compute_dataset_statistics(results)

# Generate pairwise tree plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = meta_analysis_plot(stats, ALGO1, ALGO2)
plt.savefig("meta_analysis")
