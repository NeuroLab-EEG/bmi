"""
Explore accuracy/AUROC for all models over all datasets
References:
    - https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_statistical_analysis.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/generated/moabb.analysis.plotting.score_plot.html#moabb.analysis.plotting.score_plot  # noqa: E501
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.plotting import score_plot


# Define metric for exploration
METRIC = "acc"

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "final_results.csv"))
results["score"] = results[METRIC]

# Generate accuracy plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = score_plot(results)
plt.savefig("score")
