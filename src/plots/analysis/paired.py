"""
Compare two algorithms over datasets
References:
    - https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_statistical_analysis.html
    - https://moabb.neurotechx.com/docs/generated/moabb.analysis.plotting.paired_plot.html#moabb.analysis.plotting.paired_plot
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.plotting import paired_plot

# Define algorithms to compare
ALGO1 = "csp_svm"
ALGO2 = "ts_svm"

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "results.csv"))

# Generate paired plot
results["score"] = results["acc"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = paired_plot(results, ALGO1, ALGO2)
plt.savefig("paired")
