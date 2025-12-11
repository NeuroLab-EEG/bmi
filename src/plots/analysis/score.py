"""
Visualize accuracy and AUROC scores from 5-fold cross-validation
References:
    - https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_statistical_analysis.html
    - https://moabb.neurotechx.com/docs/generated/moabb.analysis.plotting.score_plot.html#moabb.analysis.plotting.score_plot
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.plotting import score_plot


def generate_plot(name, results):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    fig = score_plot(results)
    plt.savefig(name)


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "results.csv"))

# Generate accuracy plot
results["score"] = results["acc"]
generate_plot("acc", results)

# Generate AUROC plot
results["score"] = results["auroc"]
generate_plot("auroc", results)
