"""
Plot carbon emission from training models
References:
    - https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/example_codecarbon.html  # noqa: E501
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.plotting import codecarbon_plot


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results from disk
results = pd.read_csv(path.join(data_path, "final_results.csv"))

# Plot carbon emission
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = codecarbon_plot(results)
plt.savefig("carbon")
