"""
Visualization of 5-fold cross-validation.

References
----------
.. [1] https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Create data splits from folds
folds = {
    "Fold 1": [10, 10, 10, 10, 10],
    "Fold 2": [10, 10, 10, 10, 10],
    "Fold 3": [10, 10, 10, 10, 10],
    "Fold 4": [10, 10, 10, 10, 10],
    "Fold 5": [10, 10, 10, 10, 10],
}

# Prepare data for plot
labels = list(folds.keys())
data = np.array(list(folds.values()))
data_cum = data.cumsum(axis=1)
active = mcolors.to_rgba("tab:blue")
inactive = mcolors.to_rgba("tab:cyan")
colors = [
    (inactive, active, active, active, active),
    (active, inactive, active, active, active),
    (active, active, inactive, active, active),
    (active, active, active, inactive, active),
    (active, active, active, active, inactive),
]

# Create plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.set_xlim(0, np.sum(data, axis=1).max())

# Plot data
for idx in range(data.shape[1]):
    widths = data[:, idx]
    starts = data_cum[:, idx] - widths
    ax.barh(
        labels,
        widths,
        left=starts,
        height=0.5,
        color=colors[idx],
    )

# Label plot
plt.title("5-Fold Cross-Validation Design")
fig.tight_layout()
fig.savefig("cv")
