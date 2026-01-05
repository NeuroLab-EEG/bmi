"""
Visualization of cross-validation.

References
----------
.. [1] https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class CrossValidation:
    def __init__(self):
        # Create plot
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

        self.outer_cv(axs[0])
        self.inner_cv(axs[1])

        fig.tight_layout()
        fig.savefig("cv")

    def _cv(self, ax, folds, color, title):
        # Prepare data for plot
        labels = list(folds.keys())
        data = np.array(list(folds.values()))
        data_cum = data.cumsum(axis=1)
        active = mcolors.to_rgba(color, 1.0)
        inactive = mcolors.to_rgba(color, 0.5)
        colors = [
            (inactive, active, active, active, active),
            (active, inactive, active, active, active),
            (active, active, inactive, active, active),
            (active, active, active, inactive, active),
            (active, active, active, active, inactive),
        ]

        # Configure plot
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
        ax.set_title(title)

    def outer_cv(self, ax):
        # Configure data for plot
        folds = {
            "Fold 1": [12, 12, 12, 12, 12],
            "Fold 2": [12, 12, 12, 12, 12],
            "Fold 3": [12, 12, 12, 12, 12],
            "Fold 4": [12, 12, 12, 12, 12],
            "Fold 5": [12, 12, 12, 12, 12],
        }
        color = "tab:blue"
        title = "Outer 5-Fold CV"
        self._cv(ax, folds, color, title)

    def inner_cv(self, ax):
        # Configure data for plot
        folds = {
            "Fold 1": [16, 16, 16],
            "Fold 2": [16, 16, 16],
            "Fold 3": [16, 16, 16],
        }
        color = "tab:orange"
        title = "Inner 3-Fold CV"
        self._cv(ax, folds, color, title)


CrossValidation()
