"""
Plot ordinary differential equation (ODE) on the SPD manifold.

References
----------
.. [1] https://matplotlib.org/stable/gallery/lines_bars_and_markers/axline.html
.. [2] https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class ODE:
    def __init__(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        # Generate plot components
        self._plot_curve(ax)
        self._plot_points(ax)
        self._plot_geodesic(ax)
        self._plot_log_map(ax)
        self._plot_exp_map(ax)

        # Create custom formatted legend
        handles, labels = ax.get_legend_handles_labels()
        arrow_proxy = Line2D(
            [0], [0], color="black", markersize=8, linestyle="-", linewidth=2
        )
        handles.append(arrow_proxy)
        labels.append("Logarithmic Map")

        # Format figure display
        ax.legend(
            handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.05, 1)
        )
        ax.set_xlabel("X1", fontsize=14)
        ax.set_ylabel("X2", fontsize=14, rotation=0, labelpad=15)
        ax.set_title("Ordinary Differential Equation on the SPD Manifold", fontsize=14)
        fig.savefig("ode", bbox_inches="tight", dpi=300)

    def _plot_curve(self, ax):
        x = np.linspace(-3, 6, 100)
        y = (1 / 10) * x**2 + 1
        ax.plot(x, y, label="SPD Manifold")
        ax.set_xticks(np.arange(-6, 8, 2))
        ax.set_yticks(np.arange(0, 6, 0.5))
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    def _plot_points(self, ax):
        p1, p2 = np.array([0, (1 / 10) * 0**2 + 1]), np.array([5, (1 / 10) * 5**2 + 1])
        ax.plot(p1[0], p1[1], "o", markersize=8, color="tab:orange", label="Identity")
        ax.plot(p2[0], p2[1], "o", markersize=8, color="tab:green", label="Destination")

    def _plot_geodesic(self, ax):
        x = np.linspace(0, 5, 10)
        y = (1 / 10) * x**2 + 1
        ax.plot(x, y, linestyle=":", color="black", linewidth=3, label="Geodesic")

    def _plot_log_map(self, ax):
        ax.annotate(
            "",
            xy=(5, 1),
            xytext=(0, 1),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

    def _plot_exp_map(self, ax):
        y_start, y_end = 1, (1 / 10) * 5**2 + 1
        ax.plot(
            [5, 5],
            [y_start, y_end],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Exponential Map",
        )


ODE()
