"""
Example of permutation test.

References
----------
.. [1] https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_customization.html
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product


class Permutation:
    def __init__(self):
        self.figure_title_fontsize = 18
        self.subplot_title_fontsize = 16
        self.subplot_fontsize = 14

        fig = plt.figure(figsize=(14, 7))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle("Permutation Test Experiment", fontsize=self.figure_title_fontsize)

        gs = GridSpec(4, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1, 2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[:, 1])

        self.subplot_original_data(ax1)
        self.subplot_hypothesis(ax2)
        self.subplot_test_statistic(ax3)
        self.subplot_sign_permutation(ax4)
        self.subplot_significance(ax5)

        fig.savefig("permutation")

    def _build_suplot(self, ax, title, text):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=self.subplot_title_fontsize)
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=self.subplot_fontsize,
            transform=ax.transAxes,
            multialignment="center",
        )

    def subplot_original_data(self, ax):
        self._build_suplot(ax, "Original Data", "1, 2, 3.3, 2.4, 5")

    def subplot_hypothesis(self, ax):
        self._build_suplot(
            ax,
            "Hypothesis",
            "$H \\colon \\theta=\\theta_0 \\qquad K \\colon \\theta < \\theta_0$",
        )

    def subplot_test_statistic(self, ax):
        self._build_suplot(ax, "Test Statistic", "$\\theta_0 = 1+2+3.3+2.4+5 = 13.7$")

    def subplot_sign_permutation(self, ax):
        self._build_suplot(
            ax,
            "Sign Permutation",
            "$+1,+2,+3.3,+2.4,+5$\n$-1,+2,+3.3,+2.4,+5$\n$\\cdots$\n$-1,-2,-3.3,-2.4,-5$",
        )

    def subplot_significance(self, ax):
        # Generate permutation test distribution
        data = [1, 2, 3.3, 2.4, 5]
        signs = product([-1, 1], repeat=len(data))
        sums = [sum(d * s for d, s in zip(data, sign_combo)) for sign_combo in signs]

        # Plot distribution
        ax.hist(sums, bins=10, color="skyblue", edgecolor="black")
        ax.set_title("Permutation Distribution", fontsize=self.subplot_title_fontsize)
        ax.tick_params(axis="both", labelsize=self.subplot_fontsize)

        # Plot hypotheses
        ax.axvline(x=0, color="red", linestyle="-", linewidth=4, label="$\\theta$")
        ax.axvline(x=13.7, color="green", linestyle="-", linewidth=4, label="$\\theta_0$")
        ax.legend(fontsize=self.subplot_fontsize)


Permutation()
