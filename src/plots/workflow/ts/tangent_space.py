"""
Tangent space visualization
"""

import numpy as np
import matplotlib.pyplot as plt


class TangentSpace:
    def __init__(self):
        self.rng = np.random.default_rng(1)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
        fig.subplots_adjust(wspace=0.5)

        self.subplot_logarithmic_map(axs[0])
        self._generate_tangent_space_points()
        self.subplot_tangent_space(axs[1])
        self.subplot_decision_boundary(axs[2])

        plt.tight_layout()
        plt.savefig("tangent_space")

    def _generate_tangent_space_points(self):
        # Generate first cluster of points
        n_points = 10
        self.x_red = self.rng.normal(loc=3, scale=2.0, size=n_points)
        self.y_red = self.rng.normal(loc=2, scale=2.0, size=n_points)

        # Generate second cluster of points
        n_points = 10
        self.x_black = self.rng.normal(loc=-2, scale=2.0, size=n_points)
        self.y_black = self.rng.normal(loc=-3, scale=2.0, size=n_points)

    def subplot_logarithmic_map(self, ax):
        # Plot first cluster of points
        n_points = 10
        x_red = self.rng.normal(loc=-4, scale=1.0, size=n_points)
        y_red = self.rng.normal(loc=-1, scale=1.0, size=n_points)
        ax.scatter(x_red, y_red, color="red", s=30)

        # Plot second cluster of points
        n_points = 10
        x_black = self.rng.normal(loc=1, scale=1.0, size=n_points)
        y_black = self.rng.normal(loc=4, scale=1.0, size=n_points)
        ax.scatter(x_black, y_black, color="black", s=30)

        # Plot overall mean point
        mean_red = np.array([x_red.mean(), y_red.mean()])
        mean_black = np.array([x_black.mean(), y_black.mean()])
        overall_mean = (mean_red + mean_black) / 2
        ax.scatter(overall_mean[0], overall_mean[1], color="gold", s=500, marker="*")

        # Plot vectors from cluster points to mean point
        all_points = np.vstack((np.column_stack((x_red, y_red)), np.column_stack((x_black, y_black))))
        for point in all_points:
            ax.plot([point[0], overall_mean[0]],
                     [point[1], overall_mean[1]],
                     color="tab:green", alpha=0.7)

        # Label plot
        ax.set_facecolor("lightgray")
        ax.set_xlabel("U1")
        ax.set_ylabel("U2")
        ax.set_title("Logarithmic Maps")

    def subplot_tangent_space(self, ax):
        # Plot clusters of points
        ax.scatter(self.x_red, self.y_red, color="red", s=30)
        ax.scatter(self.x_black, self.y_black, color="black", s=30)

        # Label plot
        ax.set_facecolor("lightgray")
        ax.set_xlabel("V1")
        ax.set_ylabel("V2")
        ax.set_title("Tangent Space")

    def subplot_decision_boundary(self, ax):
        # Plot tangent space points
        self.subplot_tangent_space(ax)

        # Plot decision boundary
        x = np.linspace(-6, 6, 100)
        L, k, x0 = 6, 1.0, 0
        y_sigmoid = L / (1 + np.exp(k*(x - x0))) - L/2
        ax.plot(x, y_sigmoid, color="blue", linewidth=2, linestyle="--")

        # Label plot
        ax.set_title("Decision Boundary")

TangentSpace()
