"""
References:
    - https://geomstats.github.io/notebooks/04_practical_methods__from_vector_spaces_to_manifolds.html
    - https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Arrow.html#matplotlib-patches-arrow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere


class Geodesic:
    def __init__(self):
        gs.random.seed(1)

        # Create figure and subplots
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        gridspec = fig.add_gridspec(1, 2, wspace=0.1)
        ax1 = fig.add_subplot(gridspec[0, 0])
        ax2 = fig.add_subplot(gridspec[0, 1], projection="3d")

        # Generate subplots
        self.subplot_tanget_space(ax1)
        self.subplot_manifold(ax2)

        plt.savefig("geodesic", bbox_inches="tight")

    def subplot_manifold(self, ax):
        # Define points
        point_a = np.array([0.0, 0.0, 1.0])
        point_b = np.array([0.0, -1.0, 0.0])

        # Compute geodesic
        sphere = Hypersphere(dim=2)
        tangent_vec = sphere.metric.log(point_b, point_a)
        geodesic = sphere.metric.geodesic(initial_point=point_a, initial_tangent_vec=tangent_vec)
        points_on_geodesic = geodesic(gs.linspace(0.0, 1.0, 10))

        # Plot points and geodesic
        ax = visualization.plot(point_a, ax=ax, space="S2", s=100, label="Point A")
        ax = visualization.plot(point_b, ax=ax, space="S2", s=100, label="Point B")
        ax = visualization.plot(points_on_geodesic, ax=ax, space="S2", color="black", label="Geodesic")

        # Plot tangent vector
        arrow = visualization.Arrow3D(point_a, vector=tangent_vec)
        arrow.draw(ax, color="black", label="Tangent Vector")

        # Label plot
        ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
        ax.set_title("Manifold")

    def subplot_tanget_space(self, ax):
        # Plot latitude circles
        radii = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        theta = np.linspace(0, 2 * np.pi, 100)
        for r in radii:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, color="darkgray", linewidth=0.5, zorder=0)

        # Plot longitude lines
        angles = np.deg2rad(np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]))
        t = np.linspace(-1, 1, 100)
        for angle in angles:
            x = t * np.cos(angle)
            y = t * np.sin(angle)
            ax.plot(x, y, color="darkgray", linewidth=0.5, zorder=0)

        # Plot starting point
        ax.plot(0, 0, "o", color="tab:blue", markersize=15, zorder=1, label="Point A")

        # Plot tangent vector
        arrowstyle = patches.ArrowStyle("->", head_length=20, head_width=8)
        arrow = patches.FancyArrowPatch(
            posA=(0, 0),
            posB=(-0.8, -0.6),
            arrowstyle=arrowstyle,
            linewidth=2,
            color="black",
            zorder=2,
            label="Tangent Vector",
        )
        ax.add_patch(arrow)

        # Configure plot
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y", rotation=0)
        ax.set_title("Tangent Space")

Geodesic()
