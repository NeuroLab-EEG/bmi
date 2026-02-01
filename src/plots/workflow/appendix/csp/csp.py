"""
Generate plots demonstrating CSP.

References
----------
.. [1] https://numpy.org/doc/stable/reference/random/index.html
.. [2] https://matplotlib.org/stable/gallery/units/ellipse_with_units.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse


class CSP:
    def __init__(self):
        # Define random seed
        self.rng = np.random.default_rng(1)

        # Define mean
        self.mean = np.zeros(2)

        # Define first sample covariance matrix
        self.eigvecs1 = np.array(
            [
                [np.cos(np.pi / 6), np.cos(2 * np.pi / 3)],
                [np.sin(np.pi / 6), np.sin(2 * np.pi / 3)],
            ]
        )
        self.eigvals1 = np.diag([9.0, 3.0])
        self.cov1 = self.eigvecs1 @ self.eigvals1 @ self.eigvecs1.T
        self.cov1_whitening = self.eigvecs1 @ np.diag(1.0 / np.sqrt(np.diag(self.eigvals1))) @ self.eigvecs1.T
        self.cov1_whitened = self.cov1_whitening @ self.cov1 @ self.cov1_whitening.T
        self.eigvals1_whitened = np.diag(np.linalg.eigvals(self.cov1_whitened))

        # Define second sample covariance matrix
        self.eigvecs2 = np.array(
            [
                [np.cos(np.pi / 3), np.cos(5 * np.pi / 6)],
                [np.sin(np.pi / 3), np.sin(5 * np.pi / 6)],
            ]
        )
        self.eigvals2 = np.diag([10.0, 2.5])
        self.cov2 = self.eigvecs2 @ self.eigvals2 @ self.eigvecs2.T
        self.cov2_whitening = self.eigvecs2 @ np.diag(1.0 / np.sqrt(np.diag(self.eigvals2))) @ self.eigvecs2.T
        self.cov2_whitened = self.cov2_whitening @ self.cov2 @ self.cov2_whitening.T
        self.eigvals2_whitened = np.diag(np.linalg.eigvals(self.cov2_whitened))

        # Define SCMS summed
        self.cov_summed = self.cov1 + self.cov2
        eigvals_summed, self.eigvecs_summed = np.linalg.eigh(self.cov_summed)
        self.eigvals_summed = np.diag(eigvals_summed)
        self.cov_summed_whitening = (
            self.eigvecs_summed @ np.diag(1.0 / np.sqrt(np.diag(self.eigvals_summed))) @ self.eigvecs_summed.T
        )
        self.cov_summed_whitened = self.cov_summed_whitening @ self.cov_summed @ self.cov_summed_whitening.T
        self.eigvals_summed_whitened = np.diag(np.linalg.eigvals(self.cov_summed_whitened))

        # Define SCMs transformed
        self.cov1_transformed = self.cov_summed_whitening @ self.cov1 @ self.cov_summed_whitening.T
        eigvals1_transformed, self.eigvecs1_transformed = np.linalg.eigh(self.cov1_transformed)
        self.eigvals1_transformed = np.diag(eigvals1_transformed)
        self.cov2_transformed = self.cov_summed_whitening @ self.cov2 @ self.cov_summed_whitening.T
        eigvals2_transformed, self.eigvecs2_transformed = np.linalg.eigh(self.cov2_transformed)
        self.eigvals2_transformed = np.diag(eigvals2_transformed)

    def __call__(self):
        self.plot_scms()
        self.plot_scms_summed()
        self.plot_scms_transformed()

    def _plot_scm(self, ax, eigvals, eigvecs, edgecolor):
        # Build SCM ellipse
        width, height = 2 * np.diag(eigvals)
        theta = np.degrees(np.arctan(eigvecs[1, 0] / eigvecs[0, 0]))
        ellipse = Ellipse(
            self.mean,
            width=width,
            height=height,
            angle=theta,
            edgecolor=edgecolor,
            facecolor=to_rgba(edgecolor, alpha=0.2),
            zorder=2,
        )

        # Plot SCM ellipse
        ax.add_patch(ellipse)
        ax.autoscale_view()
        ax.grid(True)
        self._center_axes(ax)

        # Plot eignenvectors with eigenvalues as magnitudes
        x0, y0 = self.mean
        u1, v1 = eigvecs[:, 0] * eigvals[0, 0]
        u2, v2 = eigvecs[:, 1] * eigvals[1, 1]
        plt.quiver(
            x0,
            y0,
            u1,
            v1,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.01,
            color=edgecolor,
            zorder=3,
        )
        plt.quiver(
            x0,
            y0,
            u2,
            v2,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.01,
            color=edgecolor,
            zorder=3,
        )

    def _center_axes(self, ax, padding=1.05):
        ax.set_aspect("equal", adjustable="box")
        cx, cy = self.mean
        lim = max(
            np.max(np.abs(np.array(ax.get_xlim()) - cx)),
            np.max(np.abs(np.array(ax.get_ylim()) - cy)),
        )
        lim *= padding
        ax.set_xlim(cx - lim, cx + lim)
        ax.set_ylim(cy - lim, cy + lim)
        ax.set_anchor("C")

    def plot_scms(self):
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(121, aspect="equal")
        self._plot_scm(ax1, self.eigvals1, self.eigvecs1, "red")
        ax1.set_title("First Averaged SCM")

        ax2 = fig.add_subplot(122, aspect="equal")
        self._plot_scm(ax2, self.eigvals2, self.eigvecs2, "purple")
        ax2.set_title("Second Averaged SCM")

        plt.savefig("scms")

    def plot_scms_summed(self):
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(121, aspect="equal")
        self._plot_scm(ax1, self.eigvals1, self.eigvecs1, "red")
        self._plot_scm(ax1, self.eigvals2, self.eigvecs2, "purple")
        ax1.set_title("Both SCMs")

        ax2 = fig.add_subplot(122, aspect="equal")
        self._plot_scm(ax2, self.eigvals_summed, self.eigvecs_summed, "black")
        ax2.set_title("Summed SCMs")

        plt.savefig("scms-summed")

    def plot_scms_transformed(self):
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(121, aspect="equal")
        self._plot_scm(ax1, self.eigvals_summed_whitened, self.eigvecs_summed, "black")
        ax1.set_title("SCMs Summed & Whitened")

        ax2 = fig.add_subplot(122, aspect="equal")
        self._plot_scm(ax2, self.eigvals1_transformed, self.eigvecs1_transformed, "red")
        self._plot_scm(ax2, self.eigvals2_transformed, self.eigvecs2_transformed, "purple")
        ax2.set_title("SCMs Transformed")

        plt.savefig("scms-transformed")


csp = CSP()()
