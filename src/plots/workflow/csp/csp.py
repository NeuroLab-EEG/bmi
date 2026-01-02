"""
Generate plots demonstrating CSP
References:
    - https://numpy.org/doc/stable/reference/random/index.html
    - https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html  # noqa: E501
    - https://matplotlib.org/stable/gallery/units/ellipse_with_units.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse


class CSP:
    def __init__(self):
        # Assign random seed
        self.rng = np.random.default_rng(1)

        # Define mean
        self.mean = np.zeros(2)

        # Define first covariance matrix
        self.U = np.array(
            [
                [np.cos(np.pi / 6), np.cos(2 * np.pi / 3)],
                [np.sin(np.pi / 6), np.sin(2 * np.pi / 3)],
            ]
        )
        self.P = np.array([[9.0, 0.0], [0.0, 1.5]])
        self.C = self.U @ self.P @ self.U.T
        self.CM = self.U @ np.diag(1.0 / np.sqrt(np.diag(self.P))) @ self.U.T
        self.C_whitened = self.CM @ self.C @ self.CM.T
        self.P_whitened = np.diag(np.linalg.eigvals(self.C_whitened))

        # Define second covariance matrix
        self.V = np.array(
            [
                [np.cos(np.pi / 3), np.cos(5 * np.pi / 6)],
                [np.sin(np.pi / 3), np.sin(5 * np.pi / 6)],
            ]
        )
        self.Q = np.array([[10.0, 0.0], [0.0, 0.75]])
        self.D = self.V @ self.Q @ self.V.T
        self.DM = self.V @ np.diag(1.0 / np.sqrt(np.diag(self.Q))) @ self.V.T
        self.D_whitened = self.DM @ self.D @ self.DM.T
        self.Q_whitened = np.diag(np.linalg.eigvals(self.D_whitened))

        # Define covariances sum
        self.E = self.C + self.D
        R, W = np.linalg.eigh(self.E)
        self.R = np.diag(R)
        self.W = W
        self.EM = self.W @ np.diag(1.0 / np.sqrt(np.diag(self.R))) @ self.W.T
        self.E_whitened = self.EM @ self.E @ self.EM.T
        self.R_whitened = np.diag(np.linalg.eigvals(self.E_whitened))

        # Define covariances transformed
        self.C_transformed = self.EM @ self.C @ self.EM.T
        eigvals, eigvecs = np.linalg.eigh(self.C_transformed)
        self.P_transformed = np.diag(eigvals)
        self.U_transformed = eigvecs
        self.D_transformed = self.EM @ self.D @ self.EM.T
        eigvals, eigvecs = np.linalg.eigh(self.D_transformed)
        self.Q_transformed = np.diag(eigvals)
        self.V_transformed = eigvecs

    def _plot_covariance(
        self, ax, covariance, eigvals, eigvecs, color, edgecolor, alpha=1.0, points=False
    ):
        # Initialize constants
        A, D, Q = covariance, eigvals, eigvecs

        # Build ellipse
        n_std = 2.3
        width = 2 * n_std * np.sqrt(D[0, 0])
        height = 2 * n_std * np.sqrt(D[1, 1])
        theta = np.degrees(np.arctan(Q[1, 0] / Q[0, 0]))
        ellipse = Ellipse(
            self.mean,
            width=width,
            height=height,
            angle=theta,
            edgecolor=edgecolor,
            facecolor=to_rgba(edgecolor, alpha=0.2),
            zorder=1
        )

        # Plot data points and ellipse
        ax.add_patch(ellipse)
        if points:
            data = self.rng.multivariate_normal(self.mean, A, size=800)
            x, y = data.T
            plt.scatter(x, y, s=3, color=color, alpha=alpha, zorder=2)
        ax.autoscale_view()
        ax.grid(True)
        self._center_axes(ax)

        # Plot quivers
        x0, y0 = self.mean
        u1, v1 = Q[:, 0] * np.sqrt(D[0, 0]) * n_std
        u2, v2 = Q[:, 1] * np.sqrt(D[1, 1]) * n_std
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
            zorder=3
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
            zorder=3
        )

    def _center_axes(self, ax):
        ax.set_aspect("equal", adjustable="box")
        cx, cy = self.mean
        lim = max(
            np.max(np.abs(np.array(ax.get_xlim()) - cx)),
            np.max(np.abs(np.array(ax.get_ylim()) - cy)),
        )
        ax.set_xlim(cx - lim, cx + lim)
        ax.set_ylim(cy - lim, cy + lim)
        ax.set_anchor("C")

    def plot_first_covariance_ellipse(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(ax, self.C, self.P, self.U, "blue", "red")
        ax.set_title("First Covariance Matrix")
        plt.savefig("covariance_ellipse")

    def plot_first_covariance_whitened(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(
            ax, self.C_whitened, self.P_whitened, self.U, "blue", "red"
        )
        ax.set_title("First Covariance Matrix Whitened")
        plt.savefig("covariance_whitened")

    def plot_second_covariance_ellipse(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(ax, self.D, self.Q, self.V, "green", "purple")
        ax.set_title("Second Covariance Matrix")
        plt.savefig("covariance_ellipse_second")

    def plot_second_covariance_whitened(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(
            ax, self.D_whitened, self.Q_whitened, self.V, "green", "purple"
        )
        ax.set_title("Second Covariance Matrix Whitened")
        plt.savefig("covariance_whitened_second")

    def plot_both_covariances(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(ax, self.C, self.P, self.U, "blue", "red", alpha=0.5)
        self._plot_covariance(ax, self.D, self.Q, self.V, "green", "purple", alpha=0.5)
        ax.set_title("Both Covariance Matrices")
        plt.savefig("both_covariances")

    def plot_both_covariances_whitened(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(
            ax, self.C_whitened, self.P_whitened, self.U, "blue", "red", alpha=0.5
        )
        self._plot_covariance(
            ax, self.D_whitened, self.Q_whitened, self.V, "green", "purple", alpha=0.5
        )
        ax.set_title("Both Covariances Matrices Whitened")
        plt.savefig("both_covariances_whitened")

    def plot_covariances_sum(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(ax, self.E, self.R, self.W, "brown", "black")
        ax.set_title("Covariance Matrices Summed")
        plt.savefig("covariances_sum")

    def plot_covariances_sum_whitened(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(
            ax, self.E_whitened, self.R_whitened, self.W, "brown", "black"
        )
        ax.set_title("Covariance Matrices Summed & Whitened")
        plt.savefig("covariances_sum_whitened")

    def plot_covariances_transformed(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, aspect="equal")
        self._plot_covariance(
            ax,
            self.C_transformed,
            self.P_transformed,
            self.U_transformed,
            "blue",
            "red",
            alpha=0.5,
        )
        self._plot_covariance(
            ax,
            self.D_transformed,
            self.Q_transformed,
            self.V_transformed,
            "green",
            "purple",
            alpha=0.5,
        )
        ax.set_title("Covariance Matrices Transformed")
        plt.savefig("covariances_transformed")

    def plot_both_covariances_subplots(self):
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(121, aspect="equal")
        self._plot_covariance(ax1, self.C, self.P, self.U, "blue", "red")
        ax1.set_title("First Averaged SCM")

        ax2 = fig.add_subplot(122, aspect="equal")
        self._plot_covariance(ax2, self.D, self.Q, self.V, "green", "purple")
        ax2.set_title("Second Averaged SCM")

        plt.savefig("scms_subplots")

    def plot_covariances_sum_subplots(self):
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(121, aspect="equal")
        self._plot_covariance(ax1, self.C, self.P, self.U, "blue", "red", alpha=0.5)
        self._plot_covariance(ax1, self.D, self.Q, self.V, "green", "purple", alpha=0.5)
        ax1.set_title("Both SCMs")

        ax2 = fig.add_subplot(122, aspect="equal")
        self._plot_covariance(ax2, self.E, self.R, self.W, "brown", "black")
        ax2.set_title("Summed SCMs")

        plt.savefig("scms_sum_subplots")

    def plot_covariances_transformed_subplots(self):
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(121, aspect="equal")
        self._plot_covariance(
            ax1, self.E_whitened, self.R_whitened, self.W, "brown", "black"
        )
        ax1.set_title("SCMs Summed & Whitened")

        ax2 = fig.add_subplot(122, aspect="equal")
        self._plot_covariance(
            ax2,
            self.C_transformed,
            self.P_transformed,
            self.U_transformed,
            "blue",
            "red",
            alpha=0.5,
        )
        self._plot_covariance(
            ax2,
            self.D_transformed,
            self.Q_transformed,
            self.V_transformed,
            "green",
            "purple",
            alpha=0.5,
        )
        ax2.set_title("SCMs Transformed")

        plt.savefig("scms_transformed_subplots")


csp = CSP()
csp.plot_both_covariances_subplots()
csp.plot_covariances_sum_subplots()
csp.plot_covariances_transformed_subplots()
