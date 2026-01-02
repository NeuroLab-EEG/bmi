"""
Plot data preprocessing pipeline for tangent space classification
References:
    - https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
    - https://matplotlib.org/stable/gallery/mplot3d/bars3d.html
    - https://matplotlib.org/stable/gallery/mplot3d/surface3d_2.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from os import getenv
from dotenv import load_dotenv
from pyriemann.estimation import Covariances
from moabb.utils import set_download_dir
from moabb.datasets import BNCI2014_001
from src.paradigm.paradigm import LogLossLeftRightImagery


class Pipeline:
    def __init__(self):
        self.rng = np.random.default_rng(1)

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), subplot_kw={"projection": "3d"})
        fig.subplots_adjust(left=0.05, wspace=0.5)
        
        self.subplot_raw_signals(axs[0])
        self.subplot_covariance_matrices(axs[1])
        self.subplot_spd_manifold(axs[2])

        plt.tight_layout()
        plt.savefig("pipeline")

    def subplot_raw_signals(self, ax):
        # Load environment variables
        load_dotenv()
        data_path = getenv("DATA_PATH")

        # Change download directory
        set_download_dir(data_path)

        # Initialize data
        dataset = BNCI2014_001()
        paradigm = LogLossLeftRightImagery()
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])

        # Configure subplot
        nrows = 4
        nsamples = 200
        signals_per_row = 3

        # Configure subplot
        colors = [
            ("#E63946", "#F1FAEE", "#1D3557"),
            ("#A8DADC", "#457B9D", "#F4A261"),
            ("#FFBE0B", "#8338EC", "#FB5607"),
            ("#06D6A0", "#118AB2", "#073B4C"),
        ]
        zorder = [4, 3, 2, 1]

        # Generate subplot
        for y_idx in range(nrows):
            for s_idx in range(signals_per_row):
                sig = X[y_idx][s_idx][0:nsamples].T
                x = np.arange(nsamples)
                y = np.ones_like(x) * y_idx
                z = sig
                ax.plot3D(x, y, z, color=colors[y_idx][s_idx], zorder=zorder[y_idx])

        # Label plot
        ax.set_yticks(range(nrows))
        ax.set_yticklabels([str(i) for i in range(nrows)])
        ax.set_xlabel("Time")
        ax.set_ylabel("Trials")
        ax.set_zlabel("Amplitude")
        ax.set_title("Raw Signals")

    def subplot_covariance_matrices(self, ax):
        # Configure subplot
        nrows = 4
        nsamples = 200
        nchannels = 6

        # Generate covariance matrices
        covariances = Covariances().fit_transform(
            self.rng.standard_normal((nrows, nchannels, nsamples))
        )
        
        # Normalize plot color scale
        vmin = min(cov.min() for cov in covariances)
        vmax = max(cov.max() for cov in covariances)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot each covariance matrix
        for y_idx, cov in enumerate(covariances):
            cov = np.rot90(cov)
            edges = np.arange(nchannels + 1)
            X, Z = np.meshgrid(edges, edges, indexing="ij")
            Y = np.full_like(X, y_idx)
            ax.plot_surface(
                X, Y, Z,
                facecolors=plt.cm.viridis(norm(cov)),
                rstride=1,
                cstride=1,
                shade=False
            )

        # Label plot
        ax.set_xlabel("Channels")
        ax.set_ylabel("Trials")
        ax.set_zlabel("Channels")
        ax.set_title("Sample Covariance Matrices")

    def subplot_spd_manifold(self, ax):        
        # Plot 3D surface
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        x, y = np.meshgrid(x, y)
        z = x**2 + y**2
        ax.plot_surface(x, y, z, color="tab:blue", edgecolor="none", alpha=0.5)

        # Plot first cluster
        n_points = 50
        x_red = self.rng.normal(loc=-4, scale=1.0, size=n_points)
        y_red = self.rng.normal(loc=-1, scale=1.0, size=n_points)
        z_red = x_red**2 + y_red**2
        ax.scatter(x_red, y_red, z_red, color="red", s=30)

        # Plot second cluster
        n_points = 50
        x_black = self.rng.normal(loc=1, scale=1.0, size=n_points)
        y_black = self.rng.normal(loc=4, scale=1.0, size=n_points)
        z_black = x_black**2 + y_black**2
        ax.scatter(x_black, y_black, z_black, color="black", s=30)

        # Plot mean point
        mean_red = np.array([x_red.mean(), y_red.mean(), z_red.mean()])
        mean_black = np.array([x_black.mean(), y_black.mean(), z_black.mean()])
        overall_mean = (mean_red + mean_black) / 2
        ax.scatter(overall_mean[0], overall_mean[1], overall_mean[2], color="gold", s=500, marker="*")

        # Label plot
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_title("SPD Manifold")

Pipeline()
