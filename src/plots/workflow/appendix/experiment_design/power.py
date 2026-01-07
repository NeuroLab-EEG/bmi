"""
A right-tailed experiment & power curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit


class Power():
    def __init__(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        self.plot_experiment(axs[0])
        self.plot_power(axs[1])

        fig.tight_layout()
        fig.savefig("power")

    def plot_experiment(self, ax):
        # Initialize values
        mu = [0, 33]
        sigma = [12, 11]
        color = ["tab:pink", "tab:gray"]
        alpha = 0.05
        x_crit = norm.ppf(1 - alpha, loc=mu[0], scale=sigma[0])

        # Plot critical value
        ax.axvline(x_crit, linestyle="--", color="black")

        # Plot null hypothesis
        x = np.linspace(mu[0] - 4 * sigma[0], mu[0] + 4 * sigma[0], 200)
        y = norm.pdf(x, loc=mu[0], scale=sigma[0])
        ax.plot(x, y, color=color[0], label="$H_0$")

        # Plot alternative hypothesis
        x = np.linspace(mu[1] - 4 * sigma[1], mu[1] + 4 * sigma[1], 200)
        y = norm.pdf(x, loc=mu[1], scale=sigma[1])
        ax.plot(x, y, color=color[1], label="$H_1$")

        # Shade alpha
        x_fill = np.linspace(x_crit, x[-1], 200)
        y_fill = norm.pdf(x_fill, loc=mu[0], scale=sigma[0])
        ax.fill_between(x_fill, 0, y_fill, color=color[0], alpha=0.5, label="$\\alpha$")

        # Shade beta
        x_fill = np.linspace(x[0], x_crit, 200)
        y_fill = norm.pdf(x_fill, loc=mu[1], scale=sigma[1])
        ax.fill_between(x_fill, 0, y_fill, color=color[1], alpha=0.3, hatch="\\\\\\", label="$\\beta$")

        # Shade power
        x_fill = np.linspace(x_crit, x[-1], 200)
        y_fill = norm.pdf(x_fill, loc=mu[1], scale=sigma[1])
        ax.fill_between(x_fill, 0, y_fill, color=color[1], alpha=0.3, hatch="///", label="Power")

        # Label plot
        ax.set_title("Right-Tailed Experiment")
        ax.legend(loc="upper left")

    def plot_power(self, ax):
        x = np.linspace(0, 12, 200)
        sigma = [expit(x - 5), expit(x - 7)]

        ax.plot(x, sigma[0], color="tab:olive", label="$\\phi_1$")
        ax.plot(x, sigma[1], color="tab:cyan", label="$\\phi_2$")

        ax.set_title("Power Curves")
        ax.legend()

Power()
