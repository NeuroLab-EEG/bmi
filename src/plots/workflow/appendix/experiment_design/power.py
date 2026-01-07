"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class Power():
    def __init__(self):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(11, 5))
        fig.subplots_adjust(wspace=0.4)

        self.mu = [0, 25, 25]
        self.sigma = [12, 12, 8]
        self.alpha = 0.05
        self.x_crit = norm.ppf(1 - self.alpha, loc=self.mu[0], scale=self.sigma[0])
        self.color = ["tab:blue", "tab:orange", "tab:green"]
        self.label = ["$H_0$", "$H_1$", "$H_2$"]

        self.subplot_errors(axs[0])
        self.subplot_experiment(axs[1])
        self.subplot_power(axs[2])

        fig.savefig("power")

    def subplot_errors(self, ax):
        # Plot normal curves
        for mu, sigma, color, label in zip(self.mu[:-1], self.sigma[:-1], self.color[:-1], self.label[:-1]):
            self._subplot_pdf(ax, mu, sigma, color, label)

    def subplot_experiment(self, ax):
        # Plot normal curves
        for mu, sigma, color, label in zip(self.mu, self.sigma, self.color, self.label):
            self._subplot_pdf(ax, mu, sigma, color, label)

        # Plot critical value
        ax.axvline(self.x_crit, linestyle="--", color="black")
        
        ax.set_title("Right-Tailed Experiment", fontsize=16)
        ax.legend(loc="upper left", fontsize=12)

    def _subplot_pdf(self, ax, mu, sigma, color, label):
        # Plot PDF
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
        y = norm.pdf(x, loc=mu, scale=sigma)
        ax.plot(x, y, color=color, label=label)

        if mu == 0:
            # Shade Type I error

        else:
            # Shade Type II error
            x_fill_beta = np.linspace(x[0], self.x_crit, 200)
            y_fill_beta = norm.pdf(x_fill_beta, loc=mu, scale=sigma)
            ax.fill_between(x_fill_beta, 0, y_fill_beta, color=color, alpha=0.3, hatch="\\\\\\")
            beta = norm.cdf(self.x_crit, loc=mu, scale=sigma)
            ax.text(mu, max(y)*0.7, f"$\\beta={beta:.2f}$", color=color, fontsize=12)

            # Shade power
            x_fill_power = np.linspace(self.x_crit, x[-1], 200)
            y_fill_power = norm.pdf(x_fill_power, loc=mu, scale=sigma)
            ax.fill_between(x_fill_power, 0, y_fill_power, color=color, alpha=0.3, hatch="xxx")
            power = 1 - beta
            ax.text(mu, max(y)*0.5, f"Power={power:.2f}", color=color, fontsize=12)

    def _shade_type_i_error(self, ):
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
        x_fill = np.linspace(self.x_crit, x[-1], 200)
        y_fill = norm.pdf(x_fill, loc=mu, scale=sigma)
        ax.fill_between(x_fill, 0, y_fill, color=color, alpha=0.3, hatch="///")
        alpha = 1 - norm.cdf(self.x_crit, loc=mu, scale=sigma)
        ax.text(mu, max(y)*0.7, f"$\\alpha={alpha:.2f}$", color=color, fontsize=12)

    def subplot_power(self, ax):
        # Plot power curves
        x_crit = norm.ppf(1 - 0.05, loc=0, scale=12)
        self._subplot_cdf(ax, 25, 12, x_crit, "tab:orange", "$\\phi_1$")
        self._subplot_cdf(ax, 25, 8, x_crit, "tab:green", "$\\phi_2$")
        
        ax.set_title("Power Curves", fontsize=16)
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\beta$", rotation=0)
        ax.legend()

    def _subplot_cdf(self, ax, mu, sigma, x_crit, color, label):
        mu_vals = np.linspace(0, 50, 200)
        power = 1 - norm.cdf(x_crit, loc=mu_vals, scale=sigma)
        ax.plot(mu_vals, power, color=color, label=label)

Power()
