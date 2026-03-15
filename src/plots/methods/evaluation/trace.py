"""
Visualize traces from posterior MCMC sampling.

References
----------
.. [1] https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html
.. [2] https://python.arviz.org/en/stable/api/generated/arviz.plot_forest.html
.. [3] https://python.arviz.org/en/stable/api/generated/arviz.plot_rank.html
.. [4] https://python.arviz.org/en/stable/api/generated/arviz.plot_ess.html
.. [5] https://python.arviz.org/en/stable/api/generated/arviz.plot_energy.html
.. [6] https://python.arviz.org/en/stable/api/generated/arviz.summary.html
.. [7] https://python.arviz.org/en/stable/api/generated/arviz.rhat.html
.. [8] https://python.arviz.org/en/stable/api/generated/arviz.ess.html
"""

import matplotlib.pyplot as plt
import arviz as az
from os import path, getenv, listdir
from dotenv import load_dotenv
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
    BNCI2014_004,
    Dreyer2023,
    Weibo2014,
    GrosseWentrup2009,
    Stieger2021,
)
from src.pipelines import CSPBLDA, CSPGP, TSBLR, TSGP, BSCNN, BDCNN


class Trace:
    def __init__(self):
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        az.style.use("arviz-doc")

    def run(self):
        for DatasetCls in self._datasets():
            for PipelineCls, var_names in self._pipelines():
                dirname = path.join(self.data_path, "metrics", DatasetCls.__name__, PipelineCls.__name__)
                filenames = [f for f in listdir(dirname) if f.endswith(".nc")]
                for filename in filenames:
                    pathname = path.join(dirname, filename)
                    idata = az.from_netcdf(pathname)
                    for plot, prefix in self._plots():
                        plot(idata, var_names=var_names, classname=PipelineCls.__name__)
                        base_name, _ = path.splitext(filename)
                        plt.savefig(f"{prefix}-{DatasetCls.__name__}-{PipelineCls.__name__}-{base_name}")

    def _datasets(self):
        yield PhysionetMI
        yield Lee2019_MI
        yield Cho2017
        yield Schirrmeister2017
        yield Shin2017A
        yield BNCI2014_001
        yield BNCI2014_004
        yield Dreyer2023
        yield Weibo2014
        yield GrosseWentrup2009
        yield Stieger2021

    def _pipelines(self):
        yield (CSPBLDA, ["pi", "mu_0", "mu_1", "sigma"])
        yield (CSPGP, ["ell", "eta", "f"])
        yield (TSBLR, ["w", "b"])
        yield (TSGP, ["eta", "f"])
        yield (BSCNN, ["w", "b"])
        yield (BDCNN, ["w", "b"])

    def _plots(self):
        yield (self._plot_trace, "trace")
        yield (self._tabulate_summary, "summary")
        yield (self._plot_forest, "forest")
        yield (self._plot_rank, "rank")
        yield (self._plot_ess, "ess")
        yield (self._plot_energy, "energy")

    def _plot_trace(self, idata, **kwargs):
        az.plot_trace(idata, var_names=kwargs["var_names"])
        plt.suptitle(f"{kwargs['classname']} Trace Plot", fontsize=12, weight="bold")
        plt.tight_layout()

    def _plot_forest(self, idata, **kwargs):
        az.plot_forest(idata, var_names=kwargs["var_names"], combined=True, r_hat=True, ess=True)
        plt.suptitle(f"{kwargs['classname']} Forest Plot", fontsize=12, weight="bold")
        plt.tight_layout()

    def _plot_rank(self, idata, **kwargs):
        az.plot_rank(idata, var_names=kwargs["var_names"])
        plt.suptitle(f"{kwargs['classname']} Rank Plot", fontsize=12, weight="bold")
        plt.tight_layout()

    def _plot_ess(self, idata, **kwargs):
        az.plot_ess(idata, var_names=kwargs["var_names"])
        plt.suptitle(f"{kwargs['classname']} ESS Plot", fontsize=12, weight="bold")
        plt.tight_layout()

    def _plot_energy(self, idata, **kwargs):
        az.plot_energy(idata)
        plt.suptitle(f"{kwargs['classname']} Energy Plot", fontsize=12, weight="bold")
        plt.tight_layout()

    def _tabulate_summary(self, idata, **kwargs):
        # Generate convergence summary statistics
        summary = az.summary(idata, var_names=kwargs["var_names"])
        stats = [
            ["Mean R-hat", f"{summary['r_hat'].mean():.4f}"],
            ["Max R-hat", f"{summary['r_hat'].max():.4f}"],
            ["Min ESS (bulk)", f"{summary['ess_bulk'].min():.0f}"],
            ["Mean ESS (bulk)", f"{summary['ess_bulk'].mean():.0f}"],
            ["Min ESS (tail)", f"{summary['ess_tail'].min():.0f}"],
            ["Mean ESS (tail)", f"{summary['ess_tail'].mean():.0f}"],
        ]

        # Create summary statistics table
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=stats,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            colWidths=[0.6, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style table header
        for i in range(2):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Style table rows
        for i in range(1, len(stats) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        # Style figure
        plt.title(
            f"{kwargs['classname']} Convergence Diagnostics Summary",
            fontsize=12,
            weight="bold",
            pad=20,
        )
        plt.tight_layout()
