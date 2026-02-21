"""
Visualize MOABB bubble plots.

References
----------
.. [1] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_dataset_bubbles.html
"""

import matplotlib.pyplot as plt
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
    Liu2024,
)
from moabb.datasets.utils import plot_datasets_cluster


class BubblePlot:
    def run(self):
        fig = plot_datasets_cluster(
            meta_gap=15.0,
            datasets=[
                PhysionetMI(),
                Liu2024(),
                Schirrmeister2017(),
                Lee2019_MI(),
                BNCI2014_001(),
                BNCI2014_004(),
                Dreyer2023(),
                Cho2017(),
                GrosseWentrup2009(),
                Weibo2014(),
                Stieger2021(),
                Shin2017A(),
            ],
            color_map={"imagery": "tab:blue"},
        )

        fig.suptitle("MI-EEG Datasets Bubble Plot", fontweight="bold", fontsize=10)
        fig.set_size_inches(4, 3)
        plt.savefig("bubble-plot")
