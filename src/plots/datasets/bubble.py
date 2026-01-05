"""
Visualize MOABB bubble plots.

References
----------
.. [1] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_dataset_bubbles.html  # noqa: E501
"""

import matplotlib.pyplot as plt
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)
from moabb.datasets.utils import plot_datasets_grid

# Visualize datasets with bubble plots
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
fig = plot_datasets_grid(
    n_col=2,
    datasets=[
        PhysionetMI(),
        Shin2017A(),
        Schirrmeister2017(),
        Cho2017(),
        Lee2019_MI(),
        BNCI2014_001(),
    ],
)
fig.suptitle("Subjects & Sessions Bubble Plot")
fig.tight_layout()
plt.savefig("bubble")
