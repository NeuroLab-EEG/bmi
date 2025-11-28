"""
Description: Visualize MOABB bubble plots
Reference:
    - https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_dataset_bubbles.html
"""

import matplotlib.pyplot as plt
from moabb.datasets import (
    AlexMI,
    BNCI2014_001,
    PhysionetMI,
    Schirrmeister2017,
    Weibo2014,
    Zhou2016
)
from moabb.datasets.utils import plot_datasets_grid

fig = plot_datasets_grid(n_col=2, datasets=[
    Weibo2014(), BNCI2014_001(),
    PhysionetMI(), Zhou2016(),
    Schirrmeister2017(), AlexMI()])
fig.suptitle("Subjects & Sessions\nBubble Plot")
fig.tight_layout()
plt.savefig("bubble")
