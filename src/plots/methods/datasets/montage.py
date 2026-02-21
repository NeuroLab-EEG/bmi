"""
Plot an EEG dataset channel montage.

References
----------
.. [1] https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
.. [2] https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
.. [3] https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html
"""

import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from mne_bids import find_matching_paths, read_raw_bids
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
from src.datasets import Liu2024


class Montage:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.data_path = getenv("DATA_PATH")

    def run(self):
        fig, axes = plt.subplots(2, 6, figsize=(12, 4))

        for row, col, DatasetCls, subdir in self._params():
            # Read directory
            root = path.join(path.expanduser(self.data_path), subdir)
            bids_paths = find_matching_paths(root=root, subjects="1", datatypes="eeg", extensions=".edf")
            bids_path = bids_paths[0]
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            if DatasetCls not in [Liu2024, GrosseWentrup2009]:
                raw.set_montage("standard_1005")

            # Plot montage
            ax = axes[row][col]
            try:
                raw.plot_sensors(show_names=False, sphere="auto", show=False, axes=ax)
            except ValueError:
                raw.plot_sensors(show_names=False, sphere=(0, 0, 0, 0.095), show=False, axes=ax)
            ax.set_title(DatasetCls.__name__, fontsize=14)

        fig.suptitle("Channel Montages", fontweight="bold", fontsize=16)
        fig.tight_layout()
        fig.savefig("montages")

    def _params(self):
        yield (0, 0, BNCI2014_001, "MNE-BIDS-bnci2014-001")
        yield (0, 1, BNCI2014_004, "MNE-BIDS-bnci2014-004")
        yield (0, 2, Cho2017, "MNE-BIDS-cho2017")
        yield (0, 3, Dreyer2023, "MNE-BIDS-dreyer2023")
        yield (0, 4, Lee2019_MI, "MNE-BIDS-lee2019-mi")
        yield (0, 5, PhysionetMI, "MNE-BIDS-physionet-motor-imagery")
        yield (1, 0, Schirrmeister2017, "MNE-BIDS-schirrmeister2017")
        yield (1, 1, Shin2017A, "MNE-BIDS-shin2017-a")
        yield (1, 2, Stieger2021, "MNE-BIDS-stieger2021")
        yield (1, 3, Weibo2014, "MNE-BIDS-weibo2014")
        yield (1, 4, Liu2024, "MNE-BIDS-liu2024")
        yield (1, 5, GrosseWentrup2009, "MNE-BIDS-grosse-wentrup2009")
