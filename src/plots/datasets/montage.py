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
    Beetl2021_A,
    Beetl2021_B,
    Dreyer2023,
    Stieger2021,
    Weibo2014,
)


class Montage:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.data_path = getenv("DATA_PATH")

        self.fig, self.axs = plt.subplots(4, 3, figsize=(32, 28))

    def __call__(self):
        for row, col, DatasetCls, subdir in self._params():
            # Define directory
            root = path.join(path.expanduser(self.data_path), subdir)
            subject = 4 if DatasetCls is Beetl2021_B else 1
            bids_paths = find_matching_paths(
                root=root, subjects=f"{subject}", datatypes="eeg", extensions=".edf"
            )
            # print(row, col, DatasetCls.__name__)
            bids_path = bids_paths[0]

            # Read directory
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            raw.set_montage("standard_1005")

            # Plot montage
            ax = self.axs[row][col]
            try:
                raw.plot_sensors(show_names=True, sphere="auto", show=False, axes=ax)
            except ValueError:
                raw.plot_sensors(show_names=True, sphere=(0, 0, 0, 0.095), show=False, axes=ax)
            ax.set_title(DatasetCls.__name__, fontsize=32)

        self.fig.tight_layout()
        self.fig.suptitle("Channels Montages", fontsize=36)
        self.fig.savefig("montage")

    def _params(self):
        yield (0, 0, Beetl2021_A, "MNE-BIDS-beetl2021-a")
        yield (0, 1, Beetl2021_B, "MNE-BIDS-beetl2021-b")
        yield (0, 2, BNCI2014_001, "MNE-BIDS-bnci2014-001")
        yield (1, 0, BNCI2014_004, "MNE-BIDS-bnci2014-004")
        yield (1, 1, Cho2017, "MNE-BIDS-cho2017")
        yield (1, 2, Dreyer2023, "MNE-BIDS-dreyer2023")
        yield (2, 0, Lee2019_MI, "MNE-BIDS-lee2019-mi")
        yield (2, 1, PhysionetMI, "MNE-BIDS-physionet-motor-imagery")
        yield (2, 2, Schirrmeister2017, "MNE-BIDS-schirrmeister2017")
        yield (3, 0, Shin2017A, "MNE-BIDS-shin2017-a")
        yield (3, 1, Stieger2021, "MNE-BIDS-stieger2021")
        yield (3, 2, Weibo2014, "MNE-BIDS-weibo2014")


Montage()()
