"""
Plot an EEG dataset channel montage.

References
----------
.. [1] https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
.. [2] https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
"""

import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
from mne_bids import find_matching_paths, read_raw_bids

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Define EEG datasets
datasets = [
    ("PhysionetMI", "MNE-BIDS-physionet-motor-imagery"),
    ("Lee2019_MI", "MNE-BIDS-lee2019-mi"),
    ("Cho2017", "MNE-BIDS-cho2017"),
    ("Schirrmeister2017", "MNE-BIDS-schirrmeister2017"),
    ("Shin2017A", "MNE-BIDS-shin2017-a"),
    ("BNCI2014_001", "MNE-BIDS-bnci2014-001"),
]

# Plot channel montages
for name, subdir in datasets:
    # Define directory
    root = path.join(path.expanduser(data_path), subdir)
    bids_paths = find_matching_paths(root=root, subjects="1", datatypes="eeg", extensions=".edf")
    bids_path = bids_paths[0]

    # Read directory
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.set_montage("standard_1005")

    # Plot montage
    fig = raw.plot_sensors(show_names=True, sphere="auto", show=False)
    plt.tight_layout()
    plt.savefig(name)
