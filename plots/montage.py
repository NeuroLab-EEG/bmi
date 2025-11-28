"""
Description: Plot an EEG dataset channel montage
References:
    - https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
    - https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
"""

from os import path, getenv
from pathlib import Path
from mne_bids import find_matching_paths, read_raw_bids
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Define EEG datasets
datasets = [
    ("AlexMI", "MNE-BIDS-alexandre-motor-imagery"),
    ("BNCI2014_001", "MNE-BIDS-bnci2014-001"),
    ("PhysionetMI", "MNE-BIDS-physionet-motor-imagery"),
    ("Schirrmeister2017", "MNE-BIDS-schirrmeister2017"),
    ("Weibo2014", "MNE-BIDS-weibo2014"),
    ("Zhou2016", "MNE-BIDS-zhou2016"),
]

# Plot channel montages
for dataset in datasets:
    name, subdir = dataset
    root = path.join(path.expanduser(data_path), "bids", subdir)
    subject, datatype, extension = "1", "eeg", ".edf"
    montage = "standard_1005"
    bids_paths = find_matching_paths(
        root=root, subjects=subject, datatypes=datatype, extensions=extension)
    bids_path = bids_paths[0]
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.set_montage(montage)
    fig = raw.plot_sensors(show_names=True, sphere="auto", show=False)
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(name)
