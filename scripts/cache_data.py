"""
Description: Cache a MOABB dataset as BIDS
References:
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_changing_download_directory.html
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_bids_conversion.html
"""

import os.path as osp
from moabb.utils import set_download_dir
from moabb.datasets import (
    AlexMI,
    BNCI2014_001,
    PhysionetMI,
    Schirrmeister2017,
    Weibo2014,
    Zhou2016
)

# Change download directory
new_path = osp.join(osp.expanduser("/data/davise5"), "data")
set_download_dir(new_path)

# Define BIDS directory
path = osp.join(osp.expanduser("/data/davise5"), "data/bids")

# Download a MOABB dataset
datasets = [AlexMI, BNCI2014_001, PhysionetMI, Schirrmeister2017, Weibo2014, Zhou2016]
for dataset in datasets:
    d = dataset()
    d.get_data()
