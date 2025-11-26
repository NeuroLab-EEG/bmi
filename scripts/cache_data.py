"""
Description: Cache MOABB datasets
References:
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_changing_download_directory.html
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_bids_conversion.html
"""

from os import path, getenv
from dotenv import load_dotenv
from moabb.utils import set_download_dir
from moabb.datasets import (
    AlexMI,
    BNCI2014_001,
    PhysionetMI,
    Schirrmeister2017,
    Weibo2014,
    Zhou2016
)

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
new_path = path.join(path.expanduser(data_path), "raw")
set_download_dir(new_path)

# Download a MOABB dataset
datasets = [AlexMI, BNCI2014_001, PhysionetMI, Schirrmeister2017, Weibo2014, Zhou2016]
for dataset in datasets:
    d = dataset()
    d.get_data()
