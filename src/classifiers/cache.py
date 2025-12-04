"""
Cache MOABB database
References:
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_changing_download_directory.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_bids_conversion.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/paper_results.html#motor-imagery-left-vs-right-hand  # noqa: E501
    - https://moabb.neurotechx.com/docs/dataset_summary.html#motor-imagery
"""

from os import getenv
from dotenv import load_dotenv
from moabb.utils import set_download_dir
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

# Download MOABB datasets
datasets = [
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
]
for dataset in datasets:
    d = dataset(accept=True) if dataset is Shin2017A else dataset()
    d.get_data(cache_config=dict(path=data_path, save_raw=True))
