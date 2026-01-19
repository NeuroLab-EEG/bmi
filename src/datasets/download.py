"""
Cache MOABB database.

References
----------
.. [1] https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_changing_download_directory.html
.. [2] https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_bids_conversion.html
.. [3] https://moabb.neurotechx.com/docs/paper_results.html#motor-imagery-left-vs-right-hand
.. [4] https://moabb.neurotechx.com/docs/dataset_summary.html#motor-imagery
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
    BNCI2014_004,
)


class Download:
    def __init__(self):
        # Configure download
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)

    def download(self):
        for dataset in self.datasets():
            d = dataset(accept=True) if dataset is Shin2017A else dataset()
            d.get_data(cache_config=dict(path=self.data_path, save_raw=True))

    def datasets(self):
        yield PhysionetMI
        yield Lee2019_MI
        yield Cho2017
        yield Schirrmeister2017
        yield Shin2017A
        yield BNCI2014_001
        yield BNCI2014_004


Download().download()
