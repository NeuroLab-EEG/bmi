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
from moabb.datasets import Shin2017A
from src.evaluation import Evaluation


class Download:
    def __init__(self):
        # Configure download
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)

    def run(self):
        for DatasetCls in Evaluation.DATASETS.values():
            dataset = DatasetCls(accept=True) if DatasetCls is Shin2017A else DatasetCls()
            dataset.get_data(cache_config=dict(path=self.data_path, save_raw=True))
