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
from moabb.datasets import (
    BNCI2014_001,
    BNCI2014_004,
    Brandl2020,
    Chang2025,
    Cho2017,
    Dreyer2023,
    Forenzo2023,
    GrosseWentrup2009,
    GuttmannFlury2025_MI,
    HefmiIch2025,
    Kumar2024,
    Lee2019_MI,
    Liu2024,
    PhysionetMI,
    Schirrmeister2017,
    Shin2017A,
    Stieger2021,
    Weibo2014,
    Yang2025,
    Zhou2020,
)
from moabb.utils import set_download_dir


class Download:
    def __init__(self):
        # Configure download
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)
        self.subject_list = None

    def run(self):
        for datasetcls in self._datasets():
            dataset = datasetcls()
            dataset.download(subject_list=self.subject_list, accept=True)

    def _datasets(self):
        yield BNCI2014_001
        yield BNCI2014_004
        yield Brandl2020
        yield Chang2025
        yield Cho2017
        yield Dreyer2023
        yield Forenzo2023
        yield GrosseWentrup2009
        yield GuttmannFlury2025_MI
        yield HefmiIch2025
        yield Kumar2024
        yield Lee2019_MI
        yield Liu2024
        yield PhysionetMI
        yield Schirrmeister2017
        yield Shin2017A
        yield Stieger2021
        yield Weibo2014
        yield Yang2025
        yield Zhou2020
