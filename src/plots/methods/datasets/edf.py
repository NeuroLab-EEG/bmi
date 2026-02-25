import pandas as pd
from os import path, getenv
from pyedflib import EdfReader
from dotenv import load_dotenv
from mne_bids import find_matching_paths
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


class EDF:
    def __init__(self):
        load_dotenv()
        self.data_path = getenv("DATA_PATH")

    def run(self):
        for DatasetCls, subdir in self._params():
            root = path.join(self.data_path, subdir)
            bids_paths = find_matching_paths(root=root, subjects="1", datatypes="eeg", extensions=".edf")
            bids_path = bids_paths[0]
            reader = EdfReader(str(bids_path))
            df = pd.DataFrame(reader.getSignalHeaders())
            df.to_csv(f"{DatasetCls.__name__}.csv", index=False)

    def _params(self):
        yield (BNCI2014_001, "MNE-BIDS-bnci2014-001")
        yield (BNCI2014_004, "MNE-BIDS-bnci2014-004")
        yield (Cho2017, "MNE-BIDS-cho2017")
        yield (Dreyer2023, "MNE-BIDS-dreyer2023")
        yield (Lee2019_MI, "MNE-BIDS-lee2019-mi")
        yield (PhysionetMI, "MNE-BIDS-physionet-motor-imagery")
        yield (Schirrmeister2017, "MNE-BIDS-schirrmeister2017")
        yield (Shin2017A, "MNE-BIDS-shin2017-a")
        yield (Stieger2021, "MNE-BIDS-stieger2021")
        yield (Weibo2014, "MNE-BIDS-weibo2014")
        yield (Liu2024, "MNE-BIDS-liu2024")
        yield (GrosseWentrup2009, "MNE-BIDS-grosse-wentrup2009")
