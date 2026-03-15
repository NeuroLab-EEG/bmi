"""
Write montage channels to disk.

References
----------
.. [1] https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
.. [2] https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
"""

import mne
import pandas as pd
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
    Dreyer2023,
    Weibo2014,
    GrosseWentrup2009,
    Stieger2021,
)
from src.datasets import Liu2024


class Channels:
    CHANNELS = (
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FCC5h",
        "FCC3h",
        "FFC1h",
        "FCC2h",
        "FCC4h",
        "FCC6h",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CCP5h",
        "CCP3h",
        "CCP1h",
        "CCP2h",
        "CCP4h",
        "CCP6h",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
    )

    def __init__(self):
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        self.params = self._params()

    def run(self):
        for DatasetCls, subdir in self._params():
            root = path.join(self.data_path, subdir)
            bids_paths = find_matching_paths(root=root, subjects="1", datatypes="eeg", extensions=".edf")
            bids_path = bids_paths[0]
            raw = read_raw_bids(bids_path=bids_path, verbose=False)

            ch_names = None
            if DatasetCls.__name__ == GrosseWentrup2009.__name__:
                ch_names = self._rename_channels(raw)

            for func in self._funcs():
                func(raw, DatasetCls.__name__, ch_names)

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

    def _funcs(self):
        yield self._save_montage
        yield self._save_intersection

    def _rename_channels(self, raw):
        orig_ch_names = raw.ch_names
        montage_name = "brainproducts-RNP-BA-128"
        std = mne.channels.make_standard_montage(montage_name)
        rename_map = {str(i + 1): name for i, name in enumerate(std.ch_names) if str(i + 1) in orig_ch_names}
        raw.rename_channels(rename_map)
        raw.set_montage(montage_name)
        return orig_ch_names

    def _save_montage(self, raw, classname, ch_names=None):
        df = pd.DataFrame(raw.ch_names, columns=["ch"])
        if ch_names is not None:
            df["orig"] = ch_names
        df.to_csv(f"{classname}-dev.csv", index=False)

    def _save_intersection(self, raw, classname, ch_names=None):
        intersection = set(raw.ch_names) & set(Channels.CHANNELS)
        df = pd.DataFrame(sorted(intersection, key=lambda ch: Channels.CHANNELS.index(ch)), columns=["ch"])
        if ch_names is not None:
            df["orig"] = [
                ch_names[raw.ch_names.index(ch)]
                for ch in sorted(intersection, key=lambda ch: Channels.CHANNELS.index(ch))
            ]
        df.to_csv(f"{classname}-prod.csv", index=False)
