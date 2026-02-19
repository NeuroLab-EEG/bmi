"""
Custom Liu2024 class for when MOABB's automated download fails.

Manually download the following files to [DATA_PATH]/Liu2024/ where DATA_PATH is defined in the root .env file.

- EDF zip URL: https://figshare.com/ndownloader/files/38516654
- electrodes URL: https://figshare.com/ndownloader/files/38516078
- events URL: https://figshare.com/ndownloader/files/38516084

References
----------
.. [1] https://moabb.neurotechx.com/docs/auto_examples/tutorials/tutorial_4_adding_a_dataset.html
.. [2] https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/liu2024.py
"""

from os import getenv
from dotenv import load_dotenv
import zipfile as z
from pathlib import Path
from moabb.utils import set_download_dir
from moabb.datasets import Liu2024 as MOABBLiu2024


class Liu2024(MOABBLiu2024):
    def __init__(self):
        super().__init__()

        # Configure download
        load_dotenv()
        self.data_dir = getenv("DATA_PATH")
        set_download_dir(self.data_dir)

        self.liu2024_url = f"{self.data_dir}/Liu2024/edffile.zip"
        self.liu2024_electrodes = f"{self.data_dir}/Liu2024/task-motor-imagery_electrodes.tsv"
        self.liu2024_events = f"{self.data_dir}/Liu2024/task-motor-imagery_events.tsv"

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        path_zip = Path("/Users/ethandavis/Desktop/bmi/data/Liu2024/edffile.zip")
        path_folder = path_zip.parent

        # Extract the zip file if it hasn't been extracted yet
        if not (path_folder / "edffile").is_dir():
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths = []
        sub = f"sub-{subject:02d}"

        # Construct the path to the subject's data file
        subject_path = (
            path_folder / "edffile" / sub / "eeg" / f"{sub}_task-motor-imagery_eeg.edf"
        )
        subject_paths.append(str(subject_path))

        return subject_paths

    def data_infos(self):
        return self.liu2024_electrodes, self.liu2024_events
