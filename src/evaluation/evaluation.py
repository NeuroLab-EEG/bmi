"""
Perform cross-subject evaluation with left-/right-hand binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html
.. [2] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html
"""

from itertools import product
from os import getenv, makedirs, path

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
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.utils import set_download_dir, setup_seed

from ..pipelines import BDCNN, BSCNN, CSPBLDA, CSPGP, CSPLDA, CSPSVM, DCNN, SCNN, TSBLR, TSGP, TSLR, TSSVM

subjects = {
    "Zhou2020": [13, 14, 15, 16, 17, 18, 19, 20],
}

sessions = {
    "Stieger2021": [1, 2, 3, 4, 5, 6],
}

# fmt: off
channels = {
    "BNCI2014_001": ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4"],
    "BNCI2014_004": ["C3", "Cz", "C4"],
    "Brandl2020": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Chang2025": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CP2", "CP4", "CP6"],
    "Cho2017": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Dreyer2023": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Forenzo2023": ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CP2", "CP4", "CP6"],
    "GrosseWentrup2009": ["6", "39", "7", "27", "59", "28", "103", "70", "66", "123", "91", "122", "41", "8", "40", "26", "58", "25", "57", "72", "105", "71", "90", "120", "89", "10", "43", "11", "54", "22", "55", "23"],
    "GrosseWentrup2009_1005": ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "GuttmannFlury2025_MI": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "HefmiIch2025": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"],
    "Kumar2024": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"],
    "Lee2019_MI": ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Liu2024": ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    "PhysionetMI": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Schirrmeister2017": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Shin2017A": ["FCC5h", "FCC3h", "FCC4h", "FCC6h", "Cz", "CCP5h", "CCP3h", "CCP4h", "CCP6h"],
    "Stieger2021": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Weibo2014": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Yang2025": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    "Zhou2020": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
}
# fmt: on


class Evaluation:
    def __init__(self):
        # Configure environment
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))
        self.data_path = getenv("DATA_PATH")
        setup_seed(self.random_state)
        set_download_dir(self.data_path)

    def run(self):
        for datasetcls, pipelinecls in product(self._datasets(), self._pipelines()):
            # Make directories
            metrics_path = path.join(
                self.data_path,
                "metrics",
                datasetcls.__name__,
                pipelinecls.__name__,
            )
            emissions_path = path.join(metrics_path, "emissions")
            makedirs(metrics_path, exist_ok=True)
            makedirs(emissions_path, exist_ok=True)

            # Configure evaluation
            dataset = datasetcls(
                subjects=subjects.get(datasetcls.__name__, None),
                sessions=sessions.get(datasetcls.__name__, None),
            )
            paradigm = LeftRightImagery(resample=128, channels=channels.get(datasetcls.__name__, None))
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset],
                paradigm=paradigm,
                hdf5_path=self.data_path,
                save_predictions=True,
                overwrite=False,
                n_splits=len(subjects[datasetcls.__name__])
                if subjects.get(datasetcls.__name__, None)
                else min(dataset.metadata.participants.n_subjects, 10),
                cache_config={
                    "use": True,
                    "save_array": True,
                    "overwrite_array": False,
                },
                codecarbon_config={
                    "save_to_file": True,
                    "output_dir": emissions_path,
                    "log_level": "critical",
                    "country_iso_code": "USA",
                    "region": "washington",
                },
            )

            # Configure pipelines
            X, _, _ = paradigm.get_data(
                dataset,
                cache_config={
                    "use": True,
                    "save_array": True,
                    "overwrite_array": False,
                },
            )
            pipeline = pipelinecls(
                data_path=metrics_path,
                random_state=self.random_state,
                n_features=X.shape[1],
                n_classes=2,
                n_timepoints=X.shape[2],
            )
            pipelines = pipeline.build()

            # Execute pipelines evaluation
            result = evaluation.process(pipelines)
            result.to_csv(path.join(metrics_path, "scores.csv"), index=False)

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

    def _pipelines(self):
        yield CSPLDA
        yield CSPBLDA
        yield CSPSVM
        yield CSPGP
        yield TSLR
        yield TSBLR
        yield TSSVM
        yield TSGP
        yield SCNN
        yield BSCNN
        yield DCNN
        yield BDCNN
