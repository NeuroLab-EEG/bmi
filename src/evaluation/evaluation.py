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
from .configs import Channels, Sessions, Splits, Subjects


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
                subjects=Subjects[datasetcls.__name__].value, sessions=Sessions[datasetcls.__name__].value
            )
            paradigm = LeftRightImagery(resample=128, channels=Channels[datasetcls.__name__].value)
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset],
                paradigm=paradigm,
                hdf5_path=self.data_path,
                overwrite=True,
                n_splits=Splits[datasetcls.__name__].value,
                cache_config=dict(
                    use=True,
                    save_array=True,
                    overwrite_array=False,
                ),
                codecarbon_config=dict(
                    save_to_file=True,
                    output_dir=emissions_path,
                    log_level="critical",
                    country_iso_code="USA",
                    region="washington",
                ),
            )

            # Configure pipelines
            X, _, _ = paradigm.get_data(
                dataset,
                cache_config=dict(
                    use=True,
                    save_array=True,
                    overwrite_array=False,
                ),
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
