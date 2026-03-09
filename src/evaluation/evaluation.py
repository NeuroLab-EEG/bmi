"""
Perform cross-subject evaluation with left-/right-hand binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html
.. [2] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html
"""

import numpy as np
from os import path, getenv, makedirs
from itertools import product
from dotenv import load_dotenv
from moabb.utils import set_download_dir
from moabb.evaluations import CrossSubjectEvaluation
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
from .configs import N_SPLITS, CHANNELS
from src.datasets import Liu2024
from src.paradigm import MultiScoreLeftRightImagery
from src.pipelines import CSPLDA, CSPSVM, TSLR, TSSVM, SCNN, DCNN, CSPBLDA, CSPGP, TSBLR, TSGP, BSCNN, BDCNN


class Evaluation:
    def __init__(self):
        # Configure environment
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)

    def run(self):
        for DatasetCls, PipelineCls in product(self._datasets(), self._pipelines()):
            # Make directories
            metrics_path = path.join(
                self.data_path,
                "metrics",
                DatasetCls.__name__,
                PipelineCls.__name__,
            )
            makedirs(metrics_path, exist_ok=True)

            # Configure evaluation
            dataset = DatasetCls()
            paradigm = MultiScoreLeftRightImagery(resample=128, channels=CHANNELS[DatasetCls])
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset],
                paradigm=paradigm,
                hdf5_path=self.data_path,
                overwrite=True,
                n_splits=N_SPLITS[DatasetCls],
                codecarbon_config=dict(
                    save_to_file=True,
                    output_dir=metrics_path,
                    log_level="critical",
                    country_iso_code="USA",
                    region="washington",
                ),
            )

            # Configure pipelines
            X, y, _ = paradigm.get_data(dataset, subjects=[1])
            pipeline = PipelineCls(
                data_path=metrics_path,
                random_state=self.random_state,
                n_features=X.shape[1],
                n_classes=len(np.unique(y)),
                n_timepoints=X.shape[2],
            )
            pipelines = pipeline.build()

            # Execute pipelines evaluation
            result = evaluation.process(pipelines)
            result.to_csv(path.join(metrics_path, "scores.csv"), index=False)

    def _datasets(self):
        yield BNCI2014_001
        yield Stieger2021
        yield Schirrmeister2017
        yield Liu2024
        yield GrosseWentrup2009
        yield PhysionetMI
        yield Lee2019_MI
        yield Cho2017
        yield Shin2017A
        yield BNCI2014_004
        yield Dreyer2023
        yield Weibo2014

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
