"""
Perform cross-subject evaluation with left-/right-hand binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html
.. [2] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html
"""

import numpy as np
from os import path, getenv, makedirs
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
        # Make directories
        metrics_path = path.join(self.data_path, "metrics")
        makedirs(metrics_path, exist_ok=True)

        for DatasetCls, n_splits in self._datasets():
            # Make subdirectories
            emissions_path = path.join(metrics_path, DatasetCls.__name__, "emissions")
            traces_path = path.join(metrics_path, DatasetCls.__name__, "traces")
            scores_path = path.join(metrics_path, DatasetCls.__name__)
            makedirs(emissions_path, exist_ok=True)
            makedirs(traces_path, exist_ok=True)
            makedirs(scores_path, exist_ok=True)

            # Configure evaluation
            dataset = DatasetCls()
            paradigm = MultiScoreLeftRightImagery(resample=128)
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset],
                paradigm=paradigm,
                hdf5_path=self.data_path,
                overwrite=True,
                n_splits=n_splits,
                codecarbon_config=dict(
                    save_to_file=True,
                    output_dir=emissions_path,
                    log_level="critical",
                    country_iso_code="USA",
                    region="washington",
                ),
            )

            # Configure pipelines
            X, y, _ = paradigm.get_data(dataset, subjects=[1])
            pipelines = {
                k: v
                for PipelineCls in self._pipelines()
                for k, v in PipelineCls(
                    data_path=traces_path,
                    random_state=self.random_state,
                    n_features=X.shape[1],
                    n_classes=len(np.unique(y)),
                    n_times=X.shape[2],
                )
                .build()
                .items()
            }

            # Execute pipelines evaluation
            result = evaluation.process(pipelines)
            result.to_csv(path.join(scores_path, "scores.csv"), index=False)

    def _datasets(self):
        yield (PhysionetMI, 10)
        yield (BNCI2014_001, 9)
        yield (Lee2019_MI, 10)
        yield (Cho2017, 10)
        yield (Schirrmeister2017, 5)
        yield (Shin2017A, 5)
        yield (BNCI2014_004, 9)
        yield (Dreyer2023, 10)
        yield (Weibo2014, 5)
        yield (GrosseWentrup2009, 5)
        yield (Stieger2021, 10)

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
