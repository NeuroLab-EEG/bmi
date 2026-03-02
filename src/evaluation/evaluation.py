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
from src.datasets import Liu2024
from src.paradigm import MultiScoreLeftRightImagery
from src.pipelines import CSPLDA, CSPSVM, TSLR, TSSVM, SCNN, DCNN, CSPBLDA, CSPGP, TSBLR, TSGP, BSCNN, BDCNN


class Evaluation:
    DATASETS = {
        BNCI2014_001.__name__: BNCI2014_001,
        Dreyer2023.__name__: Dreyer2023,
        Stieger2021.__name__: Stieger2021,
        PhysionetMI.__name__: PhysionetMI,
        Lee2019_MI.__name__: Lee2019_MI,
        Cho2017.__name__: Cho2017,
        Schirrmeister2017.__name__: Schirrmeister2017,
        Shin2017A.__name__: Shin2017A,
        BNCI2014_004.__name__: BNCI2014_004,
        Weibo2014.__name__: Weibo2014,
        GrosseWentrup2009.__name__: GrosseWentrup2009,
        Liu2024.__name__: Liu2024,
    }

    N_SPLITS = {
        BNCI2014_001.__name__: 9,
        Dreyer2023.__name__: 10,
        Stieger2021.__name__: 10,
        PhysionetMI.__name__: 10,
        Lee2019_MI.__name__: 10,
        Cho2017.__name__: 10,
        Schirrmeister2017.__name__: 5,
        Shin2017A.__name__: 5,
        BNCI2014_004.__name__: 9,
        Weibo2014.__name__: 5,
        GrosseWentrup2009.__name__: 5,
        Liu2024.__name__: 10,
    }

    PIPELINES = {
        CSPLDA.__name__: CSPLDA,
        CSPBLDA.__name__: CSPBLDA,
        CSPSVM.__name__: CSPSVM,
        CSPGP.__name__: CSPGP,
        TSLR.__name__: TSLR,
        TSBLR.__name__: TSBLR,
        TSSVM.__name__: TSSVM,
        TSGP.__name__: TSGP,
        SCNN.__name__: SCNN,
        BSCNN.__name__: BSCNN,
        DCNN.__name__: DCNN,
        BDCNN.__name__: BDCNN,
    }

    def __init__(self, dataset=None, pipeline=None):
        # Configure environment
        load_dotenv()
        self.random_state = int(getenv("RANDOM_STATE"))
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)

        self.DatasetCls = Evaluation.DATASETS[dataset]
        self.n_splits = Evaluation.N_SPLITS[dataset]
        self.PipelineCls = Evaluation.PIPELINES[pipeline]

    def run(self):
        # Make directories
        metrics_path = path.join(
            self.data_path,
            "metrics",
            self.DatasetCls.__name__,
            self.PipelineCls.__name__,
        )
        makedirs(metrics_path, exist_ok=True)

        # Configure evaluation
        dataset = self.DatasetCls()
        paradigm = MultiScoreLeftRightImagery(resample=128)
        evaluation = CrossSubjectEvaluation(
            datasets=[dataset],
            paradigm=paradigm,
            hdf5_path=self.data_path,
            overwrite=True,
            n_splits=self.n_splits,
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
        pipeline = self.PipelineCls(
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
