"""
Perform cross-subject evaluation with left-/right-hand binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation
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
)
from src.paradigm.paradigm import MultiScoreLeftRightImagery
from src.pipelines.raw_signal.csplda import CSPLDA
from src.pipelines.raw_signal.cspsvm import CSPSVM
from src.pipelines.riemannian.tslr import TSLR
from src.pipelines.riemannian.tssvm import TSSVM
from src.pipelines.deep_learning.scnn import SCNN
from src.pipelines.deep_learning.dcnn import DCNN


class Evaluation:
    def __init__(self):
        # Configure environment
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)

    def __call__(self):
        # Make directories
        metrics_path = path.join(self.data_path, "metrics")
        makedirs(metrics_path, exist_ok=True)
        
        for PipelineCls, ParadigmCls, resample, DatasetCls, jobs, epochs, splits in self._params():
            # Make subdirectories
            emissions_path = path.join(metrics_path, DatasetCls.__name__, "emissions")
            scores_path = path.join(metrics_path, DatasetCls.__name__, "scores")
            makedirs(emissions_path, exist_ok=True)
            makedirs(scores_path, exist_ok=True)

            # Configure evaluation
            dataset = DatasetCls()
            paradigm = ParadigmCls(resample=resample)
            X, y, metadata = paradigm.get_data(dataset, subjects=[1])
            pipeline = PipelineCls(
                n_chans=X.shape[1],
                n_outputs=len(np.unique(y)),
                n_times=X.shape[2]
            )
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset],
                paradigm=paradigm,
                hdf5_path=self.data_path,
                overwrite=True,
                n_jobs=jobs,
                return_epochs=epochs,
                n_splits=splits,
                codecarbon_config=dict(
                    save_to_file=True,
                    output_dir=emissions_path,
                    log_level="critical",
                    tracking_mode="process",
                    country_iso_code="USA",
                    region="washington",
                ),
            )

            # Execute evaluation
            result = evaluation.process(pipeline.pipeline(), pipeline.params())
            result.to_csv(path.join(scores_path, f"{PipelineCls.__name__}.csv"), index=False)

    def _params(self):
        yield from self._physionetmi()
        yield from self._lee2019_mi()
        yield from self._cho2017()
        yield from self._schirrmeister2017()
        yield from self._shin2017a()
        yield from self._bnci2014_001()

    def _physionetmi(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 160, PhysionetMI, 1, True, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 160, PhysionetMI, 1, True, 10)

    def _lee2019_mi(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 1, True, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 1, True, 10)

    def _cho2017(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 512, Cho2017, 1, True, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 512, Cho2017, 1, True, 10)

    def _schirrmeister2017(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (CSPSVM, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (TSLR, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (TSSVM, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (SCNN, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 1, True, 5)
        yield (DCNN, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 1, True, 5)

    def _shin2017a(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (CSPSVM, MultiScoreLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (TSLR, MultiScoreLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (TSSVM, MultiScoreLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (SCNN, MultiScoreLeftRightImagery, 200, Shin2017A, 1, True, 5)
        yield (DCNN, MultiScoreLeftRightImagery, 200, Shin2017A, 1, True, 5)

    def _bnci2014_001(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (CSPSVM, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (TSLR, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (TSSVM, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (SCNN, MultiScoreLeftRightImagery, 250, BNCI2014_001, 1, True, 9)
        yield (DCNN, MultiScoreLeftRightImagery, 250, BNCI2014_001, 1, True, 9)


Evaluation()()
