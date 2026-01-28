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
    BNCI2014_004,
    Beetl2021_A,
    Beetl2021_B,
    Dreyer2023,
    Weibo2014,
    Zhou2016,
    GrosseWentrup2009,
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

        for PipelineCls, ParadigmCls, resample, DatasetCls, n_jobs, n_splits in self._params():
            # Make subdirectories
            emissions_path = path.join(metrics_path, DatasetCls.__name__, "emissions")
            scores_path = path.join(metrics_path, DatasetCls.__name__, "scores")
            makedirs(emissions_path, exist_ok=True)
            makedirs(scores_path, exist_ok=True)

            # Configure evaluation
            dataset = DatasetCls()
            paradigm = ParadigmCls(resample=resample)
            subject = 4 if DatasetCls is Beetl2021_B else 1
            X, y, _ = paradigm.get_data(dataset, subjects=[subject])
            pipeline = PipelineCls(n_features=X.shape[1], n_classes=len(np.unique(y)), n_times=X.shape[2])
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset],
                paradigm=paradigm,
                hdf5_path=self.data_path,
                overwrite=True,
                n_jobs=n_jobs,
                n_splits=n_splits,
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
        yield from self._bnci2014_004()
        yield from self._beetl2021_a()
        yield from self._beetl2021_b()
        yield from self._dreyer2023()
        yield from self._weibo2014()
        yield from self._zhou2016()
        yield from self._grossewentrup2009()

    def _physionetmi(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 160, PhysionetMI, 36, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 160, PhysionetMI, 1, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 160, PhysionetMI, 1, 10)

    def _lee2019_mi(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 36, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 1, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 1000, Lee2019_MI, 1, 10)

    def _cho2017(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 512, Cho2017, 36, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 512, Cho2017, 36, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 512, Cho2017, 36, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 512, Cho2017, 36, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 512, Cho2017, 1, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 512, Cho2017, 1, 10)

    def _schirrmeister2017(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, 5)
        yield (CSPSVM, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, 5)
        yield (TSLR, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, 5)
        yield (TSSVM, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 36, 5)
        yield (SCNN, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 1, 5)
        yield (DCNN, MultiScoreLeftRightImagery, 500, Schirrmeister2017, 1, 5)

    def _shin2017a(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 200, Shin2017A, 36, 5)
        yield (CSPSVM, MultiScoreLeftRightImagery, 200, Shin2017A, 36, 5)
        yield (TSLR, MultiScoreLeftRightImagery, 200, Shin2017A, 36, 5)
        yield (TSSVM, MultiScoreLeftRightImagery, 200, Shin2017A, 36, 5)
        yield (SCNN, MultiScoreLeftRightImagery, 200, Shin2017A, 1, 5)
        yield (DCNN, MultiScoreLeftRightImagery, 200, Shin2017A, 1, 5)

    def _bnci2014_001(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, 9)
        yield (CSPSVM, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, 9)
        yield (TSLR, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, 9)
        yield (TSSVM, MultiScoreLeftRightImagery, 250, BNCI2014_001, 36, 9)
        yield (SCNN, MultiScoreLeftRightImagery, 250, BNCI2014_001, 1, 9)
        yield (DCNN, MultiScoreLeftRightImagery, 250, BNCI2014_001, 1, 9)

    def _bnci2014_004(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 250, BNCI2014_004, 36, 9)
        yield (CSPSVM, MultiScoreLeftRightImagery, 250, BNCI2014_004, 36, 9)
        yield (TSLR, MultiScoreLeftRightImagery, 250, BNCI2014_004, 36, 9)
        yield (TSSVM, MultiScoreLeftRightImagery, 250, BNCI2014_004, 36, 9)
        yield (SCNN, MultiScoreLeftRightImagery, 250, BNCI2014_004, 1, 9)
        yield (DCNN, MultiScoreLeftRightImagery, 250, BNCI2014_004, 1, 9)

    def _beetl2021_a(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 500, Beetl2021_A, 36, 3)
        yield (CSPSVM, MultiScoreLeftRightImagery, 500, Beetl2021_A, 36, 3)
        yield (TSLR, MultiScoreLeftRightImagery, 500, Beetl2021_A, 36, 3)
        yield (TSSVM, MultiScoreLeftRightImagery, 500, Beetl2021_A, 36, 3)
        yield (SCNN, MultiScoreLeftRightImagery, 500, Beetl2021_A, 1, 3)
        yield (DCNN, MultiScoreLeftRightImagery, 500, Beetl2021_A, 1, 3)

    def _beetl2021_b(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 200, Beetl2021_B, 36, 2)
        yield (CSPSVM, MultiScoreLeftRightImagery, 200, Beetl2021_B, 36, 2)
        yield (TSLR, MultiScoreLeftRightImagery, 200, Beetl2021_B, 36, 2)
        yield (TSSVM, MultiScoreLeftRightImagery, 200, Beetl2021_B, 36, 2)
        yield (SCNN, MultiScoreLeftRightImagery, 200, Beetl2021_B, 1, 2)
        yield (DCNN, MultiScoreLeftRightImagery, 200, Beetl2021_B, 1, 2)

    def _dreyer2023(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 512, Dreyer2023, 36, 10)
        yield (CSPSVM, MultiScoreLeftRightImagery, 512, Dreyer2023, 36, 10)
        yield (TSLR, MultiScoreLeftRightImagery, 512, Dreyer2023, 36, 10)
        yield (TSSVM, MultiScoreLeftRightImagery, 512, Dreyer2023, 36, 10)
        yield (SCNN, MultiScoreLeftRightImagery, 512, Dreyer2023, 1, 10)
        yield (DCNN, MultiScoreLeftRightImagery, 512, Dreyer2023, 1, 10)

    def _weibo2014(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 200, Weibo2014, 36, 5)
        yield (CSPSVM, MultiScoreLeftRightImagery, 200, Weibo2014, 36, 5)
        yield (TSLR, MultiScoreLeftRightImagery, 200, Weibo2014, 36, 5)
        yield (TSSVM, MultiScoreLeftRightImagery, 200, Weibo2014, 36, 5)
        yield (SCNN, MultiScoreLeftRightImagery, 200, Weibo2014, 1, 5)
        yield (DCNN, MultiScoreLeftRightImagery, 200, Weibo2014, 1, 5)

    def _zhou2016(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 250, Zhou2016, 36, 4)
        yield (CSPSVM, MultiScoreLeftRightImagery, 250, Zhou2016, 36, 4)
        yield (TSLR, MultiScoreLeftRightImagery, 250, Zhou2016, 36, 4)
        yield (TSSVM, MultiScoreLeftRightImagery, 250, Zhou2016, 36, 4)
        yield (SCNN, MultiScoreLeftRightImagery, 250, Zhou2016, 1, 4)
        yield (DCNN, MultiScoreLeftRightImagery, 250, Zhou2016, 1, 4)

    def _grossewentrup2009(self):
        yield (CSPLDA, MultiScoreLeftRightImagery, 500, GrosseWentrup2009, 36, 5)
        yield (CSPSVM, MultiScoreLeftRightImagery, 500, GrosseWentrup2009, 36, 5)
        yield (TSLR, MultiScoreLeftRightImagery, 500, GrosseWentrup2009, 36, 5)
        yield (TSSVM, MultiScoreLeftRightImagery, 500, GrosseWentrup2009, 36, 5)
        yield (SCNN, MultiScoreLeftRightImagery, 500, GrosseWentrup2009, 1, 5)
        yield (DCNN, MultiScoreLeftRightImagery, 500, GrosseWentrup2009, 1, 5)

Evaluation()()
