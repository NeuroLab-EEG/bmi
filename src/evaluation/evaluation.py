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
    Dreyer2023,
    Weibo2014,
    GrosseWentrup2009,
)
from src.paradigm import MultiScoreLeftRightImagery
from src.pipelines import CSPLDA, CSPSVM, TSLR, TSSVM, SCNN, DCNN, CSPBLDA, CSPGP, TSBLR, TSGP


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

        for PipelineCls, DatasetCls, n_jobs, n_splits in self._params():
            # Make subdirectories
            emissions_path = path.join(metrics_path, DatasetCls.__name__, "emissions")
            scores_path = path.join(metrics_path, DatasetCls.__name__, "scores")
            makedirs(emissions_path, exist_ok=True)
            makedirs(scores_path, exist_ok=True)

            # Configure evaluation
            dataset = DatasetCls()
            paradigm = MultiScoreLeftRightImagery(resample=128)
            X, y, _ = paradigm.get_data(dataset, subjects=[1])
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
                    country_iso_code="USA",
                    region="washington",
                ),
            )

            # Execute evaluation
            result = evaluation.process(pipeline.pipeline(), pipeline.params())
            result.to_csv(path.join(scores_path, f"{PipelineCls.__name__}.csv"), index=False)

    def _params(self):
        yield from self._bnci2014_001()
        yield from self._physionetmi()
        yield from self._lee2019_mi()
        yield from self._cho2017()
        yield from self._schirrmeister2017()
        yield from self._shin2017a()
        yield from self._bnci2014_004()
        yield from self._dreyer2023()
        yield from self._weibo2014()
        yield from self._grossewentrup2009()

    def _physionetmi(self):
        yield (CSPLDA, PhysionetMI, 36, 10)
        yield (CSPSVM, PhysionetMI, 36, 10)
        yield (TSLR, PhysionetMI, 36, 10)
        yield (TSSVM, PhysionetMI, 36, 10)
        yield (SCNN, PhysionetMI, 1, 10)
        yield (DCNN, PhysionetMI, 1, 10)
        yield (CSPBLDA, PhysionetMI, 1, 10)
        yield (CSPGP, PhysionetMI, 1, 10)
        yield (TSBLR, PhysionetMI, 1, 10)
        yield (TSGP, PhysionetMI, 1, 10)

    def _lee2019_mi(self):
        yield (CSPLDA, Lee2019_MI, 36, 10)
        yield (CSPSVM, Lee2019_MI, 36, 10)
        yield (TSLR, Lee2019_MI, 36, 10)
        yield (TSSVM, Lee2019_MI, 36, 10)
        yield (SCNN, Lee2019_MI, 1, 10)
        yield (DCNN, Lee2019_MI, 1, 10)
        yield (CSPBLDA, Lee2019_MI, 1, 10)
        yield (CSPGP, Lee2019_MI, 1, 10)
        yield (TSBLR, Lee2019_MI, 1, 10)
        yield (TSGP, Lee2019_MI, 1, 10)

    def _cho2017(self):
        yield (CSPLDA, Cho2017, 36, 10)
        yield (CSPSVM, Cho2017, 36, 10)
        yield (TSLR, Cho2017, 36, 10)
        yield (TSSVM, Cho2017, 36, 10)
        yield (SCNN, Cho2017, 1, 10)
        yield (DCNN, Cho2017, 1, 10)
        yield (CSPBLDA, Cho2017, 1, 10)
        yield (CSPGP, Cho2017, 1, 10)
        yield (TSBLR, Cho2017, 1, 10)
        yield (TSGP, Cho2017, 1, 10)

    def _schirrmeister2017(self):
        yield (CSPLDA, Schirrmeister2017, 36, 5)
        yield (CSPSVM, Schirrmeister2017, 36, 5)
        yield (TSLR, Schirrmeister2017, 36, 5)
        yield (TSSVM, Schirrmeister2017, 36, 5)
        yield (SCNN, Schirrmeister2017, 1, 5)
        yield (DCNN, Schirrmeister2017, 1, 5)
        yield (CSPBLDA, Schirrmeister2017, 1, 5)
        yield (CSPGP, Schirrmeister2017, 1, 5)
        yield (TSBLR, Schirrmeister2017, 1, 5)
        yield (TSGP, Schirrmeister2017, 1, 5)

    def _shin2017a(self):
        yield (CSPLDA, Shin2017A, 36, 5)
        yield (CSPSVM, Shin2017A, 36, 5)
        yield (TSLR, Shin2017A, 36, 5)
        yield (TSSVM, Shin2017A, 36, 5)
        yield (SCNN, Shin2017A, 1, 5)
        yield (DCNN, Shin2017A, 1, 5)
        yield (CSPBLDA, Shin2017A, 1, 5)
        yield (CSPGP, Shin2017A, 1, 5)
        yield (TSBLR, Shin2017A, 1, 5)
        yield (TSGP, Shin2017A, 1, 5)

    def _bnci2014_001(self):
        yield (CSPLDA, BNCI2014_001, 36, 9)
        yield (CSPSVM, BNCI2014_001, 36, 9)
        yield (TSLR, BNCI2014_001, 36, 9)
        yield (TSSVM, BNCI2014_001, 36, 9)
        yield (SCNN, BNCI2014_001, 1, 9)
        yield (DCNN, BNCI2014_001, 1, 9)
        yield (CSPBLDA, BNCI2014_001, 1, 9)
        yield (CSPGP, BNCI2014_001, 1, 9)
        yield (TSBLR, BNCI2014_001, 1, 9)
        yield (TSGP, BNCI2014_001, 1, 9)

    def _bnci2014_004(self):
        yield (CSPLDA, BNCI2014_004, 36, 9)
        yield (CSPSVM, BNCI2014_004, 36, 9)
        yield (TSLR, BNCI2014_004, 36, 9)
        yield (TSSVM, BNCI2014_004, 36, 9)
        yield (SCNN, BNCI2014_004, 1, 9)
        yield (DCNN, BNCI2014_004, 1, 9)
        yield (CSPBLDA, BNCI2014_004, 1, 9)
        yield (CSPGP, BNCI2014_004, 1, 9)
        yield (TSBLR, BNCI2014_004, 1, 9)
        yield (TSGP, BNCI2014_004, 1, 9)

    def _dreyer2023(self):
        yield (CSPLDA, Dreyer2023, 36, 10)
        yield (CSPSVM, Dreyer2023, 36, 10)
        yield (TSLR, Dreyer2023, 36, 10)
        yield (TSSVM, Dreyer2023, 36, 10)
        yield (SCNN, Dreyer2023, 1, 10)
        yield (DCNN, Dreyer2023, 1, 10)
        yield (CSPBLDA, Dreyer2023, 1, 10)
        yield (CSPGP, Dreyer2023, 1, 10)
        yield (TSBLR, Dreyer2023, 1, 10)
        yield (TSGP, Dreyer2023, 1, 10)

    def _weibo2014(self):
        yield (CSPLDA, Weibo2014, 36, 5)
        yield (CSPSVM, Weibo2014, 36, 5)
        yield (TSLR, Weibo2014, 36, 5)
        yield (TSSVM, Weibo2014, 36, 5)
        yield (SCNN, Weibo2014, 1, 5)
        yield (DCNN, Weibo2014, 1, 5)
        yield (CSPBLDA, Weibo2014, 1, 5)
        yield (CSPGP, Weibo2014, 1, 5)
        yield (TSBLR, Weibo2014, 1, 5)
        yield (TSGP, Weibo2014, 1, 5)

    def _grossewentrup2009(self):
        yield (CSPLDA, GrosseWentrup2009, 36, 5)
        yield (CSPSVM, GrosseWentrup2009, 36, 5)
        yield (TSLR, GrosseWentrup2009, 36, 5)
        yield (TSSVM, GrosseWentrup2009, 36, 5)
        yield (SCNN, GrosseWentrup2009, 1, 5)
        yield (DCNN, GrosseWentrup2009, 1, 5)
        yield (CSPBLDA, GrosseWentrup2009, 1, 5)
        yield (CSPGP, GrosseWentrup2009, 1, 5)
        yield (TSBLR, GrosseWentrup2009, 1, 5)
        yield (TSGP, GrosseWentrup2009, 1, 5)
