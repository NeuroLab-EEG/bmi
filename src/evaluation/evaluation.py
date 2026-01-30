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
from src.paradigm import MultiScoreLeftRightImagery
from src.pipelines import (
    CSPLDA, CSPSVM, TSLR, TSSVM, SCNN, DCNN, TSBLR
)


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
        yield from self._beetl2021_a()
        yield from self._beetl2021_b()
        yield from self._dreyer2023()
        yield from self._weibo2014()
        yield from self._zhou2016()
        yield from self._grossewentrup2009()

    def _physionetmi(self):
        yield (CSPLDA, PhysionetMI, 36, 10)
        yield (CSPSVM, PhysionetMI, 36, 10)
        yield (TSLR, PhysionetMI, 36, 10)
        yield (TSSVM, PhysionetMI, 36, 10)
        yield (SCNN, PhysionetMI, 1, 10)
        yield (DCNN, PhysionetMI, 1, 10)
        yield (TSBLR, PhysionetMI, 1, 10)

    def _lee2019_mi(self):
        yield (CSPLDA, Lee2019_MI, 36, 10)
        yield (CSPSVM, Lee2019_MI, 36, 10)
        yield (TSLR, Lee2019_MI, 36, 10)
        yield (TSSVM, Lee2019_MI, 36, 10)
        yield (SCNN, Lee2019_MI, 1, 10)
        yield (DCNN, Lee2019_MI, 1, 10)
        yield (TSBLR, Lee2019_MI, 1, 10)

    def _cho2017(self):
        yield (CSPLDA, Cho2017, 36, 10)
        yield (CSPSVM, Cho2017, 36, 10)
        yield (TSLR, Cho2017, 36, 10)
        yield (TSSVM, Cho2017, 36, 10)
        yield (SCNN, Cho2017, 1, 10)
        yield (DCNN, Cho2017, 1, 10)
        yield (TSBLR, Cho2017, 1, 10)

    def _schirrmeister2017(self):
        yield (CSPLDA, Schirrmeister2017, 36, 5)
        yield (CSPSVM, Schirrmeister2017, 36, 5)
        yield (TSLR, Schirrmeister2017, 36, 5)
        yield (TSSVM, Schirrmeister2017, 36, 5)
        yield (SCNN, Schirrmeister2017, 1, 5)
        yield (DCNN, Schirrmeister2017, 1, 5)
        yield (TSBLR, Schirrmeister2017, 1, 5)

    def _shin2017a(self):
        yield (CSPLDA, Shin2017A, 36, 5)
        yield (CSPSVM, Shin2017A, 36, 5)
        yield (TSLR, Shin2017A, 36, 5)
        yield (TSSVM, Shin2017A, 36, 5)
        yield (SCNN, Shin2017A, 1, 5)
        yield (DCNN, Shin2017A, 1, 5)
        yield (TSBLR, Shin2017A, 1, 5)

    def _bnci2014_001(self):
        yield (CSPLDA, BNCI2014_001, 36, 9)
        yield (CSPSVM, BNCI2014_001, 36, 9)
        yield (TSLR, BNCI2014_001, 36, 9)
        yield (TSSVM, BNCI2014_001, 36, 9)
        yield (SCNN, BNCI2014_001, 1, 9)
        yield (DCNN, BNCI2014_001, 1, 9)
        yield (TSBLR, BNCI2014_001, 1, 9)

    def _bnci2014_004(self):
        yield (CSPLDA, BNCI2014_004, 36, 9)
        yield (CSPSVM, BNCI2014_004, 36, 9)
        yield (TSLR, BNCI2014_004, 36, 9)
        yield (TSSVM, BNCI2014_004, 36, 9)
        yield (SCNN, BNCI2014_004, 1, 9)
        yield (DCNN, BNCI2014_004, 1, 9)
        yield (TSBLR, BNCI2014_004, 1, 9)

    def _beetl2021_a(self):
        yield (CSPLDA, Beetl2021_A, 36, 3)
        yield (CSPSVM, Beetl2021_A, 36, 3)
        yield (TSLR, Beetl2021_A, 36, 3)
        yield (TSSVM, Beetl2021_A, 36, 3)
        yield (SCNN, Beetl2021_A, 1, 3)
        yield (DCNN, Beetl2021_A, 1, 3)
        yield (TSBLR, Beetl2021_A, 1, 3)

    def _beetl2021_b(self):
        yield (CSPLDA, Beetl2021_B, 36, 2)
        yield (CSPSVM, Beetl2021_B, 36, 2)
        yield (TSLR, Beetl2021_B, 36, 2)
        yield (TSSVM, Beetl2021_B, 36, 2)
        yield (SCNN, Beetl2021_B, 1, 2)
        yield (DCNN, Beetl2021_B, 1, 2)
        yield (TSBLR, Beetl2021_B, 1, 2)

    def _dreyer2023(self):
        yield (CSPLDA, Dreyer2023, 36, 10)
        yield (CSPSVM, Dreyer2023, 36, 10)
        yield (TSLR, Dreyer2023, 36, 10)
        yield (TSSVM, Dreyer2023, 36, 10)
        yield (SCNN, Dreyer2023, 1, 10)
        yield (DCNN, Dreyer2023, 1, 10)
        yield (TSBLR, Dreyer2023, 1, 10)

    def _weibo2014(self):
        yield (CSPLDA, Weibo2014, 36, 5)
        yield (CSPSVM, Weibo2014, 36, 5)
        yield (TSLR, Weibo2014, 36, 5)
        yield (TSSVM, Weibo2014, 36, 5)
        yield (SCNN, Weibo2014, 1, 5)
        yield (DCNN, Weibo2014, 1, 5)
        yield (TSBLR, Weibo2014, 1, 5)

    def _zhou2016(self):
        yield (CSPLDA, Zhou2016, 36, 4)
        yield (CSPSVM, Zhou2016, 36, 4)
        yield (TSLR, Zhou2016, 36, 4)
        yield (TSSVM, Zhou2016, 36, 4)
        yield (SCNN, Zhou2016, 1, 4)
        yield (DCNN, Zhou2016, 1, 4)
        yield (TSBLR, Zhou2016, 1, 4)

    def _grossewentrup2009(self):
        yield (CSPLDA, GrosseWentrup2009, 36, 5)
        yield (CSPSVM, GrosseWentrup2009, 36, 5)
        yield (TSLR, GrosseWentrup2009, 36, 5)
        yield (TSSVM, GrosseWentrup2009, 36, 5)
        yield (SCNN, GrosseWentrup2009, 1, 5)
        yield (DCNN, GrosseWentrup2009, 1, 5)
        yield (TSBLR, GrosseWentrup2009, 1, 5)


Evaluation()()
