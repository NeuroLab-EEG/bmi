"""
Perform cross-subject evaluation with binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation
.. [2] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html
"""

from os import path, getenv
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
from src.paradigm.paradigm import LogLossLeftRightImagery
from src.pipelines.raw_signal import CSPLDA, CSPSVM
from src.pipelines.riemannian import TSLR, TSSVM
from src.pipelines.deep_learning import SCNN, DCNN


class Evaluation:
    def __init__(self):
        # Configure evaluation
        load_dotenv()
        self.data_path = getenv("DATA_PATH")
        set_download_dir(self.data_path)

    def evaluate(self):
        for pipeline, paradigm, resample, dataset, jobs, epochs, splits in self.params():
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset()],
                paradigm=paradigm(resample=resample),
                hdf5_path=self.data_path,
                save_model=True,
                n_jobs=jobs,
                return_epochs=epochs,
                n_splits=splits,
                codecarbon_config=dict(
                    save_to_file=True,
                    output_dir=path.join(self.data_path, "codecarbon"),
                    log_level="error",
                    tracking_mode="process",
                ),
            )
            p = pipeline()
            evaluation.process(p.pipeline(), p.params())

    def params(self):
        yield from self.physionetmi()
        yield from self.lee2019_mi()
        yield from self.cho2017()
        yield from self.schirrmeister2017()
        yield from self.shin2017a()
        yield from self.bnci2014_001()

    def physionetmi(self):
        yield (CSPLDA, LogLossLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (CSPSVM, LogLossLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (TSLR, LogLossLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (TSSVM, LogLossLeftRightImagery, 160, PhysionetMI, 36, False, 10)
        yield (SCNN, LogLossLeftRightImagery, 160, PhysionetMI, 4, True, 10)
        yield (DCNN, LogLossLeftRightImagery, 160, PhysionetMI, 4, True, 10)

    def lee2019_mi(self):
        yield (CSPLDA, LogLossLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (CSPSVM, LogLossLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (TSLR, LogLossLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (TSSVM, LogLossLeftRightImagery, 1000, Lee2019_MI, 36, False, 10)
        yield (SCNN, LogLossLeftRightImagery, 1000, Lee2019_MI, 4, True, 10)
        yield (DCNN, LogLossLeftRightImagery, 1000, Lee2019_MI, 4, True, 10)

    def cho2017(self):
        yield (CSPLDA, LogLossLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (CSPSVM, LogLossLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (TSLR, LogLossLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (TSSVM, LogLossLeftRightImagery, 512, Cho2017, 36, False, 10)
        yield (SCNN, LogLossLeftRightImagery, 512, Cho2017, 4, True, 10)
        yield (DCNN, LogLossLeftRightImagery, 512, Cho2017, 4, True, 10)

    def schirrmeister2017(self):
        yield (CSPLDA, LogLossLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (CSPSVM, LogLossLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (TSLR, LogLossLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (TSSVM, LogLossLeftRightImagery, 500, Schirrmeister2017, 36, False, 5)
        yield (SCNN, LogLossLeftRightImagery, 500, Schirrmeister2017, 4, True, 5)
        yield (DCNN, LogLossLeftRightImagery, 500, Schirrmeister2017, 4, True, 5)

    def shin2017a(self):
        yield (CSPLDA, LogLossLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (CSPSVM, LogLossLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (TSLR, LogLossLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (TSSVM, LogLossLeftRightImagery, 200, Shin2017A, 36, False, 5)
        yield (SCNN, LogLossLeftRightImagery, 200, Shin2017A, 4, True, 5)
        yield (DCNN, LogLossLeftRightImagery, 200, Shin2017A, 4, True, 5)

    def bnci2014_001(self):
        yield (CSPLDA, LogLossLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (CSPSVM, LogLossLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (TSLR, LogLossLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (TSSVM, LogLossLeftRightImagery, 250, BNCI2014_001, 36, False, 9)
        yield (SCNN, LogLossLeftRightImagery, 250, BNCI2014_001, 4, True, 9)
        yield (DCNN, LogLossLeftRightImagery, 250, BNCI2014_001, 4, True, 9)


Evaluation().evaluate()
