"""
Perform cross-subject evaluation with binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation  # noqa: E501
.. [2] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html  # noqa: E501
"""

from os import getenv
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
        for pipeline, paradigm, resample, dataset, epochs, splits in self.parameters():
            evaluation = CrossSubjectEvaluation(
                datasets=[dataset()],
                paradigm=paradigm(resample=resample),
                hdf5_path=self.data_path,
                save_model=True,
                return_epochs=epochs,
                n_splits=splits,
            )
            p = pipeline()
            evaluation.process(p.pipeline(), p.params())

    def parameters(self):
        yield from self.physionetmi()
        yield from self.lee2019_mi()
        yield from self.cho2017()
        yield from self.schirrmeister2017()
        yield from self.shin2017a()
        yield from self.bnci2014_001()

    def physionetmi(self):
        yield from [
            (CSPLDA, LogLossLeftRightImagery, 160, PhysionetMI, False, 10),
            (CSPSVM, LogLossLeftRightImagery, 160, PhysionetMI, False, 10),
            (TSLR, LogLossLeftRightImagery, 160, PhysionetMI, False, 10),
            (TSSVM, LogLossLeftRightImagery, 160, PhysionetMI, False, 10),
            (SCNN, LogLossLeftRightImagery, 160, PhysionetMI, True, 10),
            (DCNN, LogLossLeftRightImagery, 160, PhysionetMI, True, 10),
        ]

    def lee2019_mi(self):
        yield from [
            (CSPLDA, LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10),
            (CSPSVM, LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10),
            (TSLR, LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10),
            (TSSVM, LogLossLeftRightImagery, 1000, Lee2019_MI, False, 10),
            (SCNN, LogLossLeftRightImagery, 1000, Lee2019_MI, True, 10),
            (DCNN, LogLossLeftRightImagery, 1000, Lee2019_MI, True, 10),
        ]

    def cho2017(self):
        yield from [
            (CSPLDA, LogLossLeftRightImagery, 512, Cho2017, False, 10),
            (CSPSVM, LogLossLeftRightImagery, 512, Cho2017, False, 10),
            (TSLR, LogLossLeftRightImagery, 512, Cho2017, False, 10),
            (TSSVM, LogLossLeftRightImagery, 512, Cho2017, False, 10),
            (SCNN, LogLossLeftRightImagery, 512, Cho2017, True, 10),
            (DCNN, LogLossLeftRightImagery, 512, Cho2017, True, 10),
        ]

    def schirrmeister2017(self):
        yield from [
            (CSPLDA, LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5),
            (CSPSVM, LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5),
            (TSLR, LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5),
            (TSSVM, LogLossLeftRightImagery, 500, Schirrmeister2017, False, 5),
            (SCNN, LogLossLeftRightImagery, 500, Schirrmeister2017, True, 5),
            (DCNN, LogLossLeftRightImagery, 500, Schirrmeister2017, True, 5),
        ]

    def shin2017a(self):
        yield from [
            (CSPLDA, LogLossLeftRightImagery, 200, Shin2017A, False, 5),
            (CSPSVM, LogLossLeftRightImagery, 200, Shin2017A, False, 5),
            (TSLR, LogLossLeftRightImagery, 200, Shin2017A, False, 5),
            (TSSVM, LogLossLeftRightImagery, 200, Shin2017A, False, 5),
            (SCNN, LogLossLeftRightImagery, 200, Shin2017A, True, 5),
            (DCNN, LogLossLeftRightImagery, 200, Shin2017A, True, 5),
        ]

    def bnci2014_001(self):
        yield from [
            (CSPLDA, LogLossLeftRightImagery, 250, BNCI2014_001, False, 9),
            (CSPSVM, LogLossLeftRightImagery, 250, BNCI2014_001, False, 9),
            (TSLR, LogLossLeftRightImagery, 250, BNCI2014_001, False, 9),
            (TSSVM, LogLossLeftRightImagery, 250, BNCI2014_001, False, 9),
            (SCNN, LogLossLeftRightImagery, 250, BNCI2014_001, True, 9),
            (DCNN, LogLossLeftRightImagery, 250, BNCI2014_001, True, 9),
        ]


Evaluation().evaluate()
