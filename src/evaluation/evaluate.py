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
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)
from moabb.evaluations import CrossSubjectEvaluation
from src.paradigm.paradigm import LogLossLeftRightImagery
from src.pipelines.raw_signal import csp_lda, csp_svm
from src.pipelines.riemannian import ts_lr, ts_svm
from src.pipelines.deep_learning import scnn, dcnn


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

# Define evaluation parameters
params = [
    # PhysionetMI
    (csp_lda(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False),
    (csp_svm(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False),
    (ts_lr(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False),
    (ts_svm(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False),
    (scnn(), LogLossLeftRightImagery(resample=160), PhysionetMI(), True),
    (dcnn(), LogLossLeftRightImagery(resample=160), PhysionetMI(), True),
    # Lee2019_MI
    (csp_lda(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    (csp_svm(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    (ts_lr(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    (ts_svm(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False),
    (scnn(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True),
    (dcnn(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True),
    # Cho2017
    (csp_lda(), LogLossLeftRightImagery(resample=512), Cho2017(), False),
    (csp_svm(), LogLossLeftRightImagery(resample=512), Cho2017(), False),
    (ts_lr(), LogLossLeftRightImagery(resample=512), Cho2017(), False),
    (ts_svm(), LogLossLeftRightImagery(resample=512), Cho2017(), False),
    (scnn(), LogLossLeftRightImagery(resample=512), Cho2017(), True),
    (dcnn(), LogLossLeftRightImagery(resample=512), Cho2017(), True),
    # Schirrmeister2017
    (csp_lda(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    (csp_svm(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    (ts_lr(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    (ts_svm(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False),
    (scnn(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True),
    (dcnn(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True),
    # Shin2017A
    (csp_lda(), LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    (csp_svm(), LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    (ts_lr(), LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    (ts_svm(), LogLossLeftRightImagery(resample=200), Shin2017A(), False),
    (scnn(), LogLossLeftRightImagery(resample=200), Shin2017A(), True),
    (dcnn(), LogLossLeftRightImagery(resample=200), Shin2017A(), True),
    # BNCI2014_001
    (csp_lda(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False),
    (csp_svm(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False),
    (ts_lr(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False),
    (ts_svm(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False),
    (scnn(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), True),
    (dcnn(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), True),
]

for (pipeline, grid), paradigm, dataset, epochs in params:
    # Initialize evaluation
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        return_epochs=epochs,
        hdf5_path=data_path,
        save_model=True,
        n_splits=5,
    )

    # Execute evaluation
    evaluation.process(pipeline, grid)
