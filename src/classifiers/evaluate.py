"""
References:
    - https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation  # noqa: E501
    - https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html  # noqa: E501
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
from src.classifiers.paradigms import LogLossLeftRightImagery
from src.pipelines.euclidean import csp_lda, csp_svm
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
    (csp_lda(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, -1),
    (csp_svm(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, -1),
    (ts_lr(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, -1),
    (ts_svm(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, -1),
    (scnn(), LogLossLeftRightImagery(resample=160), PhysionetMI(), True, 1),
    (dcnn(), LogLossLeftRightImagery(resample=160), PhysionetMI(), True, 1),
    # Lee2019_MI
    (csp_lda(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, -1),
    (csp_svm(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, -1),
    (ts_lr(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, -1),
    (ts_svm(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, -1),
    (scnn(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True, 1),
    (dcnn(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True, 1),
    # Cho2017
    (csp_lda(), LogLossLeftRightImagery(resample=512), Cho2017(), False, -1),
    (csp_svm(), LogLossLeftRightImagery(resample=512), Cho2017(), False, -1),
    (ts_lr(), LogLossLeftRightImagery(resample=512), Cho2017(), False, -1),
    (ts_svm(), LogLossLeftRightImagery(resample=512), Cho2017(), False, -1),
    (scnn(), LogLossLeftRightImagery(resample=512), Cho2017(), True, 1),
    (dcnn(), LogLossLeftRightImagery(resample=512), Cho2017(), True, 1),
    # Schirrmeister2017
    (csp_lda(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, -1),
    (csp_svm(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, -1),
    (ts_lr(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, -1),
    (ts_svm(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, -1),
    (scnn(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True, 1),
    (dcnn(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True, 1),
    # Shin2017A
    (csp_lda(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, -1),
    (csp_svm(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, -1),
    (ts_lr(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, -1),
    (ts_svm(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, -1),
    (scnn(), LogLossLeftRightImagery(resample=200), Shin2017A(), True, 1),
    (dcnn(), LogLossLeftRightImagery(resample=200), Shin2017A(), True, 1),
    # BNCI2014_001
    (csp_lda(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, -1),
    (csp_svm(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, -1),
    (ts_lr(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, -1),
    (ts_svm(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, -1),
    (scnn(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), True, 1),
    (dcnn(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), True, 1),
]

for (pipeline, grid), paradigm, dataset, epochs, jobs in params:
    # Initialize evaluation
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        return_epochs=epochs,
        hdf5_path=data_path,
        n_jobs=jobs,
        save_model=True,
        n_splits=5,
    )

    # Execute evaluation
    evaluation.process(pipeline, grid)
