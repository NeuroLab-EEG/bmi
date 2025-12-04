"""
References:
    - https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation  # noqa: E501
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
from src.pipelines.euclidean import csp_lda, csp_svm, csp_svm_params
from src.pipelines.riemannian import ts_lr, ts_svm, ts_svm_params
from src.pipelines.deep_learning import scnn, dcnn


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

# Initialize spatial filter based evaluation
evaluation_sf = CrossSubjectEvaluation(
    paradigm=LogLossLeftRightImagery(),
    datasets=[
        PhysionetMI(),
        Lee2019_MI(),
        Cho2017(),
        Schirrmeister2017(),
        Shin2017A(),
        BNCI2014_001(),
    ],
    return_epochs=False,
    hdf5_path=data_path,
    n_jobs=-1,
    save_model=True,
    n_splits=5,
)

# Execute spatial filter based evaluation
evaluation_sf.process(
    dict(
        csp_lda=csp_lda,
        csp_svm=csp_svm,
        ts_lr=ts_lr,
        ts_svm=ts_svm,
    ),
    dict(
        csp_svm=csp_svm_params,
        ts_svm=ts_svm_params,
    ),
)

# Initialize deep learning evaluation
evaluation_dl = CrossSubjectEvaluation(
    paradigm=LogLossLeftRightImagery(),
    datasets=[
        PhysionetMI(),
        Lee2019_MI(),
        Cho2017(),
        Schirrmeister2017(),
        Shin2017A(),
        BNCI2014_001(),
    ],
    return_epochs=True,  # MNE epochs required
    hdf5_path=data_path,
    n_jobs=-1,
    save_model=True,
    n_splits=5,
)

# Execute deep learning based evaluation
evaluation_dl.process(
    dict(
        scnn=scnn,
        dcnn=dcnn,
    ),
    dict(),
)
