"""
References:
    - https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation
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
    BNCI2014_001
)
from moabb.evaluations import CrossSubjectEvaluation
from classifiers.paradigms import LogLossLeftRightImagery
from classifiers.pipelines import pipelines, grids


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

# Initialize evaluation
evaluation = CrossSubjectEvaluation(
    paradigm=LogLossLeftRightImagery(),
    datasets=[
        BNCI2014_001()
    ],
    hdf5_path=data_path,
    n_jobs=-1,
    save_model=True,
    n_splits=5
)

# Execute evaluation
evaluation.process(pipelines(), grids())
