"""
References:
    - https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation
"""

from os import getenv
from dotenv import load_dotenv
from moabb.utils import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSubjectEvaluation
from bmi.classifiers.paradigms import LogLossLeftRightImagery
from bmi.classifiers.pipelines import csp_lda


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change data directory
set_download_dir(data_path)

# Initialize evaluation
evaluation = CrossSubjectEvaluation(
    paradigm=LogLossLeftRightImagery(),
    datasets=[
        BNCI2014_001()
    ],
    hdf5_path=data_path,
    overwrite=True,
    n_jobs=-1,
    save_model=True,
    n_splits=5
)

# Execute evaluation
evaluation.process({
    "csp_lda": csp_lda()
})
