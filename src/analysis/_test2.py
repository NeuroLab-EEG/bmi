"""
References:
    - https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/noplot_load_model.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_within_session_splitter.html  # noqa: E501
    - https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectSplitter.html#moabb.evaluations.CrossSubjectSplitter  # noqa: E501
    - https://github.com/NeuroTechX/moabb/blob/develop/moabb/evaluations/evaluations.py#L581-L772  # noqa: E501
"""

import numpy as np
from os import path, getenv
from dotenv import load_dotenv
from pickle import load
from moabb.datasets import BNCI2014_001
from moabb.utils import set_download_dir
from moabb.evaluations import CrossSubjectSplitter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import get_scorer
from sklearn.preprocessing import LabelEncoder
from src.classifiers.paradigms import LogLossLeftRightImagery


# TODO: CSP+LDA diagnostics include
# TODO: CSP filter norms, LDA coefficients
# TODO: (optional) per class average covariance condition number

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

# Load dataset
dataset = BNCI2014_001()
paradigm = LogLossLeftRightImagery()
X, y, metadata = paradigm.get_data(dataset=dataset)

# Transform labels
le = LabelEncoder()
y = le.fit_transform(y)

# Extract metadata
groups = metadata.subject.values
sessions = metadata.session.values

# Define scoring rules
# TODO: Add more scoring rules
scorer = get_scorer(paradigm.scoring)

# Split dataset into same folds as evaluation
cv = CrossSubjectSplitter(cv_class=GroupKFold, **dict(n_splits=5))

for cv_ind, (train, test) in enumerate(cv.split(y, metadata)):
    # Load classifier from one fold of evaluation
    subject = groups[test[0]]
    with open(
        path.join(
            data_path,
            "Search_CrossSubject",
            "BNCI2014-001",
            str(subject),
            "csp_lda",
            f"fitted_model_{cv_ind}.pkl",
        ),
        "rb",
    ) as pickle_file:
        model = load(pickle_file)

    # Measure scores per session same as evaluation
    for session in np.unique(sessions[test]):
        ix = sessions[test] == session
        score = scorer(model, X[test[ix]], y[test[ix]])
        print(score)
