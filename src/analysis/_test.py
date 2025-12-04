"""
References:
    - https://github.com/NeuroTechX/moabb/blob/develop/moabb/analysis/results.py
"""

from os import getenv
from dotenv import load_dotenv
from moabb.analysis.results import Results
from moabb.evaluations import CrossSubjectEvaluation
from src.classifiers.paradigms import LogLossLeftRightImagery


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read results
df = Results(
    CrossSubjectEvaluation, LogLossLeftRightImagery, hdf5_path=data_path
).to_dataframe()
print(df)
